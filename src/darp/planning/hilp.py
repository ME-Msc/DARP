"""HILP-style partial frontier search for the paper algorithm."""

# TODO(phase-8.1): Replace the heuristic p-ILP proxy with Gurobi-selected
# frontier variables once ILP encoders are implemented.
# TODO(phase-8.3): Add warm-start transfer from one p-ILP solve to the next.

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter

from darp.adapter.runtime import PyRDDLGymRuntime
from darp.model.and_or_tree import ANDORSearchInterface
from darp.model.duration import HistoryDurationEvaluator
from darp.planning.expand import ExpandedAction, expand_frontier_item
from darp.planning.full_ilp import FullILPPlanner
from darp.planning.preprocess import FrontierItem, preprocess_search_tree
from darp.planning.rollout import (
    ActionDecision,
    _RuntimeDeadlineExceeded,
    _raise_if_deadline_expired,
)


@dataclass(frozen=True)
class HILPSearchStats:
    """Summarize one HILP-style partial-tree search. / 汇总一次 HILP 风格 partial-tree 搜索。"""

    iterations: int
    expanded_count: int
    frontier_count: int
    selected_frontier: tuple[str, ...]
    used_gurobi: bool = False


@dataclass
class HILPPlanner:
    """Run paper Algorithm 3 style frontier expansion. / 运行论文 Algorithm 3 风格的 frontier expansion。"""

    lookahead_depth: int = 4
    max_iterations: int = 4
    frontier_width: int = 1
    name: str = "hilp-partial-tree"
    last_stats: HILPSearchStats | None = field(default=None, init=False)

    def choose_action(
        self,
        runtime: PyRDDLGymRuntime,
        interface: ANDORSearchInterface,
        duration_evaluator: HistoryDurationEvaluator,
        *,
        remaining_depth: int,
        time_budget_ms: float | None = None,
    ) -> ActionDecision:
        r"""Choose an action with HILP-style partial-tree expansion.

        Paper correspondence:

        - Algorithm 3 keeps three sets: expanded histories :math:`E`, frontier
          histories :math:`F`, and non-expanded descendants :math:`N`.
        - Each iteration solves a partial ILP over :math:`E \cup F`, selects
          frontier nodes with positive policy mass, expands them, and repeats.
        - This Phase 7 implementation keeps the same :math:`E/F` bookkeeping
          and calls the Phase 7 `Expand` operation, but uses the full-tree
          dynamic-programming evaluator as a readable p-ILP scoring proxy.
          Phase 8 replaces that proxy with a Gurobi p-ILP model.

        / 使用 HILP 的 :math:`E/F` frontier 更新框架选择动作；当前以可读的 DP
        分数近似 p-ILP，Phase 8 会替换成 Gurobi p-ILP。
        """

        started_at = perf_counter()
        if self.lookahead_depth < 1:
            raise ValueError("lookahead_depth must be at least 1.")
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be at least 1.")
        if self.frontier_width < 1:
            raise ValueError("frontier_width must be at least 1.")
        if remaining_depth < 1:
            raise ValueError("remaining_depth must be at least 1.")
        if time_budget_ms is not None and time_budget_ms < 0.0:
            raise ValueError("time_budget_ms must be non-negative.")

        deadline = None if time_budget_ms is None else started_at + time_budget_ms / 1000.0
        depth = max(1, min(self.lookahead_depth, remaining_depth))
        full_tree = FullILPPlanner(lookahead_depth=depth)
        tree = preprocess_search_tree(runtime, interface)
        frontier = list(tree.frontier)
        expanded: list[ExpandedAction] = []
        selected_labels: list[str] = []
        complete = True
        fallback_reason = None

        try:
            for iteration in range(self.max_iterations):
                _raise_if_deadline_expired(deadline)
                if not frontier:
                    self.last_stats = HILPSearchStats(
                        iterations=iteration,
                        expanded_count=len(expanded),
                        frontier_count=0,
                        selected_frontier=tuple(selected_labels),
                    )
                    break
                scored = self._score_frontier(
                    frontier,
                    full_tree,
                    interface,
                    duration_evaluator,
                    depth=depth,
                    deadline=deadline,
                )
                selected = [item for _, item in scored[: self.frontier_width]]
                for item in selected:
                    if item not in frontier:
                        continue
                    frontier.remove(item)
                    selected_labels.append(item.node.history.label())
                    expanded_item = expand_frontier_item(item, interface, duration_evaluator)
                    expanded.append(expanded_item)
                    frontier.extend(expanded_item.child_frontier)
            else:
                self.last_stats = HILPSearchStats(
                    iterations=self.max_iterations,
                    expanded_count=len(expanded),
                    frontier_count=len(frontier),
                    selected_frontier=tuple(selected_labels),
                )
        except _RuntimeDeadlineExceeded as exc:
            complete = False
            fallback_reason = str(exc)
            self.last_stats = HILPSearchStats(
                iterations=len(selected_labels),
                expanded_count=len(expanded),
                frontier_count=len(frontier),
                selected_frontier=tuple(selected_labels),
            )

        decision = full_tree.choose_action(
            runtime,
            interface,
            duration_evaluator,
            remaining_depth=depth,
            time_budget_ms=_remaining_time_budget_ms(started_at, time_budget_ms),
        )
        elapsed_ms = (perf_counter() - started_at) * 1000.0
        timed_out = deadline is not None and perf_counter() > deadline
        if not complete or timed_out:
            return ActionDecision(
                action=decision.action,
                label=decision.label,
                value=decision.value,
                action_values=decision.action_values,
                remaining_depth=decision.remaining_depth,
                elapsed_ms=elapsed_ms,
                time_budget_ms=time_budget_ms,
                complete=False,
                timed_out=timed_out or decision.timed_out,
                fallback_reason=fallback_reason or decision.fallback_reason,
            )
        return ActionDecision(
            action=decision.action,
            label=decision.label,
            value=decision.value,
            action_values=decision.action_values,
            remaining_depth=decision.remaining_depth,
            elapsed_ms=elapsed_ms,
            time_budget_ms=time_budget_ms,
            complete=decision.complete,
            timed_out=decision.timed_out,
            fallback_reason=decision.fallback_reason,
        )

    def _score_frontier(
        self,
        frontier: list[FrontierItem],
        full_tree: FullILPPlanner,
        interface: ANDORSearchInterface,
        duration_evaluator: HistoryDurationEvaluator,
        *,
        depth: int,
        deadline: float | None,
    ) -> list[tuple[float, FrontierItem]]:
        r"""Score frontier histories as the current p-ILP proxy.

        In Algorithm 3, this step is where the p-ILP returns selected frontier
        histories. Phase 7 ranks them by the same generated-tree value used by
        the full-tree baseline:

        .. math::

           score(q) \approx Q(q,a)

        / 对 frontier history 进行当前 p-ILP 代理打分。
        """

        cache: dict[tuple[str, tuple[str, ...], tuple[str, ...], int], float] = {}
        scored = [
            (
                full_tree.frontier_value(
                    item,
                    interface,
                    duration_evaluator,
                    depth=depth,
                    deadline=deadline,
                    cache=cache,
                ),
                item,
            )
            for item in frontier
        ]
        return sorted(scored, key=lambda pair: pair[0], reverse=True)


def _remaining_time_budget_ms(started_at: float, time_budget_ms: float | None) -> float | None:
    """Return remaining milliseconds under an outer deadline. / 返回外层 deadline 下剩余毫秒数。"""
    if time_budget_ms is None:
        return None
    return max(0.0, time_budget_ms - (perf_counter() - started_at) * 1000.0)
