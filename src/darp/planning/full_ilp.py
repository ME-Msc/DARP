"""Full-tree baseline for the paper's ILP policy-tree objective."""

# TODO(phase-9.1): Replace generated observation branches with full stochastic
# observation support before benchmark-scale CC-POMDP experiments.

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter

from darp.adapter.runtime import PyRDDLGymRuntime, _json_ready
from darp.ilp.gurobi import GurobiILPSolver, GurobiUnavailableError, gurobi_available
from darp.ilp.model import ILPSolveResult
from darp.model.and_or_tree import ANDORSearchInterface
from darp.model.duration import HistoryDurationEvaluator
from darp.planning.ilp_tree import build_generated_full_tree_ilp
from darp.planning.expand import expand_frontier_item
from darp.planning.preprocess import FrontierItem, preprocess_search_tree
from darp.planning.rollout import (
    ActionDecision,
    _RuntimeDeadlineExceeded,
    _raise_if_deadline_expired,
)


@dataclass
class FullILPPlanner:
    """Evaluate a complete finite AND-OR tree baseline. / 评估完整有限 AND-OR tree baseline。"""

    lookahead_depth: int = 4
    use_gurobi: bool = True
    require_gurobi: bool = False
    risk_budget: float | None = None
    name: str = "full-ilp-gurobi"
    last_ilp_result: ILPSolveResult | None = field(default=None, init=False)

    def choose_action(
        self,
        runtime: PyRDDLGymRuntime,
        interface: ANDORSearchInterface,
        duration_evaluator: HistoryDurationEvaluator,
        *,
        remaining_depth: int,
        time_budget_ms: float | None = None,
    ) -> ActionDecision:
        r"""Choose the root action by solving the full generated tree.

        Paper correspondence:

        - The paper's full ILP baseline builds the full history tree and
          chooses binary policy variables with constraints equivalent to:

          .. math::

             \sum_{a \in A(root)} x_{root,a} = 1,
             \qquad
             \sum_{a \in A(q)} x_{q,a} = x_{parent(q)}

        - Phase 8 encodes the generated tree as a Gurobi binary ILP. The
          recursive dynamic-programming value is still used as a readable
          fallback and as root-action diagnostics:

          .. math::

             V(q) = \max_a \left[u(q,a) + \gamma V(qao)\right].

        / 通过 Gurobi full-ILP 选择根动作；递归 DP 仍作为诊断和缺少 Gurobi 时的 fallback。
        """

        started_at = perf_counter()
        if self.lookahead_depth < 1:
            raise ValueError("lookahead_depth must be at least 1.")
        if remaining_depth < 1:
            raise ValueError("remaining_depth must be at least 1.")
        if time_budget_ms is not None and time_budget_ms < 0.0:
            raise ValueError("time_budget_ms must be non-negative.")

        deadline = None if time_budget_ms is None else started_at + time_budget_ms / 1000.0
        depth = max(1, min(self.lookahead_depth, remaining_depth))
        tree = preprocess_search_tree(runtime, interface)
        action_values: dict[str, float] = {}
        best_item = tree.frontier[0]
        best_value = float("-inf")
        complete = True
        fallback_reason = None
        cache: dict[tuple[str, tuple[str, ...], tuple[str, ...], int], float] = {}

        try:
            for item in tree.frontier:
                _raise_if_deadline_expired(deadline)
                value = self.frontier_value(
                    item,
                    interface,
                    duration_evaluator,
                    depth=depth,
                    deadline=deadline,
                    cache=cache,
                )
                action_values[item.action_label] = value
                if value > best_value:
                    best_item = item
                    best_value = value
        except _RuntimeDeadlineExceeded as exc:
            complete = False
            fallback_reason = str(exc)
            if best_value == float("-inf"):
                best_value = 0.0
                action_values.setdefault(best_item.action_label, best_value)

        if self.use_gurobi:
            try:
                if not gurobi_available():
                    raise GurobiUnavailableError("gurobipy is required for DARP Phase 8 ILP solving.")
                ilp_tree = build_generated_full_tree_ilp(
                    runtime.clone(),
                    interface,
                    duration_evaluator,
                    depth=depth,
                    risk_budget=self.risk_budget,
                    deadline=deadline,
                )
                self.last_ilp_result = GurobiILPSolver().solve(
                    ilp_tree.spec,
                    time_limit_ms=_remaining_time_budget_ms(started_at, time_budget_ms),
                )
                selected_root = next(
                    (
                        var_id
                        for var_id in self.last_ilp_result.selected_variables
                        if var_id in ilp_tree.root_variable_ids
                    ),
                    None,
                )
                if selected_root is not None:
                    best_item = ilp_tree.variable_items[selected_root]
                    best_value = action_values.get(best_item.action_label, best_value)
                    if not self.last_ilp_result.is_optimal:
                        complete = False
                        fallback_reason = self.last_ilp_result.message
                elif self.require_gurobi:
                    raise RuntimeError("Gurobi full-tree ILP did not select a root action.")
                else:
                    complete = False
                    fallback_reason = "Gurobi full-tree ILP did not select a root action; used DP fallback"
            except GurobiUnavailableError as exc:
                self.last_ilp_result = None
                if self.require_gurobi:
                    raise
                complete = False
                fallback_reason = str(exc)
            except _RuntimeDeadlineExceeded as exc:
                complete = False
                fallback_reason = str(exc)

        elapsed_ms = (perf_counter() - started_at) * 1000.0
        timed_out = deadline is not None and perf_counter() > deadline
        if timed_out and complete:
            complete = False
            fallback_reason = "deadline expired after full-tree decision"
        return ActionDecision(
            action=dict(best_item.node.metadata["assignment"]),
            label=best_item.action_label,
            value=best_value,
            action_values=action_values,
            remaining_depth=depth,
            elapsed_ms=elapsed_ms,
            time_budget_ms=time_budget_ms,
            complete=complete,
            timed_out=timed_out,
            fallback_reason=fallback_reason,
        )

    def frontier_value(
        self,
        item: FrontierItem,
        interface: ANDORSearchInterface,
        duration_evaluator: HistoryDurationEvaluator,
        *,
        depth: int,
        deadline: float | None = None,
        cache: dict[tuple[str, tuple[str, ...], tuple[str, ...], int], float] | None = None,
    ) -> float:
        r"""Return the best value from an action-history frontier item.

        The recurrence mirrors the paper tree objective without introducing ILP
        variables yet:

        .. math::

           Q(q,a) = u_q + \gamma \max_{a'} Q(qoa')

        where `Expand` supplies :math:`u_q`, :math:`\tau(q)`, and terminal
        checks. / 返回 action-history 的递归价值，`Expand` 提供 :math:`u_q`
        与 :math:`\tau(q)`。
        """

        _raise_if_deadline_expired(deadline)
        memo = cache if cache is not None else {}
        key = _frontier_cache_key(item, depth)
        if key in memo:
            return memo[key]
        expanded = expand_frontier_item(item, interface, duration_evaluator)
        value = expanded.metrics.utility
        if depth > 1 and expanded.metrics.should_expand and expanded.child_frontier:
            future = max(
                self.frontier_value(
                    child,
                    interface,
                    duration_evaluator,
                    depth=depth - 1,
                    deadline=deadline,
                    cache=memo,
                )
                for child in expanded.child_frontier
            )
            value += item.parent_runtime.discount * future
        memo[key] = value
        return value


def _frontier_cache_key(item: FrontierItem, depth: int) -> tuple[str, tuple[str, ...], tuple[str, ...], int]:
    """Return a memoization key for one frontier item. / 返回 frontier item 的缓存键。"""
    state = tuple(sorted((str(key), repr(_json_ready(value))) for key, value in item.parent_runtime.state.items()))
    return (repr(state), item.node.history.actions, item.node.history.observations, depth)


def _remaining_time_budget_ms(started_at: float, time_budget_ms: float | None) -> float | None:
    """Return remaining milliseconds under an outer deadline. / 返回外层 deadline 下剩余毫秒数。"""
    if time_budget_ms is None:
        return None
    return max(0.0, time_budget_ms - (perf_counter() - started_at) * 1000.0)
