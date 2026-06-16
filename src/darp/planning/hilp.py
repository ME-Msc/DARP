"""HILP-style partial frontier search for the paper algorithm."""

# TODO(phase-9.1): Add warm-start transfer and benchmark-scale pruning for
# larger exact finite kernels in HILP experiments.

from __future__ import annotations

from dataclasses import dataclass, field, replace
from time import perf_counter
from typing import Mapping

from darp.adapter.exact import StateKey
from darp.adapter.runtime import PyRDDLGymRuntime
from darp.ilp.gurobi import GurobiILPSolver
from darp.ilp.model import ILPSolveResult
from darp.model.and_or_tree import ANDORSearchInterface
from darp.model.duration import HistoryDurationEvaluator
from darp.planning.expand import ExpandedAction, expand_frontier_item
from darp.planning.ilp_tree import (
    Algorithm1ExpansionRecord,
    PolicyTreeILP,
    _action_var_id,
    build_partial_tree_ilp,
)
from darp.planning.preprocess import FrontierItem, initialize_root_frontier
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
    partial_variable_count: int = 0


@dataclass
class HILPPlanner:
    """Run paper Algorithm 3 style frontier expansion. / 运行论文 Algorithm 3 风格的 frontier expansion。"""

    lookahead_depth: int = 4
    max_iterations: int = 4
    frontier_width: int = 1
    risk_budget: float | None = None
    name: str = "hilp-partial-tree"
    last_stats: HILPSearchStats | None = field(default=None, init=False)
    last_ilp_result: ILPSolveResult | None = field(default=None, init=False)
    last_partial_tree: PolicyTreeILP | None = field(default=None, init=False)

    def choose_action(
        self,
        runtime: PyRDDLGymRuntime,
        interface: ANDORSearchInterface,
        duration_evaluator: HistoryDurationEvaluator,
        *,
        remaining_depth: int,
        root_belief: Mapping[StateKey, float] | None = None,
        time_budget_ms: float | None = None,
    ) -> ActionDecision:
        r"""Choose an action with HILP-style partial-tree expansion.

        Paper correspondence:

        - Algorithm 3 keeps three sets: expanded histories $$E$$, frontier
          histories $$F$$, and non-expanded descendants $$N$$.
        - Each iteration solves a partial ILP over $$E \cup F$$, selects
          frontier nodes with positive policy mass, expands them, and repeats.
        - DARP's partial ILP keeps the same Definition 3.1 root/flow rows as
          full-ILP for histories in $$E$$.  Histories in $$F$$ are frontier
          leaves: they have exact one-step $$u_q,r_q$$ constants, but no
          child-flow rows yet.

        / 使用 HILP 的 $$E/F$$ frontier 更新框架；每轮和最终 action selection
        都来自当前 partial-tree ILP，不再回退调用 full-ILP。
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
        depth_limit = max(1, min(self.lookahead_depth, remaining_depth))
        root_frontier = initialize_root_frontier(runtime, interface, root_belief=root_belief)
        frontier: dict[str, FrontierItem] = {
            _action_var_id(item): item
            for item in root_frontier.frontier
        }
        expanded_records: dict[str, Algorithm1ExpansionRecord] = {}
        frontier_expansions: dict[str, ExpandedAction] = {}
        heuristic_cache: dict[tuple[object, ...], float] = {}
        selected_labels: list[str] = []
        complete = True
        fallback_reason = None
        final_tree: PolicyTreeILP | None = None
        final_result: ILPSolveResult | None = None

        try:
            for iteration in range(self.max_iterations):
                _raise_if_deadline_expired(deadline)
                if not frontier:
                    self.last_stats = HILPSearchStats(
                        iterations=iteration,
                        expanded_count=len(expanded_records),
                        frontier_count=0,
                        selected_frontier=tuple(selected_labels),
                        used_gurobi=final_result is not None,
                        partial_variable_count=len(final_tree.spec.variables) if final_tree else 0,
                    )
                    break
                final_tree, final_result = self._solve_partial_policy_ilp(
                    runtime,
                    interface,
                    duration_evaluator,
                    expanded_records=tuple(expanded_records.values()),
                    frontier=tuple(frontier.values()),
                    frontier_expansions=frontier_expansions,
                    heuristic_cache=heuristic_cache,
                    depth_limit=depth_limit,
                    root_belief=root_belief,
                    deadline=deadline,
                    started_at=started_at,
                    time_budget_ms=time_budget_ms,
                )
                selected = self._selected_expandable_frontier(
                    final_result,
                    final_tree,
                    frontier,
                    frontier_expansions,
                    depth_limit=depth_limit,
                )
                if not selected:
                    break
                for var_id, item in selected:
                    expanded_item = frontier_expansions[var_id]
                    del frontier[var_id]
                    selected_labels.append(item.node.history.label())
                    expanded_records[var_id] = Algorithm1ExpansionRecord(
                        var_id=var_id,
                        item=item,
                        expanded=expanded_item,
                        continues=bool(expanded_item.child_frontier),
                    )
                    for child in expanded_item.child_frontier:
                        child_var_id = _action_var_id(child)
                        if child_var_id not in expanded_records and child_var_id not in frontier:
                            frontier[child_var_id] = child
            else:
                self.last_stats = HILPSearchStats(
                    iterations=self.max_iterations,
                    expanded_count=len(expanded_records),
                    frontier_count=len(frontier),
                    selected_frontier=tuple(selected_labels),
                    used_gurobi=final_result is not None,
                    partial_variable_count=len(final_tree.spec.variables) if final_tree else 0,
                )
        except _RuntimeDeadlineExceeded as exc:
            complete = False
            fallback_reason = str(exc)
            self.last_stats = HILPSearchStats(
                iterations=len(selected_labels),
                expanded_count=len(expanded_records),
                frontier_count=len(frontier),
                selected_frontier=tuple(selected_labels),
                used_gurobi=final_result is not None,
                partial_variable_count=len(final_tree.spec.variables) if final_tree else 0,
            )

        final_tree, final_result = self._solve_partial_policy_ilp(
            runtime,
            interface,
            duration_evaluator,
            expanded_records=tuple(expanded_records.values()),
            frontier=tuple(frontier.values()),
            frontier_expansions=frontier_expansions,
            heuristic_cache=heuristic_cache,
            depth_limit=depth_limit,
            root_belief=root_belief,
            deadline=deadline,
            started_at=started_at,
            time_budget_ms=time_budget_ms,
        )
        self.last_partial_tree = final_tree
        self.last_ilp_result = final_result
        if self.last_stats is None:
            self.last_stats = HILPSearchStats(
                iterations=len(selected_labels),
                expanded_count=len(expanded_records),
                frontier_count=len(frontier),
                selected_frontier=tuple(selected_labels),
                used_gurobi=True,
                partial_variable_count=len(final_tree.spec.variables),
            )
        selected_root = _selected_root_variable(final_result, final_tree)
        if selected_root is None:
            raise RuntimeError(
                "Gurobi HILP partial-tree ILP did not select a root action. "
                f"status={final_result.status}"
            )
        selected_item = final_tree.variable_items[selected_root]
        elapsed_ms = (perf_counter() - started_at) * 1000.0
        timed_out = deadline is not None and perf_counter() > deadline
        return ActionDecision(
            action=dict(selected_item.node.metadata["assignment"]),
            label=selected_item.action_label,
            value=float(final_result.objective_value or 0.0),
            action_values=_root_objective_values(final_tree),
            remaining_depth=depth_limit,
            elapsed_ms=elapsed_ms,
            time_budget_ms=time_budget_ms,
            complete=complete and final_result.is_optimal and not timed_out,
            timed_out=timed_out,
            fallback_reason=fallback_reason or final_result.message,
        )

    def _solve_partial_policy_ilp(
        self,
        runtime: PyRDDLGymRuntime,
        interface: ANDORSearchInterface,
        duration_evaluator: HistoryDurationEvaluator,
        *,
        expanded_records: tuple[Algorithm1ExpansionRecord, ...],
        frontier: tuple[FrontierItem, ...],
        frontier_expansions: dict[str, ExpandedAction],
        heuristic_cache: dict[tuple[object, ...], float],
        depth_limit: int,
        root_belief: Mapping[StateKey, float] | None,
        deadline: float | None,
        started_at: float,
        time_budget_ms: float | None,
    ) -> tuple[PolicyTreeILP, ILPSolveResult]:
        r"""Solve Algorithm 3's current partial-tree p-ILP.

        For every frontier history $$q\in F$$, DARP first calls Algorithm 2
        only far enough to obtain the leaf constants $$u_q,r_q$$.  The partial
        ILP then decides both the current root action and which frontier leaves
        carry positive policy mass.

        / 求解当前 $$E\cup F$$ partial-tree p-ILP；frontier 只作为叶子，
        不会触发完整 horizon 枚举。
        """

        _raise_if_deadline_expired(deadline)
        frontier_records = tuple(
            _frontier_leaf_record(
                item,
                interface,
                duration_evaluator,
                frontier_expansions,
                heuristic_cache,
                depth_limit=depth_limit,
                deadline=deadline,
            )
            for item in frontier
        )
        partial_ilp = build_partial_tree_ilp(
            runtime=runtime,
            interface=interface,
            expanded_records=expanded_records,
            frontier_records=frontier_records,
            risk_budget=self.risk_budget,
            root_belief=root_belief,
        )
        result = GurobiILPSolver().solve(
            partial_ilp.spec,
            time_limit_ms=_remaining_time_budget_ms(started_at, time_budget_ms),
        )
        return partial_ilp, result

    def _selected_expandable_frontier(
        self,
        result: ILPSolveResult,
        partial_tree: PolicyTreeILP,
        frontier: Mapping[str, FrontierItem],
        frontier_expansions: Mapping[str, ExpandedAction],
        *,
        depth_limit: int,
    ) -> tuple[tuple[str, FrontierItem], ...]:
        """Return selected frontier leaves that may be expanded. / 返回解中选中且可继续展开的 frontier leaf。"""
        selected: list[tuple[str, FrontierItem]] = []
        frontier_ids = set(partial_tree.frontier_variable_ids)
        for var_id in result.selected_variables:
            if var_id not in frontier_ids or var_id not in frontier:
                continue
            item = frontier[var_id]
            expanded = frontier_expansions[var_id]
            if item.node.history.depth >= depth_limit:
                continue
            if not expanded.child_frontier:
                continue
            selected.append((var_id, item))
            if len(selected) >= self.frontier_width:
                break
        return tuple(selected)


def _frontier_leaf_record(
    item: FrontierItem,
    interface: ANDORSearchInterface,
    duration_evaluator: HistoryDurationEvaluator,
    frontier_expansions: dict[str, ExpandedAction],
    heuristic_cache: dict[tuple[object, ...], float],
    *,
    depth_limit: int,
    deadline: float | None,
) -> Algorithm1ExpansionRecord:
    r"""Return a frontier leaf record with exact heuristic constants.

    The frontier record stays a leaf in the p-ILP, but its objective uses a
    bounded exact heuristic:

    $$\hat V(q)=u_q+\sum_o \max_a \hat V(qoa).$$

    / frontier 在 p-ILP 中仍是叶子，但目标系数使用受深度限制的 exact
    启发式估值，而不是只看一步 reward。
    """
    _raise_if_deadline_expired(deadline)
    var_id = _action_var_id(item)
    expanded = _cached_expand(item, interface, duration_evaluator, frontier_expansions)
    heuristic_value = _exact_frontier_heuristic_value(
        item,
        interface,
        duration_evaluator,
        frontier_expansions,
        heuristic_cache,
        depth_limit=depth_limit,
        deadline=deadline,
    )
    leaf_expanded = replace(
        expanded,
        metrics=replace(expanded.metrics, utility=heuristic_value),
    )
    return Algorithm1ExpansionRecord(
        var_id=var_id,
        item=item,
        expanded=leaf_expanded,
        continues=False,
    )


def _exact_frontier_heuristic_value(
    item: FrontierItem,
    interface: ANDORSearchInterface,
    duration_evaluator: HistoryDurationEvaluator,
    frontier_expansions: dict[str, ExpandedAction],
    heuristic_cache: dict[tuple[object, ...], float],
    *,
    depth_limit: int,
    deadline: float | None,
) -> float:
    r"""Return a bounded exact leaf heuristic for HILP frontier nodes.

    The recursion is local to HILP's current lookahead cap and uses Algorithm 2
    exact branches:

    $$\hat V(q)=u_q+\sum_o \max_a \hat V(qoa),\qquad depth(q)<d.$$

    It is a heuristic for partial-tree leaves, not the full-ILP baseline.

    / 只在 HILP lookahead 上界内递归估计 frontier 叶子价值；它帮助 partial
    ILP 看到未来 reward，但不会枚举完整 horizon。
    """
    _raise_if_deadline_expired(deadline)
    cache_key = _heuristic_cache_key(item, duration_evaluator, depth_limit)
    if cache_key in heuristic_cache:
        return heuristic_cache[cache_key]
    expanded = _cached_expand(item, interface, duration_evaluator, frontier_expansions)
    value = expanded.metrics.utility
    if item.node.history.depth < depth_limit:
        for branch in expanded.observation_frontiers:
            if not branch.should_expand or not branch.child_frontier:
                continue
            value += max(
                _exact_frontier_heuristic_value(
                    child,
                    interface,
                    duration_evaluator,
                    frontier_expansions,
                    heuristic_cache,
                    depth_limit=depth_limit,
                    deadline=deadline,
                )
                for child in branch.child_frontier
            )
    heuristic_cache[cache_key] = value
    return value


def _heuristic_cache_key(
    item: FrontierItem,
    duration_evaluator: HistoryDurationEvaluator,
    depth_limit: int,
) -> tuple[object, ...]:
    """Return a semantic cache key for equivalent HILP leaf subproblems. / 为等价 HILP 叶子子问题返回语义缓存键。"""
    prefix_action_count = min(len(item.duration_beliefs), len(item.node.history.actions))
    prefix_progress = duration_evaluator.progress_for_actions(
        item.node.history.actions[:prefix_action_count],
        item.duration_beliefs,
    )
    return (
        depth_limit - item.node.history.depth,
        item.action_label,
        round(float(item.rho), 12),
        round(prefix_progress.mean, 12),
        round(prefix_progress.variance, 12),
        _belief_key(item.belief),
        _belief_key(item.safe_belief),
    )


def _belief_key(belief: Mapping[StateKey, float] | None) -> tuple[tuple[str, float], ...]:
    """Return a stable cache key for a belief distribution. / 返回 belief 分布的稳定缓存键。"""
    if not belief:
        return ()
    return tuple(
        (repr(state), round(float(probability), 12))
        for state, probability in sorted(belief.items(), key=lambda pair: repr(pair[0]))
        if abs(float(probability)) > 1e-15
    )


def _cached_expand(
    item: FrontierItem,
    interface: ANDORSearchInterface,
    duration_evaluator: HistoryDurationEvaluator,
    expansions: dict[str, ExpandedAction],
) -> ExpandedAction:
    """Return Algorithm 2 expansion from a small cache. / 从小缓存返回 Algorithm 2 expansion。"""
    var_id = _action_var_id(item)
    if var_id not in expansions:
        expansions[var_id] = expand_frontier_item(item, interface, duration_evaluator)
    return expansions[var_id]


def _selected_root_variable(result: ILPSolveResult, tree: PolicyTreeILP) -> str | None:
    """Return the root action selected by the partial ILP. / 返回 partial ILP 选中的 root action。"""
    root_ids = set(tree.root_variable_ids)
    return next((var_id for var_id in result.selected_variables if var_id in root_ids), None)


def _root_objective_values(tree: PolicyTreeILP) -> dict[str, float]:
    """Return root-action objective coefficients for diagnostics. / 返回根 action 目标系数用于诊断。"""
    values: dict[str, float] = {}
    for var_id in tree.root_variable_ids:
        item = tree.variable_items[var_id]
        values[item.action_label] = float(tree.spec.objective.get(var_id, 0.0))
    return values


def _remaining_time_budget_ms(started_at: float, time_budget_ms: float | None) -> float | None:
    """Return remaining milliseconds under an outer deadline. / 返回外层 deadline 下剩余毫秒数。"""
    if time_budget_ms is None:
        return None
    return max(0.0, time_budget_ms - (perf_counter() - started_at) * 1000.0)
