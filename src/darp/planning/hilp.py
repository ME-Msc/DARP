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
from darp.planning.rollout import ActionDecision


@dataclass(frozen=True)
class HILPSearchStats:
    """Summarize one HILP-style partial-tree search. / 汇总一次 HILP 风格 partial-tree 搜索。"""

    iterations: int
    expanded_count: int
    frontier_count: int
    selected_frontier: tuple[str, ...]
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
    ) -> ActionDecision:
        r"""Choose an action with HILP-style partial-tree expansion.

        Paper correspondence:

        - Algorithm 3 keeps three sets: expanded action histories $$E$$,
          frontier action histories $$F$$, and not-yet-generated descendants
          $$N$$.
        - Each iteration solves a partial ILP over $$E \cup F$$, selects
          frontier nodes with positive policy mass, expands them, and repeats.
        - DARP's partial ILP keeps the same Definition 3.1 root/flow rows as
          full-ILP for histories in $$E$$.  Histories in $$F$$ are frontier
          leaves: they have exact one-step $$u_q,r_q$$ constants, but no
          child-flow rows yet.
        - The CC-POMDP time budget is the domain horizon inside
          `duration_evaluator`; it is consumed by action durations through
          $$\tau(q)$$.  It is not Python wall-clock runtime.

        / 使用 HILP 的 $$E/F$$ frontier 更新框架；每轮和最终 action selection
        都来自当前 partial-tree ILP，不再回退调用 full-ILP。规划问题中的时间预算
        由 `duration_evaluator` 的 action duration 和 $$\tau(q)$$ 表示，不使用
        Python 运行时间。
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

        # HILP does not use this as a full-tree cutoff. It only bounds the
        # exact terminal heuristic attached to frontier leaves in $$F$$.
        # HILP 不把它当作 full-tree 截断；它只限制 $$F$$ 叶子的启发式估值深度。
        heuristic_depth = max(1, min(self.lookahead_depth, remaining_depth))
        root_frontier = initialize_root_frontier(runtime, interface, root_belief=root_belief)
        # Algorithm 3: $$F$$ starts from all root action histories. / 初始 frontier。
        frontier_f: dict[str, FrontierItem] = {
            _action_var_id(item): item
            for item in root_frontier.frontier
        }
        # Algorithm 3: $$E$$ stores histories that have already been expanded.
        # Algorithm 3：$$E$$ 保存已经调用过 Expand 的 action histories。
        expanded_e: dict[str, Algorithm1ExpansionRecord] = {}
        frontier_expansions: dict[str, ExpandedAction] = {}
        heuristic_cache: dict[tuple[object, ...], tuple[float, float]] = {}
        selected_frontier_labels: list[str] = []
        partial_tree: PolicyTreeILP | None = None
        partial_result: ILPSolveResult | None = None
        expansion_rounds = 0
        needs_final_solve = True

        for _ in range(self.max_iterations):
            if not frontier_f:
                break
            partial_tree, partial_result = self._solve_partial_policy_ilp(
                runtime,
                interface,
                duration_evaluator,
                expanded_records=tuple(expanded_e.values()),
                frontier=tuple(frontier_f.values()),
                frontier_expansions=frontier_expansions,
                heuristic_cache=heuristic_cache,
                heuristic_depth=heuristic_depth,
                root_belief=root_belief,
            )
            needs_final_solve = False
            # Algorithm 3: solve p-ILP on $$E\cup F$$, then expand selected
            # positive-mass frontier histories.  Descendants behind $$F$$ are
            # the implicit $$N$$ set until Expand materializes them.
            # Algorithm 3：对 $$E\cup F$$ 解 p-ILP，再展开被选中的 frontier；
            # 尚未生成的后代就是隐式 $$N$$。
            selected = self._selected_expandable_frontier(
                partial_result,
                partial_tree,
                frontier_f,
                frontier_expansions,
                heuristic_depth=heuristic_depth,
            )
            if not selected:
                break
            expansion_rounds += 1
            for var_id, item in selected:
                expanded_item = frontier_expansions[var_id]
                del frontier_f[var_id]
                selected_frontier_labels.append(item.node.history.label())
                expanded_e[var_id] = Algorithm1ExpansionRecord(
                    var_id=var_id,
                    item=item,
                    expanded=expanded_item,
                    continues=bool(expanded_item.child_frontier),
                )
                for child in expanded_item.child_frontier:
                    child_var_id = _action_var_id(child)
                    if child_var_id not in expanded_e and child_var_id not in frontier_f:
                        frontier_f[child_var_id] = child
            needs_final_solve = True

        if partial_tree is None or partial_result is None or needs_final_solve:
            partial_tree, partial_result = self._solve_partial_policy_ilp(
                runtime,
                interface,
                duration_evaluator,
                expanded_records=tuple(expanded_e.values()),
                frontier=tuple(frontier_f.values()),
                frontier_expansions=frontier_expansions,
                heuristic_cache=heuristic_cache,
                heuristic_depth=heuristic_depth,
                root_belief=root_belief,
            )
        self.last_partial_tree = partial_tree
        self.last_ilp_result = partial_result
        self.last_stats = HILPSearchStats(
            iterations=expansion_rounds,
            expanded_count=len(expanded_e),
            frontier_count=len(frontier_f),
            selected_frontier=tuple(selected_frontier_labels),
            partial_variable_count=len(partial_tree.spec.variables),
        )
        selected_root = _selected_root_variable(partial_result, partial_tree)
        if selected_root is None:
            raise RuntimeError(
                "Gurobi HILP partial-tree ILP did not select a root action. "
                f"status={partial_result.status}"
            )
        selected_item = partial_tree.variable_items[selected_root]
        elapsed_ms = (perf_counter() - started_at) * 1000.0
        return ActionDecision(
            action=dict(selected_item.node.metadata["assignment"]),
            label=selected_item.action_label,
            value=float(partial_result.objective_value or 0.0),
            action_values=_root_objective_values(partial_tree),
            remaining_depth=heuristic_depth,
            elapsed_ms=elapsed_ms,
            complete=partial_result.is_optimal,
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
        heuristic_cache: dict[tuple[object, ...], tuple[float, float]],
        heuristic_depth: int,
        root_belief: Mapping[StateKey, float] | None,
    ) -> tuple[PolicyTreeILP, ILPSolveResult]:
        r"""Solve Algorithm 3's current partial-tree p-ILP.

        For every frontier history $$q\in F$$, DARP first calls Algorithm 2
        only far enough to obtain the leaf constants $$u_q,r_q$$.  The partial
        ILP then decides both the current root action and which frontier leaves
        carry positive policy mass.

        / 求解当前 $$E\cup F$$ partial-tree p-ILP；frontier 只作为叶子，
        不会触发完整 horizon 枚举。
        """

        frontier_records = tuple(
            _frontier_leaf_record(
                item,
                interface,
                duration_evaluator,
                frontier_expansions,
                heuristic_cache,
                heuristic_depth=heuristic_depth,
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
        result = GurobiILPSolver().solve(partial_ilp.spec)
        return partial_ilp, result

    def _selected_expandable_frontier(
        self,
        result: ILPSolveResult,
        partial_tree: PolicyTreeILP,
        frontier: Mapping[str, FrontierItem],
        frontier_expansions: Mapping[str, ExpandedAction],
        *,
        heuristic_depth: int,
    ) -> tuple[tuple[str, FrontierItem], ...]:
        """Return selected frontier leaves that may be expanded. / 返回解中选中且可继续展开的 frontier leaf。"""
        selected: list[tuple[str, FrontierItem]] = []
        frontier_ids = set(partial_tree.frontier_variable_ids)
        for var_id, value in result.variable_values.items():
            if var_id not in frontier_ids or var_id not in frontier:
                continue
            if float(value) <= 0.5:
                continue
            item = frontier[var_id]
            expanded = frontier_expansions[var_id]
            if item.node.history.depth >= heuristic_depth:
                continue
            if not expanded.child_frontier:
                continue
            selected.append((var_id, item))
        # When the partial policy selects many frontier leaves, expand the
        # strongest ones first.  This is a practical width cap around Algorithm
        # 3, not a second ILP. / 当 p-ILP 同时选中多个 frontier 时，优先展开
        # 目标系数更高的节点；这是 Algorithm 3 外围的宽度上限，不是第二个 ILP。
        selected.sort(
            key=lambda pair: (
                -float(partial_tree.spec.objective.get(pair[0], 0.0)),
                pair[1].node.history.depth,
                pair[1].node.history.label(),
            )
        )
        return tuple(selected[: self.frontier_width])


def _frontier_leaf_record(
    item: FrontierItem,
    interface: ANDORSearchInterface,
    duration_evaluator: HistoryDurationEvaluator,
    frontier_expansions: dict[str, ExpandedAction],
    heuristic_cache: dict[tuple[object, ...], tuple[float, float]],
    *,
    heuristic_depth: int,
) -> Algorithm1ExpansionRecord:
    r"""Return a frontier leaf record with exact heuristic constants.

    The frontier record stays a leaf in the p-ILP, but its objective uses a
    bounded exact heuristic:

    $$\hat V(q)=u_q+\sum_o \max_a \hat V(qoa),\qquad
      \hat R(q)=r_q+\sum_o \hat R(qoa^*)$$

    where $$a^*$$ is the action selected by the same leaf heuristic.  This
    keeps the partial-tree objective and chance-constraint rows aligned.

    / frontier 在 p-ILP 中仍是叶子，但目标系数使用受深度限制的 exact
    启发式估值，而不是只看一步 reward；风险系数也使用同一启发式 policy
    的未来风险估计。
    """
    var_id = _action_var_id(item)
    expanded = _cached_expand(item, interface, duration_evaluator, frontier_expansions)
    heuristic_utility, heuristic_risk = _exact_frontier_heuristic_metrics(
        item,
        interface,
        duration_evaluator,
        frontier_expansions,
        heuristic_cache,
        heuristic_depth=heuristic_depth,
    )
    leaf_expanded = replace(
        expanded,
        metrics=replace(
            expanded.metrics,
            utility=heuristic_utility,
            risk=heuristic_risk,
        ),
    )
    return Algorithm1ExpansionRecord(
        var_id=var_id,
        item=item,
        expanded=leaf_expanded,
        continues=False,
    )


def _exact_frontier_heuristic_metrics(
    item: FrontierItem,
    interface: ANDORSearchInterface,
    duration_evaluator: HistoryDurationEvaluator,
    frontier_expansions: dict[str, ExpandedAction],
    heuristic_cache: dict[tuple[object, ...], tuple[float, float]],
    *,
    heuristic_depth: int,
) -> tuple[float, float]:
    r"""Return bounded exact utility/risk heuristics for HILP frontier nodes.

    The recursion is local to HILP's current lookahead cap and uses Algorithm 2
    exact branches:

    $$\hat V(q)=u_q+\sum_o \max_a \hat V(qoa),\qquad
      \hat R(q)=r_q+\sum_o \hat R(qoa^*),\qquad depth(q)<d.$$

    It is a heuristic for partial-tree leaves, not the full-ILP baseline.  It
    uses exact finite transition/observation branches, but only to the HILP
    heuristic depth.

    / 只在 HILP lookahead 上界内递归估计 frontier 叶子的 utility 和 risk；
    它帮助 partial ILP 看到未来 reward/risk，但不会枚举完整 horizon。
    """
    cache_key = _heuristic_cache_key(item, duration_evaluator, heuristic_depth)
    if cache_key in heuristic_cache:
        return heuristic_cache[cache_key]
    expanded = _cached_expand(item, interface, duration_evaluator, frontier_expansions)
    utility = expanded.metrics.utility
    risk = expanded.metrics.risk
    if item.node.history.depth < heuristic_depth:
        for branch in expanded.observation_frontiers:
            if not branch.should_expand or not branch.child_frontier:
                continue
            child_heuristics = tuple(
                _exact_frontier_heuristic_metrics(
                    child,
                    interface,
                    duration_evaluator,
                    frontier_expansions,
                    heuristic_cache,
                    heuristic_depth=heuristic_depth,
                )
                for child in branch.child_frontier
            )
            if child_heuristics:
                child_utility, child_risk = max(
                    child_heuristics,
                    key=lambda pair: pair[0],
                )
                utility += child_utility
                risk += child_risk
    heuristic_cache[cache_key] = (utility, risk)
    return utility, risk


def _heuristic_cache_key(
    item: FrontierItem,
    duration_evaluator: HistoryDurationEvaluator,
    heuristic_depth: int,
) -> tuple[object, ...]:
    """Return a semantic cache key for equivalent HILP leaf subproblems. / 为等价 HILP 叶子子问题返回语义缓存键。"""
    prefix_action_count = min(len(item.duration_beliefs), len(item.node.history.actions))
    prefix_progress = duration_evaluator.progress_for_actions(
        item.node.history.actions[:prefix_action_count],
        item.duration_beliefs,
    )
    return (
        heuristic_depth - item.node.history.depth,
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
