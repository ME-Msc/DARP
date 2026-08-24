"""HILP-style partial frontier search for the paper algorithm."""

# TODO: Reuse a persistent Gurobi model across refinements. Incumbent values
# are already transferred as MIP starts, but variables/rows are still rebuilt.

from __future__ import annotations

from dataclasses import dataclass, field, replace
from math import isfinite
from time import perf_counter
from typing import Literal, Mapping

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

HILPHeuristicMode = Literal["one-step-greedy", "reachable-bellman"]


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

    heuristic_lookahead_depth: int = 4
    expansion_rounds: int | None = None
    frontier_width: int = 1
    heuristic_mode: HILPHeuristicMode = "one-step-greedy"
    risk_budget: float | None = None
    solver_time_limit_ms: float | None = 60_000.0
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
        - Each iteration solves a partial ILP over $$E \cup F$$, then expands
          only incumbent frontier histories with $$x_q>0$$. The heuristic
          orders that selected batch; it never replaces the p-ILP decision.
        - DARP's partial ILP keeps the same Definition 3.1 root/flow rows as
          full-ILP for histories in $$E$$.  Histories in $$F$$ are frontier
          leaves: they have exact one-step $$u_q,r_q$$ constants, but no
          child-flow rows yet.
        - The CC-POMDP time budget is the domain horizon inside
          `duration_evaluator`; it is consumed by action durations through
          $$\tau(q)$$.  It is not Python wall-clock runtime.

        / 使用 HILP 的 $$E/F$$ frontier 更新框架；每轮只展开 p-ILP incumbent
        中 $$x_q>0$$ 的 frontier，heuristic 仅用于批内排序，最终 root action
        也必须来自 incumbent。规划问题
        中的时间预算由 `duration_evaluator` 的 action duration 和 $$\tau(q)$$
        表示，不使用 Python 运行时间。
        """

        started_at = perf_counter()
        if self.heuristic_lookahead_depth < 0:
            raise ValueError("heuristic_lookahead_depth must be at least 0.")
        if self.expansion_rounds is not None and self.expansion_rounds < 0:
            raise ValueError("expansion_rounds must be non-negative when provided.")
        if self.frontier_width < 1:
            raise ValueError("frontier_width must be at least 1.")
        if remaining_depth < 1:
            raise ValueError("remaining_depth must be at least 1.")
        if self.risk_budget is not None and (
            not isfinite(float(self.risk_budget))
            or not 0.0 <= float(self.risk_budget) <= 1.0
        ):
            raise ValueError("risk_budget must be a finite probability in [0, 1].")
        if self.solver_time_limit_ms is not None and (
            not isfinite(float(self.solver_time_limit_ms))
            or float(self.solver_time_limit_ms) <= 0.0
        ):
            raise ValueError("solver_time_limit_ms must be finite and positive when provided.")
        if self.heuristic_mode not in ("one-step-greedy", "reachable-bellman"):
            raise ValueError(f"Unsupported HILP heuristic mode: {self.heuristic_mode}")

        # HILP tree expansion is bounded by the RDDL remaining horizon and the
        # duration stopping condition.  `heuristic_lookahead_depth` is only for
        # frontier scoring, not for truncating the partial policy tree.
        # HILP 树展开由 RDDL 剩余 horizon 和 duration stopping condition 限制；
        # `heuristic_lookahead_depth` 只用于 frontier 估值，不截断 partial tree。
        expansion_depth_limit = max(1, remaining_depth)
        solver_deadline = (
            started_at + float(self.solver_time_limit_ms) / 1000.0
            if self.solver_time_limit_ms is not None
            else None
        )
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
        heuristic_cache: dict[tuple[object, ...], object] = {}
        selected_frontier_labels: list[str] = []
        partial_tree: PolicyTreeILP | None = None
        partial_result: ILPSolveResult | None = None
        expansion_rounds = 0
        needs_final_solve = True
        solver_limit_hit = False
        timing_totals = {
            "tree_ilp_build_ms": 0.0,
            "frontier_expand_ms": 0.0,
            "heuristic_eval_ms": 0.0,
            "ilp_encode_ms": 0.0,
            "gurobi_solve_ms": 0.0,
            "gurobi_call_ms": 0.0,
            "mip_start_variables": 0.0,
        }

        while frontier_f and (
            self.expansion_rounds is None
            or expansion_rounds < self.expansion_rounds
        ):
            partial_tree, partial_result = self._solve_partial_policy_ilp(
                runtime,
                interface,
                duration_evaluator,
                expanded_records=tuple(expanded_e.values()),
                frontier=tuple(frontier_f.values()),
                frontier_expansions=frontier_expansions,
                heuristic_cache=heuristic_cache,
                expansion_depth_limit=expansion_depth_limit,
                root_belief=root_belief,
                warm_start=(partial_result.variable_values if partial_result is not None else None),
                solver_deadline=solver_deadline,
                timing_totals=timing_totals,
            )
            needs_final_solve = False
            if partial_result.status == "time_limit":
                solver_limit_hit = True
                break
            # Algorithm 3: solve over E U F, then refine only incumbent leaves.
            # Descendants behind F are the implicit N set until materialized.
            selected = self._selected_expandable_frontier(
                partial_tree,
                partial_result,
                frontier_f,
                frontier_expansions,
                expansion_depth_limit=expansion_depth_limit,
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
                expansion_depth_limit=expansion_depth_limit,
                root_belief=root_belief,
                warm_start=(partial_result.variable_values if partial_result is not None else None),
                solver_deadline=solver_deadline,
                timing_totals=timing_totals,
            )
            solver_limit_hit = solver_limit_hit or partial_result.status == "time_limit"
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
            if partial_result.status == "time_limit":
                raise TimeoutError("Gurobi reached the HILP wall budget before finding a root incumbent.")
            raise RuntimeError(
                "Gurobi HILP partial-tree ILP did not select a root action. "
                f"status={partial_result.status}"
            )
        postprocess_started_at = perf_counter()
        selected_item = partial_tree.variable_items[selected_root]
        postprocess_ms = (perf_counter() - postprocess_started_at) * 1000.0
        elapsed_ms = (perf_counter() - started_at) * 1000.0
        decision_ms = (
            timing_totals["tree_ilp_build_ms"]
            + timing_totals["gurobi_solve_ms"]
            + postprocess_ms
        )
        cache_info = (
            interface.exact_kernel.cache_info()
            if interface.exact_kernel is not None and hasattr(interface.exact_kernel, "cache_info")
            else {}
        )
        uses_terminal_heuristic = (
            self.heuristic_mode == "reachable-bellman"
            and interface.exact_kernel is not None
            and bool(getattr(interface.exact_kernel, "terminal_utility_enabled", True))
            and callable(getattr(interface.exact_kernel, "terminal_utility", None))
        )
        selected_expandable = self._selected_expandable_frontier(
            partial_tree,
            partial_result,
            frontier_f,
            frontier_expansions,
            expansion_depth_limit=expansion_depth_limit,
        )
        refinement_exhausted = not selected_expandable
        globally_expandable = self._globally_expandable_frontier(
            partial_tree,
            frontier_f,
            frontier_expansions,
            expansion_depth_limit=expansion_depth_limit,
        )
        required_bound_depth = max(
            (
                expansion_depth_limit - item.node.history.depth
                for _, item in globally_expandable
            ),
            default=0,
        )
        certifying_utility_bound = (
            not globally_expandable
            or (
                self.heuristic_mode == "reachable-bellman"
                and self.heuristic_lookahead_depth >= required_bound_depth
            )
        )
        search_complete = (
            partial_result.is_optimal
            and not solver_limit_hit
            and refinement_exhausted
            and certifying_utility_bound
            and not uses_terminal_heuristic
        )
        return ActionDecision(
            action=dict(selected_item.node.metadata["assignment"]),
            label=selected_item.action_label,
            value=float(partial_result.objective_value or 0.0),
            action_values=_root_objective_values(partial_tree),
            remaining_depth=expansion_depth_limit,
            elapsed_ms=elapsed_ms,
            complete=search_complete,
            timing={
                "planner_elapsed_ms": elapsed_ms,
                "decision_ms": decision_ms,
                "tree_ilp_build_ms": timing_totals["tree_ilp_build_ms"],
                "frontier_expand_ms": timing_totals["frontier_expand_ms"],
                "heuristic_eval_ms": timing_totals["heuristic_eval_ms"],
                "ilp_encode_ms": timing_totals["ilp_encode_ms"],
                "gurobi_solve_ms": timing_totals["gurobi_solve_ms"],
                "gurobi_call_ms": timing_totals["gurobi_call_ms"],
                "mip_start_variables": timing_totals["mip_start_variables"],
                "postprocess_ms": postprocess_ms,
                "ilp_variables": float(len(partial_tree.spec.variables)),
                "ilp_constraints": float(len(partial_tree.spec.constraints)),
                "expanded_nodes": float(len(expanded_e)),
                "frontier_nodes": float(len(frontier_f)),
                "expansion_rounds": float(expansion_rounds),
                "search_complete": 1.0 if search_complete else 0.0,
                "frontier_refinement_exhausted": 1.0 if refinement_exhausted else 0.0,
                "global_expandable_frontier": float(len(globally_expandable)),
                "certifying_utility_bound": 1.0 if certifying_utility_bound else 0.0,
                "solver_time_limit_hit": 1.0 if solver_limit_hit else 0.0,
                "used_terminal_heuristic": 1.0 if uses_terminal_heuristic else 0.0,
                "heuristic_lookahead_depth": float(self.heuristic_lookahead_depth),
                "hilp_heuristic_one_step_greedy": 1.0 if self.heuristic_mode == "one-step-greedy" else 0.0,
                "hilp_heuristic_reachable_bellman": 1.0 if self.heuristic_mode == "reachable-bellman" else 0.0,
                **{f"exact_{name}": float(value) for name, value in cache_info.items()},
            },
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
        heuristic_cache: dict[tuple[object, ...], object],
        expansion_depth_limit: int,
        root_belief: Mapping[StateKey, float] | None,
        warm_start: Mapping[str, float] | None,
        solver_deadline: float | None,
        timing_totals: dict[str, float] | None = None,
    ) -> tuple[PolicyTreeILP, ILPSolveResult]:
        r"""Solve Algorithm 3's current partial-tree p-ILP.

        For every frontier history $$q\in F$$, DARP first calls Algorithm 2
        only far enough to obtain the leaf constants $$u_q,r_q$$.  The partial
        ILP then decides both the current root action and which frontier leaves
        carry positive policy mass.

        / 求解当前 $$E\cup F$$ partial-tree p-ILP；frontier 只作为叶子，
        不会触发完整 horizon 枚举。
        """

        build_started_at = perf_counter()
        frontier_records_list: list[Algorithm1ExpansionRecord] = []
        frontier_expand_ms = 0.0
        heuristic_eval_ms = 0.0
        for item in frontier:
            record, record_timing = _frontier_leaf_record(
                item,
                interface,
                duration_evaluator,
                frontier_expansions,
                heuristic_cache,
                heuristic_mode=self.heuristic_mode,
                heuristic_lookahead_depth=min(
                    self.heuristic_lookahead_depth,
                    max(0, expansion_depth_limit - item.node.history.depth),
                ),
                risk_budget=self.risk_budget,
            )
            frontier_records_list.append(record)
            frontier_expand_ms += record_timing["frontier_expand_ms"]
            heuristic_eval_ms += record_timing["heuristic_eval_ms"]
        frontier_records = tuple(frontier_records_list)
        encode_started_at = perf_counter()
        partial_ilp = build_partial_tree_ilp(
            runtime=runtime,
            interface=interface,
            expanded_records=expanded_records,
            frontier_records=frontier_records,
            risk_budget=self.risk_budget,
            root_belief=root_belief,
        )
        ilp_encode_ms = (perf_counter() - encode_started_at) * 1000.0
        build_ms = (perf_counter() - build_started_at) * 1000.0
        solve_started_at = perf_counter()
        remaining_solver_ms: float | None = None
        if solver_deadline is not None:
            remaining_solver_ms = (solver_deadline - solve_started_at) * 1000.0
            if remaining_solver_ms <= 0.0:
                raise TimeoutError("HILP wall budget expired before the next Gurobi refinement.")
        shared_start = (
            {var_id: value for var_id, value in warm_start.items() if var_id in partial_ilp.spec.variable_ids()}
            if warm_start
            else None
        )
        result = GurobiILPSolver().solve(
            partial_ilp.spec,
            time_limit_ms=remaining_solver_ms,
            warm_start=shared_start,
        )
        solver_call_ms = (perf_counter() - solve_started_at) * 1000.0
        if timing_totals is not None:
            timing_totals["tree_ilp_build_ms"] = timing_totals.get("tree_ilp_build_ms", 0.0) + build_ms
            timing_totals["frontier_expand_ms"] = timing_totals.get("frontier_expand_ms", 0.0) + frontier_expand_ms
            timing_totals["heuristic_eval_ms"] = timing_totals.get("heuristic_eval_ms", 0.0) + heuristic_eval_ms
            timing_totals["ilp_encode_ms"] = timing_totals.get("ilp_encode_ms", 0.0) + ilp_encode_ms
            timing_totals["gurobi_solve_ms"] = timing_totals.get("gurobi_solve_ms", 0.0) + float(result.runtime_ms)
            timing_totals["gurobi_call_ms"] = timing_totals.get("gurobi_call_ms", 0.0) + solver_call_ms
            timing_totals["mip_start_variables"] = timing_totals.get("mip_start_variables", 0.0) + float(
                len(shared_start or {})
            )
        return partial_ilp, result

    def _selected_expandable_frontier(
        self,
        partial_tree: PolicyTreeILP,
        partial_result: ILPSolveResult,
        frontier: Mapping[str, FrontierItem],
        frontier_expansions: Mapping[str, ExpandedAction],
        *,
        expansion_depth_limit: int,
    ) -> tuple[tuple[str, FrontierItem], ...]:
        r"""Return only frontier leaves selected by the current p-ILP.

        This is Algorithm 3 lines 12-17: a frontier history enters ``E`` only
        when its incumbent value satisfies :math:`x_q>0`. ``frontier_width``
        is merely a batching limit within that selected set.

        / 严格对应 Algorithm 3：先筛选 p-ILP 中 ``x_q>0`` 的 frontier，再在
        该集合内按 heuristic 排序并应用批量宽度。
        """
        selected: list[tuple[float, bool, str, FrontierItem]] = []
        frontier_ids = set(partial_tree.frontier_variable_ids)
        incumbent_ids = {
            var_id
            for var_id, value in partial_result.variable_values.items()
            if float(value) > 0.5
        }
        for var_id, item in frontier.items():
            if var_id not in frontier_ids or var_id not in incumbent_ids:
                continue
            expanded = frontier_expansions[var_id]
            if item.node.history.depth >= expansion_depth_limit:
                continue
            if not expanded.child_frontier:
                continue
            # The p-ILP objective coefficient is $$h_q^u$$ for frontier leaves,
            # so it is also the greedy expansion score. / frontier 的目标系数就是
            # $$h_q^u$$，也是贪心展开分数。
            score = float(partial_tree.spec.objective.get(var_id, 0.0))
            selected.append((score, _is_noop_item(item), var_id, item))
        # Expand the frontier with the largest heuristic utility.  A
        # deterministic tie-break keeps no-op after real actions when scores are
        # equal, which avoids arbitrary solver ordering on flat rewards.
        # 展开 heuristic 最大的 frontier；若分数相同，真实动作优先于 noop。
        selected.sort(
            key=lambda pair: (
                -pair[0],
                pair[1],
                pair[3].node.history.depth,
                pair[3].node.history.label(),
            )
        )
        return tuple((var_id, item) for _, _, var_id, item in selected[: self.frontier_width])

    @staticmethod
    def _globally_expandable_frontier(
        partial_tree: PolicyTreeILP,
        frontier: Mapping[str, FrontierItem],
        frontier_expansions: Mapping[str, ExpandedAction],
        *,
        expansion_depth_limit: int,
    ) -> tuple[tuple[str, FrontierItem], ...]:
        """Return every frontier with materializable descendants.

        This set is deliberately independent of the incumbent. It is used for
        completeness certification: a non-admissible one-step score cannot
        prove an unselected alternative subtree is inferior.
        """

        frontier_ids = set(partial_tree.frontier_variable_ids)
        return tuple(
            (var_id, item)
            for var_id, item in frontier.items()
            if var_id in frontier_ids
            and item.node.history.depth < expansion_depth_limit
            and bool(frontier_expansions[var_id].child_frontier)
        )


def _frontier_leaf_record(
    item: FrontierItem,
    interface: ANDORSearchInterface,
    duration_evaluator: HistoryDurationEvaluator,
    frontier_expansions: dict[str, ExpandedAction],
    heuristic_cache: dict[tuple[object, ...], object],
    *,
    heuristic_mode: HILPHeuristicMode,
    heuristic_lookahead_depth: int,
    risk_budget: float | None,
) -> tuple[Algorithm1ExpansionRecord, dict[str, float]]:
    r"""Return a frontier leaf record with the selected utility heuristic.

    Reference-code correspondence:

    - The author's ``heuristic_search`` stores one coefficient ``h`` on each
      frontier history and expands only frontier histories selected by p-ILP.
    - ``one-step-greedy`` sets the frontier coefficient to the exact constant already
      computed by Algorithm 2:

      $$h_q^u := u_q = \rho(q)\sum_s b_q(s)U(s,a_q).$$

    - ``reachable-bellman`` computes a frontier-local fully observable MDP
      relaxation only over states reachable from the current action's successor
      support:

      $$V_t(s)=\max_a\left[U(s,a)+\sum_{s'}T(s,a,s')V_{t-1}(s')\right].$$

      It does not sample and does not expand observation branches.
      If a kernel exposes ``terminal_utility(belief, risk_budget)``, that
      admissible domain tail bound initializes $$V_0$$ (the paper grid uses
      negative Manhattan distance in reward-maximization form).
    - The risk coefficient remains the one-step safe-belief $$r_q$$, matching
      the reference code's unfinished risk-heuristic path.

    / 这里根据模式选择 frontier 的 utility heuristic：``one-step-greedy`` 直接
    使用一步 $$u_q$$；``reachable-bellman`` 只在当前 action 后继可达状态上做
    全可观测 Bellman 表。不采样，也不展开 observation 分支。风险项仍使用一步
    safe-belief $$r_q$$。
    """
    var_id = _action_var_id(item)
    expand_started_at = perf_counter()
    expanded = _cached_expand(item, interface, duration_evaluator, frontier_expansions)
    frontier_expand_ms = (perf_counter() - expand_started_at) * 1000.0
    heuristic_eval_ms = 0.0
    if heuristic_mode == "reachable-bellman":
        heuristic_started_at = perf_counter()
        heuristic_utility = _reachable_bellman_frontier_utility(
            item,
            interface,
            heuristic_cache,
            heuristic_lookahead_depth=heuristic_lookahead_depth,
            risk_budget=risk_budget,
        )
        expanded = replace(
            expanded,
            metrics=replace(
                expanded.metrics,
                utility=heuristic_utility,
            ),
        )
        heuristic_eval_ms = (perf_counter() - heuristic_started_at) * 1000.0
    elif heuristic_mode != "one-step-greedy":
        raise ValueError(f"Unsupported HILP heuristic mode: {heuristic_mode}")
    return (
        Algorithm1ExpansionRecord(
            var_id=var_id,
            item=item,
            expanded=expanded,
            continues=False,
        ),
        {
            "frontier_expand_ms": frontier_expand_ms,
            "heuristic_eval_ms": heuristic_eval_ms,
        },
    )


def _reachable_bellman_frontier_utility(
    item: FrontierItem,
    interface: ANDORSearchInterface,
    heuristic_cache: dict[tuple[object, ...], object],
    *,
    heuristic_lookahead_depth: int,
    risk_budget: float | None,
) -> float:
    r"""Return $$h_q^u$$ from a reachable-state fully observable Bellman table.

    The frontier action $$a_q$$ is already fixed, so only its successor support
    seeds the future value table:

    $$h_q^u=\rho(q)\sum_s b_q(s)
      \left[U(s,a_q)+\sum_{s'}T(s,a_q,s')V_{d-1}(s')\right].$$

    / 用当前 action 的后继状态作为 Bellman 表种子；不枚举全局状态空间。
    """
    exact_kernel = interface.exact_kernel
    if exact_kernel is None:
        raise ValueError("Reachable Bellman HILP heuristic requires interface.exact_kernel.")
    belief = item.belief
    if not belief:
        return 0.0
    action = _action_assignment(item)
    future_horizon = max(0, heuristic_lookahead_depth)
    successor_states: set[StateKey] = set()
    for state, probability in belief.items():
        if float(probability) <= 0.0:
            continue
        successor_states.update(
            _transition_distribution(
                exact_kernel,
                state,
                item.action_label,
                action,
                heuristic_cache,
            )
        )
    future_values = _reachable_bellman_value_table(
        exact_kernel,
        interface,
        seed_states=tuple(successor_states),
        horizon=future_horizon,
        heuristic_cache=heuristic_cache,
        risk_budget=risk_budget,
    )
    expected_q = sum(
        float(probability)
        * _fully_observable_action_value(
            exact_kernel,
            state,
            item.action_label,
            action,
            future_values,
            heuristic_cache,
        )
        for state, probability in belief.items()
        if float(probability) > 0.0
    )
    return float(item.rho) * expected_q


def _reachable_bellman_value_table(
    exact_kernel: object,
    interface: ANDORSearchInterface,
    *,
    seed_states: tuple[StateKey, ...],
    horizon: int,
    heuristic_cache: dict[tuple[object, ...], object],
    risk_budget: float | None,
) -> Mapping[StateKey, float]:
    r"""Return $$V_H$$ over states reachable from ``seed_states``.

    Complexity is $$O(H|S_{reach}||A|d_T)$$ for sparse transitions, not full
    history-tree expansion. / 复杂度按可达状态集合计算，而不是按完整历史树计算。
    """
    if not seed_states:
        return {}
    cache_key = ("bellman_values", _state_set_key(seed_states), horizon, risk_budget)
    if cache_key in heuristic_cache:
        return heuristic_cache[cache_key]  # type: ignore[return-value]
    state_space = _reachable_state_space(
        exact_kernel,
        interface,
        seed_states=seed_states,
        horizon=horizon,
        heuristic_cache=heuristic_cache,
    )
    terminal_utility = (
        getattr(exact_kernel, "terminal_utility", None)
        if bool(getattr(exact_kernel, "terminal_utility_enabled", True))
        else None
    )
    has_terminal_heuristic = callable(terminal_utility)
    previous = {
        state: (
            float(terminal_utility({state: 1.0}, risk_budget))
            if has_terminal_heuristic
            else 0.0
        )
        for state in state_space
    }
    for depth in range(1, horizon + 1):
        current: dict[StateKey, float] = {}
        for state in state_space:
            best_continuation = max(
                _fully_observable_action_value(
                    exact_kernel,
                    state,
                    choice.label,
                    choice.assignment,
                    previous,
                    heuristic_cache,
                )
                for choice in interface.actions
            )
            # Without a domain tail heuristic, the relaxed MDP may stop with
            # zero value to preserve an optimistic upper bound for negative
            # rewards. A supplied terminal heuristic already defines the
            # continuation obligation and must not be bypassed by stopping.
            current[state] = (
                best_continuation
                if has_terminal_heuristic
                else max(0.0, best_continuation)
            )
        previous = current
    heuristic_cache[cache_key] = previous
    return previous


def _reachable_state_space(
    exact_kernel: object,
    interface: ANDORSearchInterface,
    *,
    seed_states: tuple[StateKey, ...],
    horizon: int,
    heuristic_cache: dict[tuple[object, ...], object],
) -> tuple[StateKey, ...]:
    """Return the finite reachable state closure. / 返回有限可达状态闭包。"""
    cache_key = ("reachable_states", _state_set_key(seed_states), horizon)
    if cache_key in heuristic_cache:
        return heuristic_cache[cache_key]  # type: ignore[return-value]
    states = set(seed_states)
    frontier = set(seed_states)
    for _ in range(max(0, horizon)):
        next_frontier: set[StateKey] = set()
        for state in frontier:
            for choice in interface.actions:
                next_frontier.update(
                    _transition_distribution(
                        exact_kernel,
                        state,
                        choice.label,
                        choice.assignment,
                        heuristic_cache,
                    )
                )
        next_frontier -= states
        if not next_frontier:
            break
        states.update(next_frontier)
        frontier = next_frontier
    state_space = tuple(sorted(states, key=repr))
    heuristic_cache[cache_key] = state_space
    return state_space


def _fully_observable_action_value(
    exact_kernel: object,
    state: StateKey,
    action_label: str,
    action: Mapping[str, object],
    future_values: Mapping[StateKey, float],
    heuristic_cache: dict[tuple[object, ...], object],
) -> float:
    r"""Return $$U(s,a)+\sum_{s'}T(s,a,s')V(s')$$. / 返回全可观测 action value。"""
    reward = _state_action_reward(exact_kernel, state, action_label, action, heuristic_cache)
    transition = _transition_distribution(
        exact_kernel,
        state,
        action_label,
        action,
        heuristic_cache,
    )
    future = sum(
        float(probability) * float(future_values.get(next_state, 0.0))
        for next_state, probability in transition.items()
    )
    return reward + future


def _state_action_reward(
    exact_kernel: object,
    state: StateKey,
    action_label: str,
    action: Mapping[str, object],
    heuristic_cache: dict[tuple[object, ...], object],
) -> float:
    """Return $$U(s,a)$$ through the exact kernel. / 通过 exact kernel 返回 $$U(s,a)$$。"""
    cache_key = ("reward", state, action_label)
    if cache_key in heuristic_cache:
        return float(heuristic_cache[cache_key])
    if hasattr(exact_kernel, "expected_state_action_reward"):
        reward = float(exact_kernel.expected_state_action_reward(state, action))  # type: ignore[attr-defined]
    elif hasattr(exact_kernel, "expected_reward") and hasattr(exact_kernel, "_context"):
        context = exact_kernel._context(exact_kernel.state_from_key(state), action)  # type: ignore[attr-defined]
        reward = float(exact_kernel.expected_reward(context))  # type: ignore[attr-defined]
    else:
        expansion = exact_kernel.expand_action({state: 1.0}, action)  # type: ignore[attr-defined]
        reward = float(expansion.utility)
    heuristic_cache[cache_key] = reward
    return reward


def _transition_distribution(
    exact_kernel: object,
    state: StateKey,
    action_label: str,
    action: Mapping[str, object],
    heuristic_cache: dict[tuple[object, ...], object],
) -> Mapping[StateKey, float]:
    r"""Return cached $$T(s,a,\cdot)$$. / 返回缓存的转移分布。"""
    cache_key = ("transition", state, action_label)
    if cache_key not in heuristic_cache:
        heuristic_cache[cache_key] = dict(
            exact_kernel.transition_distribution(  # type: ignore[attr-defined]
                exact_kernel.state_from_key(state),  # type: ignore[attr-defined]
                action,
            )
        )
    return heuristic_cache[cache_key]  # type: ignore[return-value]


def _action_assignment(item: FrontierItem) -> Mapping[str, object]:
    """Return the concrete action assignment for a frontier item. / 返回 frontier item 的具体动作赋值。"""
    assignment = item.node.metadata.get("assignment")
    if not isinstance(assignment, Mapping):
        raise ValueError(f"Frontier item has no action assignment: {item.node.history.label()}")
    return assignment


def _state_set_key(states: tuple[StateKey, ...]) -> tuple[str, ...]:
    """Return a stable cache key for a state set. / 返回状态集合的稳定缓存键。"""
    return tuple(sorted((repr(state) for state in states)))


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


def _is_noop_item(item: FrontierItem) -> bool:
    """Return whether a frontier item represents no enabled action. / 判断 frontier 是否为 noop。"""
    assignment = item.node.metadata.get("assignment")
    if isinstance(assignment, Mapping):
        return not any(bool(value) for value in assignment.values())
    return item.action_label == "noop"


def _selected_root_variable(result: ILPSolveResult, tree: PolicyTreeILP) -> str | None:
    """Return the root action selected by the p-ILP incumbent. / 返回 p-ILP incumbent 选中的根动作。"""
    selected_ids = set(result.selected_variables)
    candidates: list[tuple[float, bool, str, str]] = []
    for var_id in tree.root_variable_ids:
        if var_id not in selected_ids:
            continue
        item = tree.variable_items[var_id]
        candidates.append(
            (
                float(tree.spec.objective.get(var_id, 0.0)),
                _is_noop_item(item),
                item.node.history.label(),
                var_id,
            )
        )
    if not candidates:
        return None
    candidates.sort(key=lambda candidate: (-candidate[0], candidate[1], candidate[2]))
    return candidates[0][3]


def _root_objective_values(tree: PolicyTreeILP) -> dict[str, float]:
    """Return root-action objective coefficients for diagnostics. / 返回根 action 目标系数用于诊断。"""
    values: dict[str, float] = {}
    for var_id in tree.root_variable_ids:
        item = tree.variable_items[var_id]
        values[item.action_label] = float(tree.spec.objective.get(var_id, 0.0))
    return values
