"""HILP-style partial frontier search for the paper algorithm."""

from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction
from math import isfinite
from time import perf_counter
from typing import Literal, Mapping

from darp.adapter.exact import StateKey
from darp.adapter.runtime import PyRDDLGymRuntime
from darp.ilp.gurobi import GurobiILPSolver
from darp.ilp.model import ILPSolveResult
from darp.model.and_or_tree import ANDORSearchInterface
from darp.model.duration import HistoryDurationEvaluator
from darp.planning.decision import ActionDecision
from darp.planning.expand import ExpandedAction, expand_frontier_item
from darp.planning.ilp_tree import (
    Algorithm1ExpansionRecord,
    PolicyTreeILP,
    _action_var_id,
    build_partial_tree_ilp,
    validate_constraint_budget,
)
from darp.planning.preprocess import FrontierItem, initialize_root_frontier
from darp.planning.policy import extract_conditional_policy

HILPHeuristicMode = Literal["one-step-greedy", "reachable-bellman"]

@dataclass
class HILPPlanner:
    """Run paper Algorithm 3 style frontier expansion. / 运行论文 Algorithm 3 风格的 frontier expansion。"""

    heuristic_lookahead_depth: int = 4
    expansion_rounds: int | None = None
    frontier_width: int = 1
    heuristic_mode: HILPHeuristicMode = "reachable-bellman"
    risk_budget: float | None = None
    solver_time_limit_ms: float | None = 60_000.0

    def choose_action(
        self,
        runtime: PyRDDLGymRuntime,
        interface: ANDORSearchInterface,
        duration_evaluator: HistoryDurationEvaluator,
        *,
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
        validate_constraint_budget(interface, self.risk_budget)
        if self.solver_time_limit_ms is not None and (
            not isfinite(float(self.solver_time_limit_ms))
            or float(self.solver_time_limit_ms) <= 0.0
        ):
            raise ValueError("solver_time_limit_ms must be finite and positive when provided.")
        if self.heuristic_mode not in ("one-step-greedy", "reachable-bellman"):
            raise ValueError(f"Unsupported HILP heuristic mode: {self.heuristic_mode}")

        # This optional limit is proved from the duration model itself; it is
        # an optimization of the paper's tau test, not an integer horizon cap.
        duration_depth_bound = duration_evaluator.action_depth_upper_bound()
        expansion_depth_limit = duration_depth_bound
        solver_deadline = (
            started_at + float(self.solver_time_limit_ms) / 1000.0
            if self.solver_time_limit_ms is not None
            else None
        )
        root_frontier = initialize_root_frontier(runtime, interface, root_belief=root_belief)
        # Algorithm 3: $$F$$ starts from all root action histories. / 初始 frontier。
        frontier_f: dict[str, FrontierItem] = {
            _action_var_id(item): item
            for item in root_frontier
        }
        # Algorithm 3: $$E$$ stores histories that have already been expanded.
        # Algorithm 3：$$E$$ 保存已经调用过 Expand 的 action histories。
        expanded_e: dict[str, Algorithm1ExpansionRecord] = {}
        frontier_expansions: dict[str, ExpandedAction] = {}
        heuristic_cache: dict[tuple[object, ...], object] = {}
        partial_tree: PolicyTreeILP | None = None
        partial_result: ILPSolveResult | None = None
        expansion_rounds = 0
        needs_final_solve = True
        solver_limit_hit = False
        timing_totals = {
            "tree_ilp_build_ms": 0.0,
            "gurobi_solve_ms": 0.0,
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
        selected_root = _selected_root_variable(partial_result, partial_tree)
        if selected_root is None:
            if partial_result.status == "time_limit":
                raise TimeoutError("Gurobi reached the HILP wall budget before finding a root incumbent.")
            raise RuntimeError(
                "Gurobi HILP partial-tree ILP did not select a root action. "
                f"status={partial_result.status}"
            )
        selected_item = partial_tree.variable_items[selected_root]
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
        )
        required_bound_depth = (
            None
            if duration_depth_bound is None and globally_expandable
            else max(
                (
                    int(duration_depth_bound) - item.node.history.depth
                    for _, item in globally_expandable
                ),
                default=0,
            )
        )
        certifying_utility_bound = (
            not globally_expandable
            or (
                required_bound_depth is not None
                and self.heuristic_mode == "reachable-bellman"
                and self.heuristic_lookahead_depth >= required_bound_depth
            )
        )
        policy = extract_conditional_policy(partial_tree, partial_result)
        search_complete = (
            partial_result.numerically_optimal
            and partial_tree.objective_coefficients_exact
            and partial_tree.constraint_coefficients_exact
            and not solver_limit_hit
            and refinement_exhausted
            and certifying_utility_bound
            and policy.duration_complete
            and policy.feasible is not False
        )
        # Executability and global search optimality are separate certificates.
        # A duration-complete incumbent has an exact achieved utility even while
        # unselected alternatives remain unrefined; ``decision.complete`` stays
        # false until the HILP search certificate also closes.
        achieved_utility = policy.achieved_utility
        return ActionDecision(
            action=dict(selected_item.node.assignment or {}),
            label=selected_item.action_label,
            value=float(
                achieved_utility
                if achieved_utility is not None
                else (partial_result.objective_value or 0.0)
            ),
            complete=search_complete,
            value_kind=(
                "achieved_utility"
                if achieved_utility is not None
                else "heuristic_objective"
            ),
            policy=policy,
            timing={
                "tree_ilp_build_ms": timing_totals["tree_ilp_build_ms"],
                "gurobi_solve_ms": timing_totals["gurobi_solve_ms"],
                "ilp_variables": float(len(partial_tree.spec.variables)),
                "ilp_constraints": float(len(partial_tree.spec.constraints)),
                "expanded_nodes": float(len(expanded_e)),
                "frontier_nodes": float(len(frontier_f)),
                "expansion_rounds": float(expansion_rounds),
                "frontier_refinement_exhausted": 1.0 if refinement_exhausted else 0.0,
                "global_expandable_frontier": float(len(globally_expandable)),
                "certifying_utility_bound": 1.0 if certifying_utility_bound else 0.0,
                "solver_numerically_optimal": (
                    1.0 if partial_result.numerically_optimal else 0.0
                ),
                "numerical_zero_gap": (
                    1.0 if partial_result.has_numerical_zero_gap else 0.0
                ),
                "objective_coefficients_exact": (
                    1.0 if partial_tree.objective_coefficients_exact else 0.0
                ),
                "constraint_coefficients_exact": (
                    1.0 if partial_tree.constraint_coefficients_exact else 0.0
                ),
                "solver_time_limit_hit": 1.0 if solver_limit_hit else 0.0,
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
        expansion_depth_limit: int | None,
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
        for item in frontier:
            remaining_action_depth = (
                None
                if expansion_depth_limit is None
                else max(0, expansion_depth_limit - item.node.history.depth)
            )
            record = _frontier_leaf_record(
                item,
                interface,
                duration_evaluator,
                frontier_expansions,
                heuristic_cache,
                heuristic_mode=self.heuristic_mode,
                heuristic_lookahead_depth=min(
                    self.heuristic_lookahead_depth,
                    (
                        self.heuristic_lookahead_depth
                        if remaining_action_depth is None
                        else remaining_action_depth
                    ),
                ),
                remaining_action_depth=remaining_action_depth,
            )
            frontier_records_list.append(record)
        frontier_records = tuple(frontier_records_list)
        partial_ilp = build_partial_tree_ilp(
            runtime=runtime,
            interface=interface,
            expanded_records=expanded_records,
            frontier_records=frontier_records,
            risk_budget=self.risk_budget,
            root_belief=root_belief,
        )
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
        if timing_totals is not None:
            timing_totals["tree_ilp_build_ms"] = timing_totals.get("tree_ilp_build_ms", 0.0) + build_ms
            timing_totals["gurobi_solve_ms"] = timing_totals.get("gurobi_solve_ms", 0.0) + float(result.runtime_ms)
        return partial_ilp, result

    def _selected_expandable_frontier(
        self,
        partial_tree: PolicyTreeILP,
        partial_result: ILPSolveResult,
        frontier: Mapping[str, FrontierItem],
        frontier_expansions: Mapping[str, ExpandedAction],
        *,
        expansion_depth_limit: int | None,
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
            if (
                expansion_depth_limit is not None
                and item.node.history.depth >= expansion_depth_limit
            ):
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
    ) -> tuple[tuple[str, FrontierItem], ...]:
        """Return every frontier with materializable descendants.

        This set is deliberately independent of the incumbent. It is used for
        completeness certification and therefore also ignores an explicit
        decision-step cap: a duration-feasible child beyond that cap remains
        part of the paper's problem.
        """
        frontier_ids = set(partial_tree.frontier_variable_ids)
        return tuple(
            (var_id, item)
            for var_id, item in frontier.items()
            if var_id in frontier_ids
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
    remaining_action_depth: int | None,
) -> Algorithm1ExpansionRecord:
    r"""Return a frontier leaf record with the selected utility heuristic.

    Reference-code correspondence:

    - The author's ``heuristic_search`` stores one coefficient ``h`` on each
      frontier history and expands only frontier histories selected by p-ILP.
    - ``one-step-greedy`` sets the frontier coefficient to the exact constant
      already computed by Algorithm 2.  It is an approximate ordering score,
      not in general an admissible bound on unfinished descendants:

      $$h_q^u := u_q = \rho(q)\sum_s b_q(s)U(s,a_q).$$

    - ``reachable-bellman`` computes a frontier-local fully observable MDP
      relaxation only over states reachable from the current action's successor
      support:

      $$V_t(s)=\max_a\left[U(s,a)+\sum_{s'}T(s,a,s')V_{t-1}(s')\right].$$

      It does not sample or expand the observation tree.  Exact one-step utility
      is retained on every duration/action-depth terminal branch; Bellman future
      value is added only to observation branches Algorithm 2 says may continue.
    - The risk coefficient remains the one-step safe-belief $$r_q$$, matching
      the reference code's unfinished risk-heuristic path.

    / 这里根据模式选择 frontier 的 utility heuristic：``one-step-greedy`` 直接
    使用一步 $$u_q$$；``reachable-bellman`` 只在当前 action 后继可达状态上做
    全可观测 Bellman 表。不采样，也不展开 observation 分支。风险项仍使用一步
    safe-belief $$r_q$$。
    """
    var_id = _action_var_id(item)
    exact_expanded = _cached_expand(item, interface, duration_evaluator, frontier_expansions)
    expanded = exact_expanded
    can_continue = (
        remaining_action_depth is None or remaining_action_depth > 0
    ) and bool(expanded.child_frontier)
    if heuristic_mode == "reachable-bellman":
        if can_continue:
            (
                heuristic_utility,
                heuristic_utility_exact,
                heuristic_objective_exact,
            ) = _reachable_bellman_frontier_utility(
                expanded,
                interface,
                heuristic_cache,
                heuristic_lookahead_depth=heuristic_lookahead_depth,
            )
            expanded = replace(
                expanded,
                metrics=replace(
                    expanded.metrics,
                    utility=heuristic_utility,
                    utility_exact=heuristic_utility_exact,
                    objective_support_preserved=(
                        expanded.metrics.objective_support_preserved
                        and heuristic_objective_exact
                    ),
                ),
            )
    elif heuristic_mode != "one-step-greedy":
        raise ValueError(f"Unsupported HILP heuristic mode: {heuristic_mode}")
    return Algorithm1ExpansionRecord(
        var_id=var_id,
        item=item,
        expanded=expanded,
        continues=False,
        exact_expanded=exact_expanded,
    )


def _reachable_bellman_frontier_utility(
    expanded: ExpandedAction,
    interface: ANDORSearchInterface,
    heuristic_cache: dict[tuple[object, ...], object],
    *,
    heuristic_lookahead_depth: int,
) -> tuple[float, Fraction | None, bool]:
    r"""Return $$h_q^u$$ from a reachable-state fully observable Bellman table.

    The frontier action $$a_q$$ is already fixed and its exact Algorithm-2
    utility is stored in ``expanded.metrics.utility``.  For each observation
    branch that passes the paper's duration stopping test, its posterior support
    seeds a fully observable future-value table:

    $$h_q^u=u_q+\sum_{o:\tau(qao)>\varsigma}\rho(qao)
      \sum_{s'}b_{qao}(s')V_d(s').$$

    Branches stopped by duration contribute exactly ``u_q`` and no invented
    future reward. / 只对通过 duration stopping condition 的 observation
    branch 加 Bellman 尾值；已终止 branch 不会获得虚构的未来 reward。
    """
    exact_kernel = interface.exact_kernel
    if exact_kernel is None:
        raise ValueError("Reachable Bellman HILP heuristic requires interface.exact_kernel.")
    future_horizon = max(0, heuristic_lookahead_depth)
    continuing_masses: list[Mapping[StateKey, Fraction]] = []
    successor_states: set[StateKey] = set()
    for branch in expanded.observation_frontiers:
        if not branch.should_expand or not branch.child_frontier:
            continue
        # Every action child of one observation node shares the same posterior.
        child = branch.child_frontier[0]
        mass = child.ordinary_mass
        if not mass:
            continue
        continuing_masses.append(mass)
        successor_states.update(
            state
            for state, probability in mass.items()
            if Fraction(probability) > 0
        )
    if not continuing_masses:
        return (
            float(expanded.metrics.utility),
            expanded.metrics.utility_exact,
            expanded.metrics.objective_support_preserved,
        )
    future_values = _reachable_bellman_value_table(
        exact_kernel,
        interface,
        seed_states=tuple(successor_states),
        horizon=future_horizon,
        heuristic_cache=heuristic_cache,
    )
    expected_future = sum(
        (
            Fraction(probability) * Fraction(future_values.get(state, Fraction(0)))
            for mass in continuing_masses
            for state, probability in mass.items()
            if Fraction(probability) > 0
        ),
        start=Fraction(0),
    )
    exact = (
        expanded.metrics.utility_exact
        if expanded.metrics.utility_exact is not None
        else Fraction.from_float(float(expanded.metrics.utility))
    ) + expected_future
    value = float(exact)
    return value, exact, Fraction.from_float(value) == exact


def _reachable_bellman_value_table(
    exact_kernel: object,
    interface: ANDORSearchInterface,
    *,
    seed_states: tuple[StateKey, ...],
    horizon: int,
    heuristic_cache: dict[tuple[object, ...], object],
) -> Mapping[StateKey, Fraction]:
    r"""Return $$V_H$$ over states reachable from ``seed_states``.

    Complexity is $$O(H|S_{reach}||A|d_T)$$ for sparse transitions, not full
    history-tree expansion. / 复杂度按可达状态集合计算，而不是按完整历史树计算。
    """
    if not seed_states:
        return {}
    cache_key = ("bellman_values", _state_set_key(seed_states), horizon)
    if cache_key in heuristic_cache:
        return heuristic_cache[cache_key]  # type: ignore[return-value]
    state_space = _reachable_state_space(
        exact_kernel,
        interface,
        seed_states=seed_states,
        horizon=horizon,
        heuristic_cache=heuristic_cache,
    )
    previous = {state: Fraction(0) for state in state_space}
    for depth in range(1, horizon + 1):
        current: dict[StateKey, Fraction] = {}
        for state in state_space:
            if interface.belief_is_terminal({state: 1.0}):
                current[state] = previous[state]
                continue
            choices = interface.action_choices({state: 1.0})
            action_values = tuple(
                _fully_observable_action_value(
                    exact_kernel,
                    state,
                    choice.label,
                    choice.assignment,
                    previous,
                    heuristic_cache,
                )
                for choice in choices
            )
            if not action_values:
                # Terminal/dead-end states have no continuation.  Keeping the
                # previous tail value is optimistic for the relaxed maximization
                # problem; exact policy flow still excludes nonterminal dead ends.
                current[state] = previous[state]
                continue
            best_continuation = max(action_values)
            # The relaxed MDP may stop with zero value, preserving an
            # optimistic upper bound when all reachable rewards are negative.
            current[state] = max(Fraction(0), best_continuation)
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
            if interface.belief_is_terminal({state: 1.0}):
                continue
            for choice in interface.action_choices({state: 1.0}):
                next_frontier.update(
                    _transition_mass(
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
    future_values: Mapping[StateKey, Fraction],
    heuristic_cache: dict[tuple[object, ...], object],
) -> Fraction:
    r"""Return $$U(s,a)+\sum_{s'}T(s,a,s')V(s')$$. / 返回全可观测 action value。"""
    reward = _state_action_reward(exact_kernel, state, action_label, action, heuristic_cache)
    transition = _transition_mass(
        exact_kernel,
        state,
        action_label,
        action,
        heuristic_cache,
    )
    future = sum(
        (
            Fraction(probability)
            * Fraction(future_values.get(next_state, Fraction(0)))
            for next_state, probability in transition.items()
        ),
        start=Fraction(0),
    )
    return reward + future


def _state_action_reward(
    exact_kernel: object,
    state: StateKey,
    action_label: str,
    action: Mapping[str, object],
    heuristic_cache: dict[tuple[object, ...], object],
) -> Fraction:
    """Return $$U(s,a)$$ through the exact kernel. / 通过 exact kernel 返回 $$U(s,a)$$。"""
    cache_key = ("reward", state, action_label)
    if cache_key in heuristic_cache:
        return Fraction(heuristic_cache[cache_key])  # type: ignore[arg-type]
    _, _, reward = exact_kernel.utility_coefficient_for_mass(  # type: ignore[attr-defined]
        {state: Fraction(1)}, action
    )
    reward = Fraction(reward)
    heuristic_cache[cache_key] = reward
    return reward


def _transition_mass(
    exact_kernel: object,
    state: StateKey,
    action_label: str,
    action: Mapping[str, object],
    heuristic_cache: dict[tuple[object, ...], object],
) -> Mapping[StateKey, Fraction]:
    r"""Return cached $$T(s,a,\cdot)$$. / 返回缓存的转移分布。"""
    cache_key = ("transition", state, action_label)
    if cache_key not in heuristic_cache:
        expansion = exact_kernel.expand_ordinary_mass(  # type: ignore[attr-defined]
            {state: Fraction(1)}, action
        )
        weights = {
            next_state: Fraction(probability)
            for next_state, probability in expansion.post_action_mass.items()
            if Fraction(probability) > 0
        }
        total = sum(weights.values(), start=Fraction(0))
        heuristic_cache[cache_key] = {
            next_state: probability / total
            for next_state, probability in weights.items()
        } if total > 0 else {}
    return heuristic_cache[cache_key]  # type: ignore[return-value]


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
    assignment = item.node.assignment
    if assignment is not None:
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
