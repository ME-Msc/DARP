"""HILP-style partial frontier search for the paper algorithm.

/ 论文算法的 HILP 风格部分 frontier 搜索实现。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from math import isfinite
from time import perf_counter

from darp.adapter.kernel import StateKey
from darp.adapter.runtime import PyRDDLGymRuntime
from darp.ilp.gurobi import GurobiILPSession
from darp.ilp.model import ILPSolveResult
from darp.model.and_or_tree import ANDORSearchInterface
from darp.model.duration import HistoryDurationEvaluator
from darp.planning.decision import ActionDecision
from darp.planning.expand import (
    ExpandedAction,
    apply_terminal_heuristic,
    evaluate_frontier_leaf_metrics,
    expand_frontier_item,
)
from darp.planning.heuristic import (
    UtilityHeuristic,
    history_heuristic_coefficient,
)
from darp.planning.ilp_tree import (
    Algorithm1ExpansionRecord,
    IncrementalPartialTreeILP,
    PolicyTreeILP,
    _action_var_id,
    validate_risk_budget,
)
from darp.planning.policy import extract_conditional_policy
from darp.planning.preprocess import FrontierItem, initialize_root_frontier


@dataclass
class HILPPlanner:
    """Run paper Algorithm 3 frontier expansion.

    / 运行论文 Algorithm 3 风格的 frontier expansion。
    """

    expansion_rounds: int | None = None
    frontier_width: int | None = None
    frontier_heuristic: UtilityHeuristic | None = None
    terminal_heuristic: bool = False
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
        """Choose one action while owning exactly one incremental ILP session.

        / 选择一个根动作；整个 HILP 搜索只持有一个可增量更新的 ILP 会话。
        """
        with GurobiILPSession() as ilp_session:
            return self._choose_action(
                runtime,
                interface,
                duration_evaluator,
                ilp_session,
                root_belief=root_belief,
            )

    def _choose_action(
        self,
        runtime: PyRDDLGymRuntime,
        interface: ANDORSearchInterface,
        duration_evaluator: HistoryDurationEvaluator,
        ilp_session: GurobiILPSession,
        *,
        root_belief: Mapping[StateKey, float] | None = None,
    ) -> ActionDecision:
        r"""Choose an action with HILP-style partial-tree expansion.

        Paper correspondence:

        - Algorithm 3 keeps three sets: expanded action histories $$E$$,
          frontier action histories $$F$$, and not-yet-generated descendants
          $$N$$.
        - Each iteration solves a partial ILP over $$E \cup F$$, then expands
          only incumbent frontier histories with $$x_q>0$$. An external
          heuristic supplies the objective coefficient of histories in $$F$$.
        - DARP's partial ILP keeps the same Definition 3.1 root/flow rows as
          full-ILP for histories in $$E$$. Histories in $$F$$ are frontier
          leaves: they have heuristic $$h_q$$ and risk $$r_q$$ constants,
          but no child-flow rows yet.
        - The CC-POMDP time budget is the domain horizon inside
        `duration_evaluator`; it is consumed by action durations through
          $$\tau(q)$$.  It is not Python wall-clock runtime.

        / 使用 HILP 的 $$E/F$$ frontier 更新框架；heuristic 是 $$F$$ 中节点的
        p-ILP 目标系数，每轮只展开 incumbent 中 $$x_q>0$$ 的 frontier，最终
        root action 也必须来自 incumbent。规划问题
        中的时间预算由 `duration_evaluator` 的 action duration 和 $$\tau(q)$$
        表示，不使用 Python 运行时间。
        """

        started_at = perf_counter()
        if self.expansion_rounds is not None and self.expansion_rounds < 0:
            raise ValueError("expansion_rounds must be non-negative when provided.")
        if self.frontier_width is not None and self.frontier_width < 1:
            raise ValueError("frontier_width must be positive when provided.")
        validate_risk_budget(self.risk_budget)
        if self.solver_time_limit_ms is not None and (
            not isfinite(float(self.solver_time_limit_ms))
            or float(self.solver_time_limit_ms) <= 0.0
        ):
            raise ValueError("solver_time_limit_ms must be finite and positive when provided.")
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
        frontier_records: dict[str, Algorithm1ExpansionRecord] = {}
        ilp_builder = IncrementalPartialTreeILP(
            runtime, interface, risk_budget=self.risk_budget, root_belief=root_belief,
        )
        # Only these events need encoding at the next solve. / 下一轮只编码这些变动。
        pending_expanded: list[Algorithm1ExpansionRecord] = []
        pending_frontier = list(root_frontier)
        partial_tree: PolicyTreeILP | None = None
        partial_result: ILPSolveResult | None = None
        expansion_rounds = 0
        needs_final_solve = True
        solver_limit_hit = False
        # Keep the state corresponding to the most recently solved p-ILP. A
        # later frontier build may consume the remaining wall budget; in that
        # case Algorithm 3 can still return its last valid incumbent.
        # 保存最近一次成功求解 p-ILP 时的状态；若下一轮构树耗尽总时限，
        # Algorithm 3 仍可返回最后一个有效 incumbent（当前最好可行解）。
        solved_frontier_f: dict[str, FrontierItem] | None = None
        solved_expanded_nodes = 0
        solved_expansion_rounds = 0
        timing_totals = {
            "tree_ilp_build_ms": 0.0,
            "warm_start_filter_ms": 0.0,
            "gurobi_solve_ms": 0.0,
            "gurobi_model_update_ms": 0.0,
            "gurobi_optimize_ms": 0.0,
            "partial_ilp_solves": 0.0,
        }

        while frontier_f and (
            self.expansion_rounds is None
            or expansion_rounds < self.expansion_rounds
        ):
            try:
                candidate_tree, candidate_result = self._solve_partial_policy_ilp(
                    interface,
                    duration_evaluator,
                    ilp_builder=ilp_builder,
                    expanded_records=pending_expanded,
                    frontier=pending_frontier,
                    frontier_records=frontier_records,
                    ilp_session=ilp_session,
                    warm_start=(
                        partial_result.variable_values
                        if partial_result is not None
                        else None
                    ),
                    solver_deadline=solver_deadline,
                    timing_totals=timing_totals,
                )
            except TimeoutError:
                if partial_tree is None or partial_result is None:
                    raise
                solver_limit_hit = True
                needs_final_solve = False
                break
            if (
                candidate_result.status == "time_limit"
                and _selected_root_variable(candidate_result, candidate_tree) is None
                and partial_tree is not None
                and partial_result is not None
            ):
                solver_limit_hit = True
                needs_final_solve = False
                break
            partial_tree, partial_result = candidate_tree, candidate_result
            solved_frontier_f = dict(frontier_f)
            solved_expanded_nodes = len(expanded_e)
            solved_expansion_rounds = expansion_rounds
            needs_final_solve = False
            if partial_result.status == "time_limit":
                solver_limit_hit = True
                break
            # Algorithm 3: solve over E U F, then refine only incumbent leaves.
            # Descendants behind F are the implicit N set until materialized.
            # Algorithm 3 每轮在 E∪F 上求解，只细化 incumbent 选中的 frontier；
            # F 后尚未实体化的后代就是隐式集合 N。
            selected = self._selected_frontier(
                partial_tree,
                partial_result,
                frontier_f,
            )
            if not selected:
                break
            expansion_rounds += 1
            for var_id, item in selected:
                frontier_record = frontier_records[var_id]
                if frontier_record.policy_expansion is None:
                    # The p-ILP selected this lazy leaf. Only now run the full
                    # Algorithm-2 observation/duration expansion and publish
                    # its children. / incumbent 选中后才完整生成 observation、
                    # duration 分支和 children。
                    frontier_record = _materialized_frontier_leaf_record(
                        item,
                        interface,
                        duration_evaluator,
                        heuristic=self.frontier_heuristic,
                        terminal_heuristic=self.terminal_heuristic,
                    )
                    frontier_records[var_id] = frontier_record
                expanded_item = frontier_record.policy_expansion
                if expanded_item is None:
                    raise RuntimeError("Selected frontier was not materialized.")
                del frontier_f[var_id]
                expanded_e[var_id] = Algorithm1ExpansionRecord(
                    var_id=var_id,
                    item=item,
                    expanded=expanded_item,
                    continues=bool(expanded_item.child_frontier),
                    policy_expansion=expanded_item,
                )
                pending_expanded.append(expanded_e[var_id])
                for child in expanded_item.child_frontier:
                    child_var_id = _action_var_id(child)
                    if child_var_id not in expanded_e and child_var_id not in frontier_f:
                        frontier_f[child_var_id] = child
                        pending_frontier.append(child)
            needs_final_solve = True

        if partial_tree is None or partial_result is None or needs_final_solve:
            try:
                candidate_tree, candidate_result = self._solve_partial_policy_ilp(
                    interface,
                    duration_evaluator,
                    ilp_builder=ilp_builder,
                    expanded_records=pending_expanded,
                    frontier=pending_frontier,
                    frontier_records=frontier_records,
                    ilp_session=ilp_session,
                    warm_start=(
                        partial_result.variable_values
                        if partial_result is not None
                        else None
                    ),
                    solver_deadline=solver_deadline,
                    timing_totals=timing_totals,
                )
            except TimeoutError:
                if partial_tree is None or partial_result is None:
                    raise
                solver_limit_hit = True
            else:
                if (
                    candidate_result.status == "time_limit"
                    and _selected_root_variable(candidate_result, candidate_tree) is None
                    and partial_tree is not None
                    and partial_result is not None
                ):
                    solver_limit_hit = True
                else:
                    partial_tree, partial_result = candidate_tree, candidate_result
                    solved_frontier_f = dict(frontier_f)
                    solved_expanded_nodes = len(expanded_e)
                    solved_expansion_rounds = expansion_rounds
                    solver_limit_hit = (
                        solver_limit_hit or partial_result.status == "time_limit"
                    )
        solved_frontier = (
            solved_frontier_f if solved_frontier_f is not None else frontier_f
        )
        selected_root = _selected_root_variable(partial_result, partial_tree)
        if selected_root is None:
            if partial_result.status == "time_limit":
                raise TimeoutError(
                    "Gurobi reached the HILP wall budget before finding "
                    "a root incumbent."
                )
            raise RuntimeError(
                "Gurobi HILP partial-tree ILP did not select a root action. "
                f"status={partial_result.status}"
            )
        selected_item = partial_tree.variable_items[selected_root]
        selected_frontier = self._selected_frontier(
            partial_tree,
            partial_result,
            solved_frontier,
        )
        refinement_exhausted = not selected_frontier
        globally_expandable = self._globally_expandable_frontier(
            partial_tree,
            solved_frontier,
            frontier_records,
        )
        certifying_utility_bound = (
            not globally_expandable
            or bool(self.frontier_heuristic and self.frontier_heuristic.upper_bound)
        )
        policy = extract_conditional_policy(partial_tree, partial_result)
        search_complete = (
            partial_result.status == "optimal"
            and not solver_limit_hit
            and refinement_exhausted
            and certifying_utility_bound
            and policy.duration_complete
            and policy.feasible is not False
        )
        # Executability and global search optimality are separate certificates.
        # A duration-complete incumbent has an achieved utility even while
        # unselected alternatives remain unrefined; ``decision.complete`` stays
        # false until the HILP search certificate also closes.
        # 策略可执行性与全局搜索最优性是两种不同的证书：duration-complete
        # incumbent 已有精确实现效用；但只要未选分支尚未细化，
        # ``decision.complete`` 仍为 false。
        achieved_utility = policy.achieved_utility
        decision = ActionDecision(
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
                "warm_start_filter_ms": timing_totals["warm_start_filter_ms"],
                "gurobi_solve_ms": timing_totals["gurobi_solve_ms"],
                "gurobi_model_update_ms": timing_totals["gurobi_model_update_ms"],
                "gurobi_optimize_ms": timing_totals["gurobi_optimize_ms"],
                "partial_ilp_solves": timing_totals["partial_ilp_solves"],
                "ilp_variables": float(len(partial_tree.spec.variables)),
                "ilp_constraints": float(len(partial_tree.spec.constraints)),
                "expanded_nodes": float(solved_expanded_nodes),
                "frontier_nodes": float(len(solved_frontier)),
                "expansion_rounds": float(solved_expansion_rounds),
                "frontier_refinement_exhausted": 1.0 if refinement_exhausted else 0.0,
                "global_expandable_frontier": float(len(globally_expandable)),
                "certifying_utility_bound": 1.0 if certifying_utility_bound else 0.0,
                "solver_time_limit_hit": 1.0 if solver_limit_hit else 0.0,
            },
        )
        return decision

    def _solve_partial_policy_ilp(
        self,
        interface: ANDORSearchInterface,
        duration_evaluator: HistoryDurationEvaluator,
        *,
        ilp_builder: IncrementalPartialTreeILP,
        expanded_records: list[Algorithm1ExpansionRecord],
        frontier: list[FrontierItem],
        frontier_records: dict[str, Algorithm1ExpansionRecord],
        ilp_session: GurobiILPSession,
        warm_start: Mapping[str, float] | None,
        solver_deadline: float | None,
        timing_totals: dict[str, float] | None = None,
    ) -> tuple[PolicyTreeILP, ILPSolveResult]:
        r"""Solve Algorithm 3's current partial-tree p-ILP.

        For every new frontier history $$q\in F$$, DARP computes the risk
        coefficient required by the p-ILP. When the configured terminal
        heuristic makes $$h_q$$ unconditional, observation branches and
        descendants are materialized only after the incumbent selects $$q$$.
        Other heuristic modes retain the eager Algorithm-2 path.

        / 求解当前 $$E\cup F$$ partial-tree p-ILP；可无条件使用 $$h_q$$ 时，
        新 frontier 先只算 risk，被 incumbent 选中后才完整展开。
        """

        build_started_at = perf_counter()
        new_frontier_records: list[Algorithm1ExpansionRecord] = []
        for item in frontier:
            var_id = _action_var_id(item)
            record = _frontier_leaf_record(
                item,
                interface,
                duration_evaluator,
                heuristic=self.frontier_heuristic,
                terminal_heuristic=self.terminal_heuristic,
            )
            frontier_records[var_id] = record
            new_frontier_records.append(record)
        # Update only changed records; retain a separate immutable-in-use
        # checkpoint for timeout fallback.
        # 只增量编码变化的记录；保留不被后续更新修改的独立快照，以便超时回退。
        ilp_builder.update(
            expanded_records=expanded_records,
            frontier_records=new_frontier_records,
        )
        # Model delta means changed coefficients/rows, not the risk budget Delta.
        # 模型差量是新增行和变化系数，不是论文中的风险预算 Delta。
        partial_ilp, model_delta = ilp_builder.snapshot()
        expanded_records.clear()
        frontier.clear()
        build_ms = (perf_counter() - build_started_at) * 1000.0
        if timing_totals is not None:
            timing_totals["tree_ilp_build_ms"] = (
                timing_totals.get("tree_ilp_build_ms", 0.0) + build_ms
            )
        remaining_solver_ms = None
        if solver_deadline is not None:
            remaining_solver_ms = (solver_deadline - perf_counter()) * 1000.0
            if remaining_solver_ms <= 0.0:
                raise TimeoutError("HILP wall budget expired before the next Gurobi refinement.")
        result = ilp_session.solve(
            partial_ilp.spec,
            delta=model_delta,
            time_limit_ms=remaining_solver_ms,
            warm_start=warm_start,
        )
        if timing_totals is not None:
            timing_totals["partial_ilp_solves"] += 1.0
            timing_totals["gurobi_solve_ms"] += float(result.runtime_ms)
            timing_totals["gurobi_model_update_ms"] += ilp_session.last_model_update_ms
            timing_totals["gurobi_optimize_ms"] += ilp_session.last_optimize_ms
        return partial_ilp, result

    def _selected_frontier(
        self,
        partial_tree: PolicyTreeILP,
        partial_result: ILPSolveResult,
        frontier: Mapping[str, FrontierItem],
    ) -> tuple[tuple[str, FrontierItem], ...]:
        r"""Return every frontier history selected by the current p-ILP.

        This is Algorithm 3 lines 12-17: a frontier history enters ``E`` only
        when its incumbent value satisfies :math:`x_q>0`. Terminal frontier
        histories also move to ``E``; this gives the next solve the same final
        expanded set as the reference implementation. ``frontier_width`` is an
        optional batching limit within the selected set.

        / 严格对应 Algorithm 3：先筛选 p-ILP 中 ``x_q>0`` 的 frontier，再在
        该集合内按 heuristic 排序并应用批量宽度。
        """
        frontier_ids = set(partial_tree.frontier_variable_ids)
        incumbent_ids = set(partial_result.selected_variables)
        incumbent_frontier = [
            (var_id, item)
            for var_id, item in frontier.items()
            if var_id in frontier_ids and var_id in incumbent_ids
        ]
        if self.frontier_width is None:
            return tuple(incumbent_frontier)

        selected: list[tuple[float, bool, str, FrontierItem]] = []
        for var_id, item in incumbent_frontier:
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
        selected = selected[: self.frontier_width]
        return tuple((var_id, item) for _, _, var_id, item in selected)

    @staticmethod
    def _globally_expandable_frontier(
        partial_tree: PolicyTreeILP,
        frontier: Mapping[str, FrontierItem],
        frontier_records: Mapping[str, Algorithm1ExpansionRecord],
    ) -> tuple[tuple[str, FrontierItem], ...]:
        """Return every frontier with materializable descendants.

        This set is deliberately independent of the incumbent. It is used for
        completeness certification and therefore also ignores an explicit
        decision-step cap: a duration-feasible child beyond that cap remains
        part of the paper's problem.

        / 返回所有仍有可生成后代的 frontier。该集合与 incumbent 无关，用于
        完整性认证；即使显式 decision-step cap 之外仍有 duration-feasible
        child，它仍属于论文所定义的问题。
        """
        frontier_ids = set(partial_tree.frontier_variable_ids)
        return tuple(
            (var_id, item)
            for var_id, item in frontier.items()
            if var_id in frontier_ids
            and (
                frontier_records[var_id].policy_expansion is None
                or bool(
                    frontier_records[var_id].policy_expansion.child_frontier
                )
            )
        )


def _frontier_leaf_record(
    item: FrontierItem,
    interface: ANDORSearchInterface,
    duration_evaluator: HistoryDurationEvaluator,
    *,
    heuristic: UtilityHeuristic | None,
    terminal_heuristic: bool,
) -> Algorithm1ExpansionRecord:
    """Return a p-ILP leaf, deferring children when its heuristic is unconditional.

    With ``terminal_heuristic=True`` every frontier leaf uses :math:`h_q`, so
    the p-ILP needs only that objective coefficient and the one-step risk.
    Full Algorithm-2 materialization is postponed until the incumbent selects
    the leaf. Other configurations retain the eager path because they must
    inspect continuation branches before deciding between :math:`h_q` and
    :math:`u_q`.

    / 终端也使用启发式时，frontier 先只算 h_q 与一步风险，选中后才展开；
    否则先判断哪些观测继续，再决定使用 h_q 还是 u_q。
    """
    if heuristic is not None and terminal_heuristic:
        return _lazy_frontier_leaf_record(item, interface, heuristic=heuristic)
    return _materialized_frontier_leaf_record(
        item,
        interface,
        duration_evaluator,
        heuristic=heuristic,
        terminal_heuristic=terminal_heuristic,
    )


def _lazy_frontier_leaf_record(
    item: FrontierItem,
    interface: ANDORSearchInterface,
    *,
    heuristic: UtilityHeuristic,
) -> Algorithm1ExpansionRecord:
    """Evaluate frontier coefficients without observations or children. / 不生成观测与后代，仅计算 frontier 系数。"""
    action = item.node.assignment
    if action is None:
        raise ValueError("A frontier action node has no action assignment.")
    kernel = interface.kernel
    if kernel is None:
        raise ValueError("An external HILP heuristic requires a finite kernel.")
    utility = history_heuristic_coefficient(
        heuristic,
        state_mass=item.ordinary_mass,
        action_label=item.action_label,
        action=action,
        non_fluents=kernel.non_fluents,
    )
    metrics = evaluate_frontier_leaf_metrics(
        item,
        interface,
        utility=utility,
    )
    placeholder = ExpandedAction(
        child_frontier=(),
        observation_frontiers=(),
        metrics=metrics,
    )
    return Algorithm1ExpansionRecord(
        var_id=_action_var_id(item),
        item=item,
        expanded=placeholder,
        continues=False,
    )


def _materialized_frontier_leaf_record(
    item: FrontierItem,
    interface: ANDORSearchInterface,
    duration_evaluator: HistoryDurationEvaluator,
    *,
    heuristic: UtilityHeuristic | None,
    terminal_heuristic: bool,
) -> Algorithm1ExpansionRecord:
    r"""Return one p-ILP frontier record.

    An external callback supplies a state utility-to-go.  DARP owns the paper's
    probability weighting and replaces the frontier objective coefficient with

    $$h_q^u=\sum_s \rho(q)b_q(s)h(s,a_q).$$

    Risk remains Algorithm 2's one-step coefficient. Without a callback,
    the one-step utility is a deliberately simple, non-certifying fallback.
    ``terminal_heuristic`` reproduces the paper Grid experiment's convention of
    using the same heuristic at a leaf. For stochastic duration, terminal
    observation branches are weighted separately when sibling branches continue.
    The callback must return the intended value (normally zero) for model-terminal
    states. Otherwise leaves retain RDDL reward.

    / 从真实 Expand 结果构建 frontier：继续分支用概率加权 h_q^u 替换 u_q，
    风险仍用首次失败的一步系数；终端效用遵循实验约定，不改变风险定义。
    """
    var_id = _action_var_id(item)
    policy_expansion = expand_frontier_item(item, interface, duration_evaluator)
    continuation_flags = tuple(
        branch.should_expand for branch in policy_expansion.observation_frontiers
    )
    exact_expansion = (
        apply_terminal_heuristic(item, policy_expansion, interface, heuristic)
        if terminal_heuristic and heuristic is not None
        else policy_expansion
    )
    expanded = exact_expansion
    has_continuation = any(continuation_flags)
    use_heuristic = heuristic is not None and has_continuation
    if use_heuristic:
        action = item.node.assignment
        if action is None:
            raise ValueError("A frontier action node has no action assignment.")
        kernel = interface.kernel
        if kernel is None:
            raise ValueError("An external HILP heuristic requires a finite kernel.")
        utility = history_heuristic_coefficient(
            heuristic,
            state_mass=item.ordinary_mass,
            action_label=item.action_label,
            action=action,
            non_fluents=kernel.non_fluents,
        )
        expanded = replace(
            exact_expansion,
            metrics=replace(
                exact_expansion.metrics,
                utility=utility,
            ),
        )
    return Algorithm1ExpansionRecord(
        var_id=var_id,
        item=item,
        expanded=expanded,
        continues=False,
        # A nonterminal frontier is not an executable policy leaf, so policy
        # validation must inspect its observation branches. At a duration
        # boundary, the optional terminal heuristic is the experiment's actual
        # terminal objective and must therefore be included in achieved utility.
        # 非终止 frontier 尚不可执行；时长终点的启发式则是实验定义的真实终端效用。
        policy_expansion=exact_expansion,
    )


def _is_noop_item(item: FrontierItem) -> bool:
    """Return whether a frontier item has no enabled action.

    / 判断 frontier 是否为 noop。
    """
    assignment = item.node.assignment
    if assignment is not None:
        return not any(bool(value) for value in assignment.values())
    return item.action_label == "noop"


def _selected_root_variable(result: ILPSolveResult, tree: PolicyTreeILP) -> str | None:
    """Return the root action selected by the p-ILP incumbent.

    / 返回 p-ILP incumbent 选中的根动作。
    """
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
