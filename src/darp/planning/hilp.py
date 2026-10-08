"""HILP-style partial frontier search for the paper algorithm.

/ 论文算法的 HILP 风格部分 frontier 搜索实现。
"""

from __future__ import annotations

import logging
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
    ExpansionMetrics,
    apply_terminal_heuristic,
    expand_frontier_item,
)
from darp.planning.heuristic import (
    UtilityHeuristic,
    history_heuristic_coefficient,
)
from darp.planning.ilp_tree import (
    HistoryRecord,
    IncrementalPartialTreeILP,
    PolicyTreeILP,
    _action_var_id,
    validate_risk_budget,
)
from darp.planning.policy import extract_conditional_policy
from darp.planning.preprocess import FrontierItem, initialize_root_frontier
from darp.planning.rank import restrict_rank_candidates, validate_rank

logger = logging.getLogger(__name__)


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
    rank_alpha: float = 1.0
    rank_lambda: float = 1.0

    def choose_action(
        self,
        runtime: PyRDDLGymRuntime,
        interface: ANDORSearchInterface,
        duration_evaluator: HistoryDurationEvaluator,
        *,
        root_belief: Mapping[StateKey, float] | None = None,
    ) -> ActionDecision:
        """Choose one action using an incremental or candidate-restricted session.

        / 原 HILP 复用增量模型；启用候选筛选时，在同一会话内重建缩减模型。
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
        r"""Run Algorithm 3 with E/F histories and solution vector x.

        E holds refined action histories; F holds p-ILP frontier histories.
        ``solution.variable_values`` is x. Only selected F histories enter E.
        The paper's N holds observation histories awaiting action generation;
        DARP generates their actions immediately and queues those in
        ``pending_frontier``. It does not store the ungenerated whole tree.
        Duration uses tau(q) and the model horizon, not solver wall time.

        / E 保存已细化动作历史，F 保存 p-ILP frontier，solution.variable_values
        对应 x。只把当前解选中的 F 节点移入 E。论文 N 是待生成后续动作的
        观测历史；代码直接生成其动作放入 pending_frontier，不另外维护 N。
        duration 按 tau(q) 与问题 horizon 判断，不使用求解器运行时间。
        """

        started_at = perf_counter()
        if self.expansion_rounds is not None and self.expansion_rounds < 0:
            raise ValueError("expansion_rounds must be non-negative when provided.")
        if self.frontier_width is not None and self.frontier_width < 1:
            raise ValueError("frontier_width must be positive when provided.")
        validate_risk_budget(self.risk_budget)
        validate_rank(self.rank_alpha, self.rank_lambda)
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
        # E holds refined history ids; cached Expand alone does not move F to E.
        # E 保存已细化历史的 id；为估值预计算 Expand 不等于已经从 F 移入 E。
        expanded_e: set[str] = set()
        frontier_records: dict[str, HistoryRecord] = {}
        ilp_builder = IncrementalPartialTreeILP(
            runtime, interface, risk_budget=self.risk_budget, root_belief=root_belief,
        )
        # Only these events need encoding at the next solve. / 下一轮只编码这些变动。
        pending_expanded: list[HistoryRecord] = []
        pending_frontier = list(root_frontier)
        partial_ilp: PolicyTreeILP | None = None
        solution: ILPSolveResult | None = None
        expansion_rounds = 0
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
            "rank_filter_ms": 0.0,
            "rank_fallbacks": 0.0,
        }

        # Algorithm 3: solve -> select x_q=1 in F -> Expand -> update E/F.
        # 论文主循环：求解 p-ILP -> 读取被选 frontier -> 展开 -> 更新 E/F。
        # Always solve the last refinement, even at the expansion-round limit.
        # 即使达到展开轮数上限，也要先求解最后一次展开后的模型。
        while True:
            try:
                candidate_tree, candidate_solution = self._solve_partial_policy_ilp(
                    interface,
                    duration_evaluator,
                    ilp_builder=ilp_builder,
                    expanded_records=pending_expanded,
                    frontier=pending_frontier,
                    frontier_records=frontier_records,
                    ilp_session=ilp_session,
                    warm_start=(
                        solution.variable_values
                        if solution is not None
                        else None
                    ),
                    solver_deadline=solver_deadline,
                    timing_totals=timing_totals,
                )
            except TimeoutError:
                if partial_ilp is None or solution is None:
                    raise
                logger.warning(
                    "HILP refinement timed out; returning the last solved policy "
                    "(rounds=%d, expanded_nodes=%d).",
                    solved_expansion_rounds, solved_expanded_nodes, exc_info=True,
                )
                solver_limit_hit = True
                break
            if (
                candidate_solution.status == "time_limit"
                and _selected_root_variable(candidate_solution, candidate_tree) is None
                and partial_ilp is not None
                and solution is not None
            ):
                logger.warning(
                    "HILP solve reached its time limit without a root incumbent; "
                    "returning the last solved policy (rounds=%d, expanded_nodes=%d).",
                    solved_expansion_rounds, solved_expanded_nodes,
                )
                solver_limit_hit = True
                break
            partial_ilp, solution = candidate_tree, candidate_solution
            solved_frontier_f = dict(frontier_f)
            solved_expanded_nodes = len(expanded_e)
            solved_expansion_rounds = expansion_rounds
            if solution.status == "time_limit":
                solver_limit_hit = True
                break
            if not frontier_f or (
                self.expansion_rounds is not None
                and expansion_rounds >= self.expansion_rounds
            ):
                break
            # Algorithm 3, lines 12-17: refine selected frontier histories.
            # / 论文第 12-17 行：只细化当前解选中的 frontier 历史。
            selected = self._selected_frontier(
                partial_ilp,
                solution,
                frontier_f,
            )
            if not selected:
                break
            expansion_rounds += 1
            for var_id, item in selected:
                expanded_item = frontier_records[var_id].policy_expansion
                if expanded_item is None:
                    # The p-ILP selected this lazy leaf. Only now run the full
                    # Algorithm-2 observation/duration expansion and publish
                    # its children. / incumbent 选中后才完整生成 observation、
                    # duration 分支和 children。
                    expanded_item = expand_frontier_item(
                        item, interface, duration_evaluator,
                    )
                    if self.terminal_heuristic and self.frontier_heuristic is not None:
                        expanded_item = apply_terminal_heuristic(
                            item, expanded_item, interface, self.frontier_heuristic,
                        )
                # F -> E: replace (h_q^u, h_q^r) by (u_q, r_q).
                # / 从 F 移入 E：用实际系数替换估值，随后编码各观测的后续动作。
                del frontier_f[var_id]
                expanded_record = HistoryRecord(
                    var_id=var_id,
                    item=item,
                    ilp_metrics=expanded_item.metrics,
                    continues=bool(expanded_item.child_frontier),
                    policy_expansion=expanded_item,
                )
                frontier_records[var_id] = expanded_record
                expanded_e.add(var_id)
                pending_expanded.append(expanded_record)
                for child in expanded_item.child_frontier:
                    child_var_id = _action_var_id(child)
                    if child_var_id not in expanded_e and child_var_id not in frontier_f:
                        frontier_f[child_var_id] = child
                        pending_frontier.append(child)

        return self._build_decision(
            partial_ilp,
            solution,
            solved_frontier=solved_frontier_f if solved_frontier_f is not None else frontier_f,
            frontier_records=frontier_records,
            solved_expanded_nodes=solved_expanded_nodes,
            solved_expansion_rounds=solved_expansion_rounds,
            solver_limit_hit=solver_limit_hit,
            timing_totals=timing_totals,
        )

    def _build_decision(
        self,
        partial_ilp: PolicyTreeILP,
        solution: ILPSolveResult,
        *,
        solved_frontier: Mapping[str, FrontierItem],
        frontier_records: Mapping[str, HistoryRecord],
        solved_expanded_nodes: int,
        solved_expansion_rounds: int,
        solver_limit_hit: bool,
        timing_totals: Mapping[str, float],
    ) -> ActionDecision:
        """Extract the last solved policy and report its search certificate.

        / 从最后一次已求解的树提取策略、判断完整性；不改变搜索或 ILP。
        """
        selected_root = _selected_root_variable(solution, partial_ilp)
        if selected_root is None:
            if solution.status == "time_limit":
                raise TimeoutError(
                    "Gurobi reached the HILP wall budget before finding "
                    "a root incumbent."
                )
            raise RuntimeError(
                "Gurobi HILP partial-tree ILP did not select a root action. "
                f"status={solution.status}"
            )
        selected_item = partial_ilp.variable_items[selected_root]
        selected_frontier = self._selected_frontier(
            partial_ilp,
            solution,
            solved_frontier,
        )
        refinement_exhausted = not selected_frontier
        globally_expandable_count = self._count_globally_expandable_frontier(
            partial_ilp,
            solved_frontier,
            frontier_records,
        )
        certifying_utility_bound = (
            not globally_expandable_count
            or bool(self.frontier_heuristic and self.frontier_heuristic.upper_bound)
        )
        policy = extract_conditional_policy(partial_ilp, solution)
        restricted = len(partial_ilp.spec.variables) < len(partial_ilp.variable_items)
        search_complete = (
            solution.status == "optimal"
            and not restricted
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
                else (solution.objective_value or 0.0)
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
                "rank_alpha": self.rank_alpha,
                "rank_lambda": self.rank_lambda,
                "rank_filter_ms": timing_totals.get("rank_filter_ms", 0.0),
                "rank_fallbacks": timing_totals.get("rank_fallbacks", 0.0),
                "rank_restricted": float(restricted),
                # Cumulative submitted/full sizes include every solve and fallback.
                # 累计提交/完整规模包含所有中间求解与回退，比仅最终一轮更能解释耗时。
                **{key: timing_totals.get(key, 0.0) for key in (
                    "ilp_e_sent_total", "ilp_e_full_total",
                    "ilp_f_sent_total", "ilp_f_full_total",
                )},
                "full_partial_ilp_variables": float(len(partial_ilp.variable_items)),
                "ilp_variables": float(len(partial_ilp.spec.variables)),
                "ilp_constraints": float(len(partial_ilp.spec.constraints)),
                "expanded_nodes": float(solved_expanded_nodes),
                "frontier_nodes": float(len(solved_frontier)),
                "expansion_rounds": float(solved_expansion_rounds),
                "frontier_refinement_exhausted": 1.0 if refinement_exhausted else 0.0,
                "global_expandable_frontier": float(globally_expandable_count),
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
        expanded_records: list[HistoryRecord],
        frontier: list[FrontierItem],
        frontier_records: dict[str, HistoryRecord],
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
        new_frontier_records: list[HistoryRecord] = []
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
        solve_spec = partial_ilp.spec
        if self.rank_lambda < 1.0:
            filter_started_at = perf_counter()
            solve_spec = restrict_rank_candidates(
                solve_spec,
                alpha=self.rank_alpha,
                lambda_=self.rank_lambda,
                frontier_variable_ids=set(partial_ilp.frontier_variable_ids),
                warm_start=warm_start,
            )
            if timing_totals is not None:
                timing_totals["rank_filter_ms"] += (perf_counter() - filter_started_at) * 1000.0

        while True:
            if self.rank_lambda < 1.0:
                # Candidate sets can shrink or change; the append-only delta is
                # valid only for the original full p-ILP. Keep that fast path.
                # 候选集可能缩小或变化，不能应用原模型的追加差量；lambda=1 保留增量路径。
                ilp_session.close()
            remaining_solver_ms = None
            if solver_deadline is not None:
                remaining_solver_ms = (solver_deadline - perf_counter()) * 1000.0
                if remaining_solver_ms <= 0.0:
                    raise TimeoutError("HILP wall budget expired before the next Gurobi refinement.")
            result = ilp_session.solve(
                solve_spec,
                delta=model_delta if self.rank_lambda == 1.0 else None,
                time_limit_ms=remaining_solver_ms,
                warm_start=warm_start,
            )
            if timing_totals is not None:
                frontier_ids = (partial_ilp.frontier_variable_ids if solve_spec is partial_ilp.spec
                                else set(partial_ilp.frontier_variable_ids))
                # Full submissions need no variable scan. / 完整提交用集合大小，避免给原 HILP 增加扫描。
                sent_f = (len(frontier_ids) if solve_spec is partial_ilp.spec else
                          sum(variable.var_id in frontier_ids for variable in solve_spec.variables))
                sizes = {
                    "ilp_e_sent_total": len(solve_spec.variables) - sent_f,
                    "ilp_e_full_total": len(partial_ilp.spec.variables) - len(frontier_ids),
                    "ilp_f_sent_total": sent_f,
                    "ilp_f_full_total": len(frontier_ids),
                }
                for key, count in sizes.items():
                    timing_totals[key] = timing_totals.get(key, 0.0) + count
                timing_totals["partial_ilp_solves"] += 1.0
                timing_totals["gurobi_solve_ms"] += float(result.runtime_ms)
                timing_totals["gurobi_model_update_ms"] += ilp_session.last_model_update_ms
                timing_totals["gurobi_optimize_ms"] += ilp_session.last_optimize_ms
            if (
                len(solve_spec.variables) < len(partial_ilp.spec.variables)
                and result.status in {"infeasible", "infeasible_or_unbounded"}
            ):
                # Restriction failure does not prove the current p-ILP infeasible.
                # 筛选后的模型无解不等于原问题无解；在同一总时限内回退完整当前 p-ILP。
                logger.warning("Restricted p-ILP is %s; retrying the full current p-ILP.", result.status)
                solve_spec = partial_ilp.spec
                if timing_totals is not None:
                    timing_totals["rank_fallbacks"] += 1.0
                continue
            # Full history maps are still needed to reconstruct the policy.
            # 只替换提交给求解器的模型，保留完整历史索引供策略重建使用。
            return replace(partial_ilp, spec=solve_spec), result

    def _selected_frontier(
        self,
        partial_ilp: PolicyTreeILP,
        solution: ILPSolveResult,
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
        frontier_ids = set(partial_ilp.frontier_variable_ids)
        incumbent_ids = set(solution.selected_variables)
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
            score = float(partial_ilp.spec.objective.get(var_id, 0.0))
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
    def _count_globally_expandable_frontier(
        partial_ilp: PolicyTreeILP,
        frontier: Mapping[str, FrontierItem],
        frontier_records: Mapping[str, HistoryRecord],
    ) -> int:
        """Count frontier histories with materializable descendants.

        This count is deliberately independent of the incumbent. It is used for
        completeness certification and therefore also ignores an explicit
        decision-step cap: a duration-feasible child beyond that cap remains
        part of the paper's problem.

        / 统计所有仍有可生成后代的 frontier。该计数与 incumbent 无关，用于
        完整性认证；即使显式 decision-step cap 之外仍有 duration-feasible
        child，它仍属于论文所定义的问题。
        """
        frontier_ids = set(partial_ilp.frontier_variable_ids)
        return sum(
            1
            for var_id in frontier
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
) -> HistoryRecord:
    r"""Build one F record with coefficients (h_q^u, h_q^r).

    When terminal leaves also use the heuristic, defer Expand until x_q=1.
    Otherwise inspect the branches now: continuing leaves use h_q^u, stopped
    leaves keep u_q. The one-step first-failure coefficient bounds future risk.
    Without a heuristic, u_q is a non-certifying fallback for h_q^u.

    / 生成 F 中的一条记录：终端也使用启发式时，只计算估值，待 x_q=1
    后才展开；否则先检查分支，继续节点用 h_q^u，停止节点保留 u_q。
    一步首次失败概率作为风险下界；没有启发式时，u_q 不保证是效用上界。
    """
    var_id = _action_var_id(item)
    policy_expansion = None
    if heuristic is None or not terminal_heuristic:
        policy_expansion = expand_frontier_item(item, interface, duration_evaluator)
        if heuristic is None or not any(
            branch.should_expand for branch in policy_expansion.observation_frontiers
        ):
            return HistoryRecord(
                var_id=var_id,
                item=item,
                ilp_metrics=policy_expansion.metrics,
                continues=False,
                policy_expansion=policy_expansion,
            )

    action = item.node.assignment
    if action is None:
        raise ValueError("A frontier action node has no action assignment.")
    kernel = interface.kernel
    if kernel is None:
        raise ValueError("An external HILP heuristic requires a finite kernel.")
    h_u_q = history_heuristic_coefficient(
        heuristic,
        state_mass=item.ordinary_mass,
        action_label=item.action_label,
        action=action,
        non_fluents=kernel.non_fluents,
    )
    metrics = ExpansionMetrics(
        utility=h_u_q,
        chance_risk=(
            kernel.first_failure_coefficient(item.safe_mass, action)
            if policy_expansion is None
            else policy_expansion.metrics.chance_risk
        ),
    )
    return HistoryRecord(
        var_id=var_id,
        item=item,
        ilp_metrics=metrics,
        continues=False,
        # Keep actual branches separate from the ILP estimate. / 真实分支与 ILP 估值分开保存。
        policy_expansion=policy_expansion,
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
