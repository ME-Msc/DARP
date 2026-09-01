"""Gurobi full-ILP planner for the paper's policy-tree objective."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from time import perf_counter

from darp.adapter.kernel import StateKey
from darp.adapter.runtime import PyRDDLGymRuntime
from darp.ilp.gurobi import GurobiILPSolver
from darp.ilp.model import ILPSolveResult
from darp.model.and_or_tree import ANDORSearchInterface
from darp.model.duration import HistoryDurationEvaluator
from darp.planning.decision import ActionDecision
from darp.planning.ilp_tree import (
    PolicyTreeILP,
    build_full_tree_ilp,
    validate_risk_budget,
)
from darp.planning.policy import extract_conditional_policy


@dataclass
class FullILPPlanner:
    """Solve the full policy-tree ILP with Gurobi. / 使用 Gurobi 求解完整 policy-tree ILP。"""

    risk_budget: float | None = None
    max_tree_nodes: int | None = 100_000
    solver_time_limit_ms: float | None = 60_000.0

    def choose_action(
        self,
        runtime: PyRDDLGymRuntime,
        interface: ANDORSearchInterface,
        duration_evaluator: HistoryDurationEvaluator,
        *,
        root_belief: Mapping[StateKey, float] | None = None,
    ) -> ActionDecision:
        r"""Choose the root action by solving the full-ILP.

        Paper correspondence:

        - Algorithm 1 initializes the root history and repeatedly calls
          Algorithm 2 (`Expand`) until all histories that violate the duration
          stopping test have been expanded. DARP implements this tree-building
          phase in `build_full_tree_ilp(...)`. With a finite kernel,
          this tree uses finite transition/observation branches from the
          pyRDDLGym grounded CPFs.

        - Algorithm 2 computes the constants for each action history $$q \in \tilde{A}$$:

          $$u_q = \rho(q)\sum_s b_q(s)U(s,a_q),\qquad
            r_q = \tilde\rho(q)r(\tilde b_q,a_q),\qquad
            \tau(q).$$

        - The full-ILP then solves the paper's policy-tree program:

          $$\max_x \sum_{q \in \tilde{A}: \tau(q-1)>\varsigma} u_q x_q$$

          subject to the root and observation-flow constraints:

          $$\sum_{a\in A} x_a = 1,\qquad
            \sum_{a\in A} x_{qoa} = x_q,$$

          plus the optional Lemma 3.3 chance-constrained risk row:

          $$\sum_q r_q x_q \le R.$$

        按论文 Algorithm 1/2 生成完整 policy tree；有限 kernel 使用
        pyRDDLGym grounded CPF 的有限 transition/observation 分支，然后直接
        用 Gurobi 直接求解 full-ILP。

        Reference-code correspondence:

        - Author ``solver.preprocess(...)`` builds a NetworkX AND-OR tree and
          stores $$u_q$$, $$r_q$$, occurrence probabilities, and beliefs on tree
          nodes. DARP's equivalent is
          ``build_full_tree_ilp -> paper_preprocess -> expand_frontier_item``.
        - Author ``solver.ILP(...)`` creates binary variables ``x[q]`` for
          action histories, then adds ``tree_c1``, ``tree_c{q}``, and
          ``capacity_c``. DARP encodes the same rows as ``root_action``,
          ``flow_*``, and ``risk_budget`` before calling `GurobiILPSolver`.

        / 作者代码中的 `preprocess` 与 `ILP` 在 DARP 中分别对应 tree generation
        与 ILP encoding 两步；变量和约束名称不同，但数学结构相同。
        """

        validate_risk_budget(self.risk_budget)
        if self.solver_time_limit_ms is not None and (
            not isfinite(float(self.solver_time_limit_ms))
            or float(self.solver_time_limit_ms) <= 0.0
        ):
            raise ValueError("solver_time_limit_ms must be finite and positive when provided.")

        build_started_at = perf_counter()
        ilp_tree = build_full_tree_ilp(
            runtime,
            interface,
            duration_evaluator,
            risk_budget=self.risk_budget,
            root_belief=root_belief,
            max_nodes=self.max_tree_nodes,
        )
        tree_ilp_build_ms = (perf_counter() - build_started_at) * 1000.0
        ilp_result = GurobiILPSolver().solve(
            ilp_tree.spec,
            time_limit_ms=self.solver_time_limit_ms,
        )
        selected_root = _selected_root_variable(ilp_result, ilp_tree)
        if selected_root is None:
            if ilp_result.status == "time_limit":
                raise TimeoutError("Gurobi reached the full-ILP solve budget before finding an incumbent.")
            raise RuntimeError(
                "Gurobi full-tree ILP did not select a root action. "
                f"status={ilp_result.status}"
            )

        selected_item = ilp_tree.variable_items[selected_root]
        policy = extract_conditional_policy(ilp_tree, ilp_result)
        decision_complete = (
            ilp_result.status == "optimal"
            and policy.duration_complete
            and policy.feasible is not False
        )
        # A duration-complete incumbent is executable and its selected
        # coefficients are achieved utility even when optimality has not yet
        # been proved (for example at a solver time limit).
        achieved_utility = policy.achieved_utility
        gurobi_ms = float(ilp_result.runtime_ms)
        return ActionDecision(
            action=dict(selected_item.node.assignment or {}),
            label=selected_item.action_label,
            value=float(
                achieved_utility
                if achieved_utility is not None
                else (ilp_result.objective_value or 0.0)
            ),
            complete=decision_complete,
            value_kind=(
                "achieved_utility"
                if achieved_utility is not None
                else "heuristic_objective"
            ),
            policy=policy,
            timing={
                "tree_ilp_build_ms": tree_ilp_build_ms,
                "gurobi_solve_ms": gurobi_ms,
                "ilp_variables": float(len(ilp_tree.spec.variables)),
                "ilp_constraints": float(len(ilp_tree.spec.constraints)),
                "expanded_nodes": float(len(ilp_tree.variable_items)),
                "solver_time_limit_hit": (
                    1.0 if ilp_result.status == "time_limit" else 0.0
                ),
            },
        )


def _selected_root_variable(result: ILPSolveResult, tree: PolicyTreeILP) -> str | None:
    """Return the selected root action variable id. / 返回被选中的根 action 变量 id。"""
    root_ids = set(tree.root_variable_ids)
    return next((var_id for var_id in result.selected_variables if var_id in root_ids), None)
