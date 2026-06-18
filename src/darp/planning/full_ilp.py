"""Gurobi full-ILP planner for the paper's policy-tree objective."""

# TODO(phase-9.1): Add benchmark-scale pruning for large exact finite kernels
# and richer finite random variables beyond the current supported subset.

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Mapping

from darp.adapter.exact import StateKey
from darp.adapter.runtime import PyRDDLGymRuntime
from darp.ilp.gurobi import GurobiILPSolver
from darp.ilp.model import ILPSolveResult
from darp.model.and_or_tree import ANDORSearchInterface
from darp.model.duration import HistoryDurationEvaluator
from darp.planning.ilp_tree import PolicyTreeILP, build_full_tree_ilp
from darp.planning.rollout import ActionDecision


@dataclass
class FullILPPlanner:
    """Solve the full policy-tree ILP with Gurobi. / 使用 Gurobi 求解完整 policy-tree ILP。"""

    risk_budget: float | None = None
    name: str = "full-ilp-gurobi"
    last_ilp_result: ILPSolveResult | None = field(default=None, init=False)
    last_policy_tree: PolicyTreeILP | None = field(default=None, init=False)

    def choose_action(
        self,
        runtime: PyRDDLGymRuntime,
        interface: ANDORSearchInterface,
        duration_evaluator: HistoryDurationEvaluator,
        *,
        remaining_depth: int,
        root_belief: Mapping[StateKey, float] | None = None,
    ) -> ActionDecision:
        r"""Choose the root action by solving the full-ILP.

        Paper correspondence:

        - Algorithm 1 initializes the root history and repeatedly calls
          Algorithm 2 (`Expand`) until all histories that violate the duration
          stopping test have been expanded. DARP implements this tree-building
          phase in `build_full_tree_ilp(...)`. With an exact kernel,
          this tree uses exact finite transition/observation branches from the
          pyRDDLGym grounded CPFs.

        - Algorithm 2 computes the constants for each action history $$q \in \tilde{A}$$:

          $$u_q = \rho^*(q)\sum_s b^*_{q-1}(s)U(s,a_q),\qquad
            r_q = \rho^*(q)r(b_q),\qquad
            \tau(q).$$

        - The full-ILP then solves the paper's policy-tree program:

          $$\max_x \sum_{q \in \tilde{A}: \tau(q-1)>\varsigma} u_q x_q$$

          subject to the root and observation-flow constraints:

          $$\sum_{a\in A} x_a = 1,\qquad
            \sum_{a\in A} x_{qoa} = x_q,$$

          plus the optional Lemma 3.3 chance-constrained risk row:

          $$\sum_q r_q x_q \le R.$$

        按论文 Algorithm 1/2 生成完整 policy tree；有 exact kernel 时使用
        pyRDDLGym grounded CPF 的有限 transition/observation 精确分支，然后直接
        用 Gurobi 求解 full-ILP；这里没有递归 DP 或 rollout fallback。

        Reference-code correspondence:

        - Author ``solver.preprocess(...)`` builds a NetworkX AND-OR tree and
          stores $$u_q$$, $$r_q$$, $$\rho^*(q)$$, and beliefs on tree
          nodes. DARP's equivalent is
          ``build_full_tree_ilp -> paper_preprocess -> expand_frontier_item``.
        - Author ``solver.ILP(...)`` creates binary variables ``x[q]`` for
          action histories, then adds ``tree_c1``, ``tree_c{q}``, and
          ``capacity_c``. DARP encodes the same rows as ``root_action``,
          ``flow_*``, and ``risk_budget`` before calling `GurobiILPSolver`.

        / 作者代码中的 `preprocess` 与 `ILP` 在 DARP 中分别对应 tree generation
        与 ILP encoding 两步；变量和约束名称不同，但数学结构相同。
        """

        started_at = perf_counter()
        if remaining_depth < 1:
            raise ValueError("remaining_depth must be at least 1.")

        ilp_tree = build_full_tree_ilp(
            runtime.clone(),
            interface,
            duration_evaluator,
            risk_budget=self.risk_budget,
            root_belief=root_belief,
        )
        self.last_policy_tree = ilp_tree
        self.last_ilp_result = GurobiILPSolver().solve(ilp_tree.spec)
        selected_root = _selected_root_variable(self.last_ilp_result, ilp_tree)
        if selected_root is None:
            raise RuntimeError(
                "Gurobi full-tree ILP did not select a root action. "
                f"status={self.last_ilp_result.status}"
            )

        selected_item = ilp_tree.variable_items[selected_root]
        elapsed_ms = (perf_counter() - started_at) * 1000.0
        return ActionDecision(
            action=dict(selected_item.node.metadata["assignment"]),
            label=selected_item.action_label,
            value=float(self.last_ilp_result.objective_value or 0.0),
            action_values=_root_objective_values(ilp_tree),
            remaining_depth=remaining_depth,
            elapsed_ms=elapsed_ms,
            complete=self.last_ilp_result.is_optimal,
        )


def _selected_root_variable(result: ILPSolveResult, tree: PolicyTreeILP) -> str | None:
    """Return the selected root action variable id. / 返回被选中的根 action 变量 id。"""
    root_ids = set(tree.root_variable_ids)
    return next((var_id for var_id in result.selected_variables if var_id in root_ids), None)


def _root_objective_values(tree: PolicyTreeILP) -> dict[str, float]:
    """Return root-action utility coefficients for diagnostics. / 返回根 action utility 系数用于诊断。"""
    values: dict[str, float] = {}
    for var_id in tree.root_variable_ids:
        item = tree.variable_items[var_id]
        values[item.action_label] = float(tree.spec.objective.get(var_id, 0.0))
    return values
