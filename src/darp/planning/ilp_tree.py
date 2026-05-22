"""Generated-tree ILP encoders for full-tree and HILP selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from darp.adapter.runtime import PyRDDLGymRuntime
from darp.ilp.model import ILPLinearConstraint, ILPModelSpec, ILPVariable
from darp.model.and_or_tree import ANDORSearchInterface
from darp.model.duration import HistoryDurationEvaluator
from darp.planning.expand import ExpansionMetrics, expand_frontier_item
from darp.planning.preprocess import FrontierItem, preprocess_search_tree
from darp.planning.rollout import _raise_if_deadline_expired


@dataclass(frozen=True)
class GeneratedPolicyTreeILP:
    """Store a generated policy-tree ILP and variable lookup maps. / 保存生成式 policy-tree ILP 及变量映射。"""

    spec: ILPModelSpec
    variable_items: Mapping[str, FrontierItem]
    variable_metrics: Mapping[str, ExpansionMetrics]
    root_variable_ids: tuple[str, ...]


@dataclass(frozen=True)
class FrontierSelectionILP:
    """Store a HILP frontier-selection p-ILP and lookup maps. / 保存 HILP frontier-selection p-ILP 及映射。"""

    spec: ILPModelSpec
    variable_items: Mapping[str, FrontierItem]


def build_generated_full_tree_ilp(
    runtime: PyRDDLGymRuntime,
    interface: ANDORSearchInterface,
    duration_evaluator: HistoryDurationEvaluator,
    *,
    depth: int,
    risk_budget: float | None = None,
    deadline: float | None = None,
) -> GeneratedPolicyTreeILP:
    r"""Encode the generated AND-OR tree as a binary full-ILP model.

    Paper correspondence:

    - Root policy constraint:
      :math:`\sum_{a \in A(root)} x_{root,a}=1`.
    - Observation-flow constraint for each generated observation node:
      :math:`\sum_{a \in A(qo)} x_{qo,a}=x_{q}`.
    - Objective over generated action histories:
      :math:`\max \sum_q u_q x_q`.
    - Risk row:
      :math:`\sum_q r_q x_q \le \Delta` when a risk budget is provided.

    The current encoder uses pyRDDLGym-generated observation branches; full
    stochastic observation support is a later modeling extension.

    / 将当前生成式 AND-OR tree 编码为二元 full-ILP；目前 observation 分支来自
    pyRDDLGym 采样/确定性生成，完整随机 observation 枚举留给后续扩展。
    """

    if depth < 1:
        raise ValueError("depth must be at least 1.")
    tree = preprocess_search_tree(runtime, interface)
    queue: list[FrontierItem] = list(tree.frontier)
    variables: dict[str, ILPVariable] = {}
    objective: dict[str, float] = {}
    constraints: list[ILPLinearConstraint] = []
    variable_items: dict[str, FrontierItem] = {}
    variable_metrics: dict[str, ExpansionMetrics] = {}
    root_ids: list[str] = []

    while queue:
        _raise_if_deadline_expired(deadline)
        item = queue.pop(0)
        var_id = _action_var_id(item)
        if var_id in variables:
            continue
        variables[var_id] = ILPVariable(
            var_id=var_id,
            label=item.node.history.label(),
            metadata={
                "action": item.action_label,
                "root_action": item.root_label,
                "node_id": item.node.node_id,
                "history": item.node.history.label(),
            },
        )
        variable_items[var_id] = item
        if item.node.history.depth == 1:
            root_ids.append(var_id)

        expanded = expand_frontier_item(item, interface, duration_evaluator)
        variable_metrics[var_id] = expanded.metrics
        objective[var_id] = expanded.metrics.utility
        if item.node.history.depth < depth and expanded.child_frontier:
            child_ids = [_action_var_id(child) for child in expanded.child_frontier]
            coefficients = {child_id: 1.0 for child_id in child_ids}
            coefficients[var_id] = coefficients.get(var_id, 0.0) - 1.0
            constraints.append(
                ILPLinearConstraint(
                    name=f"flow_{var_id}",
                    coefficients=coefficients,
                    sense="==",
                    rhs=0.0,
                )
            )
            queue.extend(expanded.child_frontier)

    if not root_ids:
        raise ValueError("Generated policy tree has no root action variables.")
    constraints.insert(
        0,
        ILPLinearConstraint(
            name="root_action",
            coefficients={var_id: 1.0 for var_id in root_ids},
            sense="==",
            rhs=1.0,
        ),
    )
    if risk_budget is not None:
        constraints.append(
            ILPLinearConstraint(
                name="risk_budget",
                coefficients={
                    var_id: metrics.risk
                    for var_id, metrics in variable_metrics.items()
                    if abs(metrics.risk) > 1e-12
                },
                sense="<=",
                rhs=float(risk_budget),
            )
        )
    spec = ILPModelSpec(
        name="darp_full_tree",
        variables=tuple(variables.values()),
        objective=objective,
        constraints=tuple(constraints),
    )
    return GeneratedPolicyTreeILP(
        spec=spec,
        variable_items=variable_items,
        variable_metrics=variable_metrics,
        root_variable_ids=tuple(root_ids),
    )


def build_frontier_selection_ilp(
    scored_frontier: Sequence[tuple[float, FrontierItem]],
    *,
    frontier_width: int,
) -> FrontierSelectionILP:
    r"""Encode HILP frontier selection as a small p-ILP.

    This is the Phase 8 p-ILP hook for Algorithm 3: select up to
    ``frontier_width`` frontier histories with the highest current score.

    .. math::

       \max \sum_{q \in F} score(q) y_q
       \quad
       1 \le \sum_{q \in F} y_q \le k

    / 将 HILP frontier 选择编码为小型 p-ILP。
    """

    if frontier_width < 1:
        raise ValueError("frontier_width must be at least 1.")
    variables: dict[str, ILPVariable] = {}
    objective: dict[str, float] = {}
    items: dict[str, FrontierItem] = {}
    for index, (score, item) in enumerate(scored_frontier):
        var_id = f"frontier_{index}_{_node_token(item.node.node_id)}"
        variables[var_id] = ILPVariable(
            var_id=var_id,
            label=item.node.history.label(),
            metadata={
                "action": item.action_label,
                "root_action": item.root_label,
                "score": score,
            },
        )
        objective[var_id] = float(score)
        items[var_id] = item
    constraints: list[ILPLinearConstraint] = []
    if variables:
        coefficients = {var_id: 1.0 for var_id in variables}
        constraints.extend(
            [
                ILPLinearConstraint(
                    name="frontier_width",
                    coefficients=coefficients,
                    sense="<=",
                    rhs=float(min(frontier_width, len(variables))),
                ),
                ILPLinearConstraint(
                    name="select_at_least_one",
                    coefficients=coefficients,
                    sense=">=",
                    rhs=1.0,
                ),
            ]
        )
    return FrontierSelectionILP(
        spec=ILPModelSpec(
            name="darp_hilp_frontier",
            variables=tuple(variables.values()),
            objective=objective,
            constraints=tuple(constraints),
        ),
        variable_items=items,
    )


def _action_var_id(item: FrontierItem) -> str:
    """Return a stable ILP variable id for an action-history item. / 返回 action-history 的稳定 ILP 变量 id。"""
    return f"x_{_node_token(item.node.node_id)}"


def _node_token(value: str) -> str:
    """Return a Gurobi-friendly token. / 返回适合 Gurobi 的 token。"""
    return "".join(char if char.isalnum() else "_" for char in value) or "empty"
