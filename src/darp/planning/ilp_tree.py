"""Policy-tree ILP encoders for full-tree and HILP selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from darp.adapter.runtime import PyRDDLGymRuntime
from darp.adapter.exact import StateKey
from darp.ilp.model import ILPLinearConstraint, ILPModelSpec, ILPVariable
from darp.model.and_or_tree import ANDORSearchInterface
from darp.model.duration import HistoryDurationEvaluator
from darp.planning.expand import ExpandedAction, ExpansionMetrics, expand_frontier_item
from darp.planning.preprocess import FrontierItem, initialize_root_frontier
from darp.planning.rollout import _raise_if_deadline_expired


@dataclass(frozen=True)
class PolicyTreeILP:
    """Store an exact policy-tree ILP and lookup maps. / 保存 exact policy-tree ILP 及变量映射。"""

    spec: ILPModelSpec
    variable_items: Mapping[str, FrontierItem]
    variable_metrics: Mapping[str, ExpansionMetrics]
    root_variable_ids: tuple[str, ...]
    frontier_variable_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class FrontierSelectionILP:
    """Store a HILP frontier-selection p-ILP and lookup maps. / 保存 HILP frontier-selection p-ILP 及映射。"""

    spec: ILPModelSpec
    variable_items: Mapping[str, FrontierItem]


@dataclass(frozen=True)
class Algorithm1ExpansionRecord:
    """Store one Algorithm 1 call to Algorithm 2. / 保存 Algorithm 1 中一次调用 Algorithm 2 的结果。"""

    var_id: str
    item: FrontierItem
    expanded: ExpandedAction
    continues: bool


def build_full_tree_ilp(
    runtime: PyRDDLGymRuntime,
    interface: ANDORSearchInterface,
    duration_evaluator: HistoryDurationEvaluator,
    *,
    risk_budget: float | None = None,
    root_belief: Mapping[StateKey, float] | None = None,
    deadline: float | None = None,
) -> PolicyTreeILP:
    r"""Encode the AND-OR policy tree as a binary full-ILP model.

    Paper correspondence:

    - Root policy constraint:

      $$\sum_{a \in A(root)} x_{root,a}=1$$


    - Observation-flow constraint for each observation node:
      
      $$\sum_{a \in A(qo)} x_{qo,a}=x_{q}$$


    - Objective over action histories:

      $$\max \sum_q u_q x_q$$
      
    - Chance-constrained risk row:
    
        $$\sum_q r_q x_q \le R,\quad R=\Delta-r(b_0)$$
     
        when a risk budget is provided

    Algorithm 2 Expand enumerates finite grounded transition/observation support
    exactly from pyRDDLGym grounded CPFs through `ExactRDDLKernel`.

    / 将 AND-OR policy tree 编码为二元 full-ILP；Algorithm 2 Expand 通过
    `ExactRDDLKernel` 从 pyRDDLGym grounded CPF 精确枚举有限
    transition/observation 支持。
    """

    records = paper_preprocess(
        runtime=runtime,
        interface=interface,
        duration_evaluator=duration_evaluator,
        root_belief=root_belief,
        deadline=deadline,
    )
    return _encode_algorithm1_records_as_full_ilp(
        records,
        risk_budget=_effective_safe_risk_budget(runtime, interface, risk_budget, root_belief),
        model_name="darp_full_tree",
    )


def build_partial_tree_ilp(
    *,
    runtime: PyRDDLGymRuntime,
    interface: ANDORSearchInterface,
    expanded_records: Sequence[Algorithm1ExpansionRecord],
    frontier_records: Sequence[Algorithm1ExpansionRecord],
    risk_budget: float | None = None,
    root_belief: Mapping[StateKey, float] | None = None,
) -> PolicyTreeILP:
    r"""Encode the current HILP partial policy tree.

    Algorithm 3 solves a p-ILP over the partial tree $$E \cup F$$ rather than
    over every horizon-feasible history.  Records in $$E$$ keep their
    Definition 3.1 observation-flow rows; records in $$F$$ are frontier leaves
    and therefore have no child-flow rows yet.

    / 编码 HILP 当前的 partial policy tree：已展开集合 $$E$$ 保留 flow 约束，
    frontier 集合 $$F$$ 作为截断叶子参与目标与风险行，不触发完整树枚举。
    """

    records = tuple(expanded_records) + tuple(frontier_records)
    return _encode_algorithm1_records_as_full_ilp(
        records,
        risk_budget=_effective_safe_risk_budget(runtime, interface, risk_budget, root_belief),
        model_name="darp_hilp_partial_tree",
        frontier_variable_ids=tuple(record.var_id for record in frontier_records),
    )


def paper_preprocess(
    *,
    runtime: PyRDDLGymRuntime,
    interface: ANDORSearchInterface,
    duration_evaluator: HistoryDurationEvaluator,
    root_belief: Mapping[StateKey, float] | None,
    deadline: float | None,
) -> tuple[Algorithm1ExpansionRecord, ...]:
    r"""Run paper Algorithm 1 `Preprocess` and return expanded action records.

    Original Algorithm 1 alternates between observation histories
    $$q\in N$$ and actions $$a\in A$$, calling Algorithm 2 for each
    $$qa$$. DARP keeps the queue at the action-history level because
    `expand_frontier_item` returns each exact observation branch and its next action
    frontier together.

    The continuation test is the paper line-8 condition:

    $$
       \text{if } \exists o\in O \text{ such that } \tau(qao)>\varsigma
       \text{ then add } qao \text{ to } N.
    $$

    / 运行论文 Algorithm 1：不断调用 `expand_frontier_item`，当
    $$\tau(qao)>\varsigma$$ 时继续加入下一层；固定 horizon 已经包含在
    `duration_evaluator.horizon` 的 $$\tau(qao)$$ 计算中。
    """

    root_frontier = initialize_root_frontier(runtime, interface, root_belief=root_belief)
    queue: list[FrontierItem] = list(root_frontier.frontier)
    records: list[Algorithm1ExpansionRecord] = []
    seen: set[str] = set()

    while queue:
        _raise_if_deadline_expired(deadline)
        item = queue.pop(0)
        var_id = _action_var_id(item)
        if var_id in seen:
            continue
        seen.add(var_id)
        expanded = expand_frontier_item(item, interface, duration_evaluator)
        # Algorithm 1 lines 7-9: Algorithm 2 Expand creates child frontier entries
        # only for $$qao$$ branches satisfying $$tau(qao) > varsigma$$.
        # 论文第 7-9 行：Algorithm 2 Expand 只为 $$tau(qao)>varsigma$$ 的 $$qao$$ 分支
        # 创建后继 frontier；不再额外引入 lookahead/depth 截断。
        continues = bool(expanded.child_frontier)
        records.append(
            Algorithm1ExpansionRecord(
                var_id=var_id,
                item=item,
                expanded=expanded,
                continues=continues,
            )
        )
        if continues:
            queue.extend(expanded.child_frontier)
    return tuple(records)


def _effective_safe_risk_budget(
    runtime: PyRDDLGymRuntime,
    interface: ANDORSearchInterface,
    risk_budget: float | None,
    root_belief: Mapping[StateKey, float] | None,
) -> float | None:
    r"""Return Lemma 3.3's residual risk budget.

    The paper rewrites the chance constraint as:

    $$
       \sum_q r_q x_q \le R,\qquad R=\Delta-r(b_0).
    $$

    / 返回 chance constraint 的剩余预算；初始 belief 的 unsafe 概率
    $$r(b_0)$$ 先从用户给定的 $$\Delta$$ 中扣除。
    """
    if risk_budget is None or interface.exact_kernel is None:
        return risk_budget
    if root_belief is None:
        root_belief = interface.exact_kernel.initial_belief_from_state(runtime.state)
    root_risk = getattr(interface.exact_kernel, "belief_state_risk_probability", None)
    initial_risk = root_risk(root_belief) if root_risk is not None else 0.0
    return float(risk_budget) - initial_risk


def _encode_algorithm1_records_as_full_ilp(
    records: Sequence[Algorithm1ExpansionRecord],
    *,
    risk_budget: float | None,
    model_name: str = "darp_full_tree",
    frontier_variable_ids: tuple[str, ...] = (),
) -> PolicyTreeILP:
    r"""Encode Algorithm 1/2 records as the paper full-ILP.

    For each action history $$q\in\tilde A$$, Algorithm 2 supplies
    constants $$u_q$$ and $$r_q$$. The encoder creates one binary
    variable $$x_q$$ and writes Definition 3.1:

    $$
       \sum_{a\in A}x_a=1,\qquad
       \sum_{a\in A}x_{qoa}=x_q.
    $$

    The objective and optional Lemma 3.3 safe-belief risk row are:

    $$
       \max \sum_q u_qx_q,\qquad
       \sum_q r_qx_q\le R.
    $$

    where $$R=\Delta-r(b_0)$$, $$r_q=\rho^*(q)r(b_q)$$.

    / 将 Algorithm 1/2 得到的 action histories 编码成论文 full-ILP；
    风险行使用 Lemma 3.3 的 safe-belief 线性化形式。
    """

    variables: dict[str, ILPVariable] = {}
    objective: dict[str, float] = {}
    constraints: list[ILPLinearConstraint] = []
    variable_items: dict[str, FrontierItem] = {}
    variable_metrics: dict[str, ExpansionMetrics] = {}
    root_ids: list[str] = []
    declared_var_ids = {record.var_id for record in records}

    for record in records:
        item = record.item
        expanded = record.expanded

        # Definition 3.1 variable: $$x_q=1$$ means this action-history is selected
        # in the deterministic policy tree. / Definition 3.1 变量：$$x_q=1$$
        # 表示 deterministic policy tree 选择该 action history。
        variables[record.var_id] = ILPVariable(
            var_id=record.var_id,
            label=item.node.history.label(),
            metadata={
                "action": item.action_label,
                "root_action": item.root_label,
                "node_id": item.node.node_id,
                "history": item.node.history.label(),
            },
        )
        variable_items[record.var_id] = item
        variable_metrics[record.var_id] = expanded.metrics
        objective[record.var_id] = expanded.metrics.utility
        if item.node.history.depth == 1:
            root_ids.append(record.var_id)
        constraints.extend(
            _definition31_flow_constraints(
                record.var_id,
                expanded,
                declared_var_ids=declared_var_ids,
                should_encode=record.continues,
            )
        )

    if not root_ids:
        raise ValueError("Policy tree has no root action variables.")

    # Definition 3.1 root row: $$\sum_{a \in A(root)} x_a = 1$$.
    # Definition 3.1 根约束：根节点必须且只能选择一个 action。
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
        # Lemma 3.3 chance constraint: $$\sum_q r_q x_q \le R,\ R=\Delta-r(b_0)$$.
        # Lemma 3.3 风险约束：由 safe-belief 递推得到的 $$r_q$$ 常量构成。
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
        name=model_name,
        variables=tuple(variables.values()),
        objective=objective,
        constraints=tuple(constraints),
    )
    return PolicyTreeILP(
        spec=spec,
        variable_items=variable_items,
        variable_metrics=variable_metrics,
        root_variable_ids=tuple(root_ids),
        frontier_variable_ids=frontier_variable_ids,
    )


def _definition31_flow_constraints(
    parent_var_id: str,
    expanded: ExpandedAction,
    *,
    declared_var_ids: set[str],
    should_encode: bool,
) -> tuple[ILPLinearConstraint, ...]:
    r"""Encode Definition 3.1 observation-flow constraint.

    For every expanded action history $$q$$ and observation
    branch $$o$$, the selected policy must choose exactly one child action
    whenever $$x_q=1$$:

    $$
       \sum_{a\in A}x_{qoa}=x_q.
    $$

    DARP writes this row only for non-leaf action histories. A history is a leaf
    when Algorithm 1 stops because $$\tau(qao)\le\varsigma$$. This mirrors
    the reference code's ``if ins.duration_model(q) < ins.horizon`` guard
    before adding child-flow rows; in DARP, that horizon is already inside
    ``duration_evaluator``.

    / 只对非叶子 action history 编码 observation-flow；duration 停止的叶子
    不应引用未声明的子变量。
    """
    if not should_encode:
        return ()
    constraints: list[ILPLinearConstraint] = []
    for index, observation_frontier in enumerate(expanded.observation_frontiers):
        child_frontier = observation_frontier.child_frontier
        if not child_frontier:
            continue
        coefficients = {_action_var_id(child): 1.0 for child in child_frontier}
        missing = set(coefficients) - declared_var_ids
        if missing:
            raise ValueError(
                "Cannot encode flow constraint with undeclared child variables: "
                + ", ".join(sorted(missing))
            )
        coefficients[parent_var_id] = coefficients.get(parent_var_id, 0.0) - 1.0
        constraints.append(
            ILPLinearConstraint(
                name=f"flow_{parent_var_id}_obs_{index}",
                coefficients=coefficients,
                sense="==",
                rhs=0.0,
            )
        )
    return tuple(constraints)


def build_frontier_selection_ilp(
    scored_frontier: Sequence[tuple[float, FrontierItem]],
    *,
    frontier_width: int,
) -> FrontierSelectionILP:
    r"""Encode HILP frontier selection as a small p-ILP.

    This is the Phase 8 p-ILP hook for Algorithm 3: select up to
    ``frontier_width`` frontier histories with the highest current score.

    $$
       \max \sum_{q \in F} score(q) y_q
       \quad
       1 \le \sum_{q \in F} y_q \le k
    $$

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
