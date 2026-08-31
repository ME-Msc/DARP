"""Policy-tree ILP encoders for full-tree and HILP partial trees.

/ full-tree 与 HILP partial-tree 共用的策略树 ILP 编码器。
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from math import isfinite

from darp.adapter.kernel import (
    RiskConstraintType,
    StateKey,
    risk_constraint_type_for_kernel,
)
from darp.adapter.runtime import PyRDDLGymRuntime
from darp.ilp.model import ILPLinearConstraint, ILPModelSpec, ILPVariable
from darp.model.and_or_tree import ANDORSearchInterface
from darp.model.duration import HistoryDurationEvaluator
from darp.planning.expand import ExpandedAction, ExpansionMetrics, expand_frontier_item
from darp.planning.preprocess import (
    FrontierItem,
    initialize_root_frontier,
    resolve_root_belief,
)


@dataclass(frozen=True)
class PolicyTreeILP:
    """Store a policy-tree ILP and lookup maps. / 保存 policy-tree ILP 及变量映射。"""

    spec: ILPModelSpec
    variable_items: Mapping[str, FrontierItem]
    root_variable_ids: tuple[str, ...]
    frontier_variable_ids: tuple[str, ...] = ()
    constraint_type: RiskConstraintType = "chance"
    # Keep the concrete Algorithm-2 expansion behind every materialized variable.
    # A lazy frontier is intentionally absent until selected; policy extraction
    # then treats an early-stopped incumbent as incomplete. The p-ILP objective
    # may replace utility with a heuristic, so materialized entries store
    # ``policy_expansion`` rather than the modified scoring record.
    # 保存所有已 materialize 变量的 Algorithm-2 展开；lazy frontier 在被
    # 选中前故意缺席，使提前停止的 policy 被判为 incomplete。
    variable_expansions: Mapping[str, ExpandedAction] = field(default_factory=dict)
    variable_continues: Mapping[str, bool] = field(default_factory=dict)
    constraint_budget: float | None = None
    initial_chance_risk: float = 0.0


@dataclass(frozen=True)
class Algorithm1ExpansionRecord:
    """Store one Algorithm-1 action history and its current expansion state."""

    var_id: str
    item: FrontierItem
    expanded: ExpandedAction
    continues: bool
    # HILP can score a frontier with a modified utility coefficient. Retain
    # the unmodified expansion for executable-policy validation.
    # frontier 可以使用修改后的 heuristic utility 评分，但必须另外保留未经
    # 修改的展开，用于可执行策略验证。
    # ``None`` marks a lazy HILP leaf whose observation branches are postponed
    # until the incumbent selects it.
    policy_expansion: ExpandedAction | None = None


@dataclass(frozen=True)
class _ConstraintEncodingContext:
    """Values shared by full- and partial-tree constraint encoders.

    / 保存两类编码器共享的约束类型、原始预算和有效 RHS。
    """

    constraint_type: RiskConstraintType
    original_budget: float | None
    effective_rhs: float | None
    initial_chance_risk: float


def build_full_tree_ilp(
    runtime: PyRDDLGymRuntime,
    interface: ANDORSearchInterface,
    duration_evaluator: HistoryDurationEvaluator,
    *,
    risk_budget: float | None = None,
    root_belief: Mapping[StateKey, float] | None = None,
    max_nodes: int | None = 100_000,
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
    from pyRDDLGym grounded CPFs through the RDDL kernel.

    / 将 AND-OR policy tree 编码为二元 full-ILP；Algorithm 2 Expand 通过
    RDDL kernel 从 pyRDDLGym grounded CPF 枚举有限
    transition/observation 支持。
    """

    records = paper_preprocess(
        runtime=runtime,
        interface=interface,
        duration_evaluator=duration_evaluator,
        root_belief=root_belief,
        max_nodes=max_nodes,
    )
    constraint = _constraint_encoding_context(
        runtime,
        interface,
        risk_budget,
        root_belief,
    )
    return _encode_algorithm1_records_as_full_ilp(
        records,
        risk_budget=constraint.effective_rhs,
        constraint_type=constraint.constraint_type,
        original_constraint_budget=constraint.original_budget,
        initial_chance_risk=constraint.initial_chance_risk,
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
    constraint = _constraint_encoding_context(
        runtime,
        interface,
        risk_budget,
        root_belief,
    )
    return _encode_algorithm1_records_as_full_ilp(
        records,
        risk_budget=constraint.effective_rhs,
        constraint_type=constraint.constraint_type,
        original_constraint_budget=constraint.original_budget,
        initial_chance_risk=constraint.initial_chance_risk,
        model_name="darp_hilp_partial_tree",
        frontier_variable_ids=tuple(record.var_id for record in frontier_records),
    )


def paper_preprocess(
    *,
    runtime: PyRDDLGymRuntime,
    interface: ANDORSearchInterface,
    duration_evaluator: HistoryDurationEvaluator,
    root_belief: Mapping[StateKey, float] | None,
    max_nodes: int | None = 100_000,
) -> tuple[Algorithm1ExpansionRecord, ...]:
    r"""Run paper Algorithm 1 `Preprocess` and return expanded action records.

    Original Algorithm 1 alternates between observation histories
    $$q\in N$$ and actions $$a\in A$$, calling Algorithm 2 for each
    $$qa$$. DARP keeps the queue at the action-history level because
    `expand_frontier_item` returns each observation branch and its next action
    frontier together.

    The continuation test is the paper line-8 condition:

    $$
       \text{if } \exists o\in O \text{ such that } \tau(qao)>\varsigma
       \text{ then add } qao \text{ to } N.
    $$

    / 运行论文 Algorithm 1：不断调用 `expand_frontier_item`，当
    $$\tau(qao)>\varsigma$$ 时继续加入下一层。`duration_evaluator`
    实现论文的 durative stopping condition。
    """

    if max_nodes is not None and max_nodes < 1:
        raise ValueError("max_nodes must be positive when provided")
    root_frontier = initialize_root_frontier(runtime, interface, root_belief=root_belief)
    queue = deque(root_frontier)
    records: list[Algorithm1ExpansionRecord] = []
    seen: set[str] = set()

    while queue:
        if max_nodes is not None and len(records) >= max_nodes:
            raise RuntimeError(
                "Full policy-tree preprocessing reached max_nodes="
                f"{max_nodes}; use HILP for normal experiments or explicitly raise the oracle cap."
            )
        item = queue.popleft()
        var_id = _action_var_id(item)
        if var_id in seen:
            continue
        seen.add(var_id)
        expanded = expand_frontier_item(item, interface, duration_evaluator)
        # Algorithm 1 lines 7-9: Algorithm 2 Expand creates child frontier entries
        # only for $$qao$$ branches satisfying $$tau(qao) > varsigma$$.
        # 论文第 7-9 行：Algorithm 2 Expand 只为 $$tau(qao)>varsigma$$ 的 $$qao$$ 分支
        # 创建后继 frontier。
        continues = bool(expanded.child_frontier)
        records.append(
            Algorithm1ExpansionRecord(
                var_id=var_id,
                item=item,
                expanded=expanded,
                continues=continues,
                policy_expansion=expanded,
            )
        )
        if continues:
            queue.extend(expanded.child_frontier)
    return tuple(records)


def _constraint_type(interface: ANDORSearchInterface) -> RiskConstraintType:
    """Return the paper constraint selected by the kernel. / 返回 kernel 选定的论文约束。"""
    if interface.kernel is None:
        return "chance"
    return risk_constraint_type_for_kernel(interface.kernel)


def validate_constraint_budget(
    interface: ANDORSearchInterface,
    risk_budget: float | None,
) -> None:
    r"""Validate :math:`\Delta` for CC or :math:`C` for expected-cost C-POMDP.

    A chance budget is a probability in ``[0, 1]``.  Lemma 3.2 permits a
    general finite real expected-cost bound, so it must not inherit that
    probability restriction.

    / chance 预算是概率；expected-cost 预算是任意有限实数。
    """
    if risk_budget is None:
        return
    numeric = float(risk_budget)
    constraint_type = _constraint_type(interface)
    if not isfinite(numeric):
        raise ValueError("risk_budget must be finite.")
    if constraint_type == "chance" and not 0.0 <= numeric <= 1.0:
        raise ValueError(
            "risk_budget must be a finite probability in [0, 1] for a chance constraint."
        )


def _constraint_encoding_context(
    runtime: PyRDDLGymRuntime,
    interface: ANDORSearchInterface,
    risk_budget: float | None,
    root_belief: Mapping[StateKey, float] | None,
) -> _ConstraintEncodingContext:
    r"""Resolve the shared Lemma 3.2/3.3 encoding constants once.

    Lemma 3.2 keeps the cumulative expected-cost bound unchanged, ``R=C``.
    Lemma 3.3 rewrites the chance constraint as:

    $$
       \sum_q r_q x_q \le R,\qquad R=\Delta-r(b_0).
    $$

    / Lemma 3.2 直接使用 ``C``；Lemma 3.3 先从 $$\Delta$$
    扣除初始 belief 的 unsafe 概率 $$r(b_0)$$。
    """
    constraint_type = _constraint_type(interface)
    validate_constraint_budget(interface, risk_budget)
    initial_risk = 0.0
    if constraint_type == "chance" and interface.kernel is not None:
        resolved_belief = resolve_root_belief(runtime, interface, root_belief)
        if resolved_belief is not None:
            belief_risk = getattr(
                interface.kernel,
                "belief_state_risk",
                None,
            )
            if belief_risk is None:
                raise TypeError(
                    "Chance constraints require belief_state_risk()."
                )
            initial_risk = float(belief_risk(resolved_belief))

    effective_rhs = None if risk_budget is None else float(risk_budget)
    if effective_rhs is not None and constraint_type == "chance":
        effective_rhs -= initial_risk

    return _ConstraintEncodingContext(
        constraint_type=constraint_type,
        original_budget=risk_budget,
        effective_rhs=effective_rhs,
        initial_chance_risk=initial_risk,
    )


def _encode_algorithm1_records_as_full_ilp(
    records: Sequence[Algorithm1ExpansionRecord],
    *,
    risk_budget: float | None,
    constraint_type: RiskConstraintType,
    original_constraint_budget: float | None,
    initial_chance_risk: float = 0.0,
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

    where $$R=\Delta-r(b_0)$$ and $$r_q$$ uses the safe-conditioned
    occurrence probability and belief.

    / 将 Algorithm 1/2 得到的 action histories 编码成论文 full-ILP；
    风险行使用 Lemma 3.3 的 safe-belief 线性化形式。
    """

    variables: dict[str, ILPVariable] = {}
    objective: dict[str, float] = {}
    constraints: list[ILPLinearConstraint] = []
    variable_items: dict[str, FrontierItem] = {}
    variable_metrics: dict[str, ExpansionMetrics] = {}
    variable_expansions: dict[str, ExpandedAction] = {}
    variable_continues: dict[str, bool] = {}
    root_ids: list[str] = []
    declared_var_ids = {record.var_id for record in records}

    for record in records:
        item = record.item
        expanded = record.expanded

        # Definition 3.1 variable: $$x_q=1$$ means this action-history is selected
        # in the deterministic policy tree. / Definition 3.1 变量：$$x_q=1$$
        # 表示 deterministic policy tree 选择该 action history。
        variables[record.var_id] = ILPVariable(var_id=record.var_id)
        variable_items[record.var_id] = item
        variable_metrics[record.var_id] = expanded.metrics
        # A time/round limit may leave an incumbent on a lazy frontier. Omit
        # its placeholder from the executable-policy map so policy validation
        # reports ``missing-expansion`` instead of a false terminal leaf.
        # 若搜索在 lazy frontier 上提前停止，不把空壳当作终止叶；policy
        # validation 会将其明确标为尚未完整展开。
        if record.policy_expansion is not None:
            variable_expansions[record.var_id] = record.policy_expansion
        variable_continues[record.var_id] = bool(record.continues)
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
        # Lemma 3.2 uses ordinary-flow penalty coefficients and R=C;
        # Lemma 3.3 uses safe-flow first-entry coefficients and
        # R=Delta-r(b0). / 两种约束必须选用各自的概率流。
        row_name = "risk_budget" if constraint_type == "chance" else "expected_cost_budget"
        constraints.append(
            ILPLinearConstraint(
                name=row_name,
                coefficients={
                    var_id: metrics.constraint_value
                    for var_id, metrics in variable_metrics.items()
                    if metrics.constraint_value != 0.0
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
        root_variable_ids=tuple(root_ids),
        frontier_variable_ids=frontier_variable_ids,
        constraint_type=constraint_type,
        variable_expansions=variable_expansions,
        variable_continues=variable_continues,
        constraint_budget=original_constraint_budget,
        initial_chance_risk=float(initial_chance_risk),
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
    constraints: list[ILPLinearConstraint] = []
    has_nonterminal_deadend = any(
        observation_frontier.should_expand
        and not observation_frontier.child_frontier
        for observation_frontier in expanded.observation_frontiers
    )
    if has_nonterminal_deadend:
        # One zero row excludes the parent for every dead-end outcome; writing
        # the same x_parent=0 equation once per observation only bloats the ILP.
        # 任一非终止 observation 成为死路时，一条 x_parent=0 就能排除父动作；
        # 无需按 observation 重复写入同一个等式，从而避免无意义地增大 ILP。
        constraints.append(
            ILPLinearConstraint(
                name=f"deadend_{parent_var_id}",
                coefficients={parent_var_id: 1.0},
                sense="==",
                rhs=0.0,
            )
        )
    for index, observation_frontier in enumerate(expanded.observation_frontiers):
        child_frontier = observation_frontier.child_frontier
        if observation_frontier.should_expand and not child_frontier:
            continue
        if not should_encode:
            continue
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


def _action_var_id(item: FrontierItem) -> str:
    """Return a collision-free arena-based policy variable id.

    Full history labels remain variable metadata; solver identifiers use the
    unique integer node arena so punctuation in action/observation labels can
    never collapse two histories to the same sanitized name.

    / 完整 history 保留在 metadata，solver id 使用唯一整数节点编号，避免
    字符清洗造成不同 history 冲突。
    """
    if item.node.node_index < 0:
        raise ValueError("Action history must be interned before ILP encoding.")
    return f"x_n{item.node.node_index}"
