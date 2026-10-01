"""Policy-tree ILP encoders for full-tree and HILP partial trees.

/ full-tree 与 HILP partial-tree 共用的策略树 ILP 编码器。
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass, field
from math import isfinite

from darp.adapter.kernel import StateKey
from darp.adapter.runtime import PyRDDLGymRuntime
from darp.ilp.model import ILPLinearConstraint, ILPModelDelta, ILPModelSpec, ILPVariable
from darp.model.and_or_tree import ANDORSearchInterface
from darp.model.duration import HistoryDurationEvaluator
from darp.planning.expand import (
    ExpandedAction,
    ExpansionMetrics,
    apply_terminal_heuristic,
    expand_frontier_item,
)
from darp.planning.heuristic import UtilityHeuristic
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
    # Keep the concrete Algorithm-2 expansion behind every materialized variable.
    # A lazy frontier is intentionally absent until selected; policy extraction
    # then treats an early-stopped incumbent as incomplete. The p-ILP objective
    # may replace utility with a heuristic, so materialized entries store
    # ``policy_expansion`` rather than the modified scoring record.
    # 保存所有已 materialize 变量的 Algorithm-2 展开；lazy frontier 在被
    # 选中前故意缺席，使提前停止的 policy 被判为 incomplete。
    variable_expansions: Mapping[str, ExpandedAction] = field(default_factory=dict)
    variable_continues: Mapping[str, bool] = field(default_factory=dict)
    risk_budget: float | None = None  # Delta: total risk budget. / Δ：总风险预算。
    initial_risk: float = 0.0  # r(b_0): risk before any action. / r(b₀)：执行动作前的初态风险。


@dataclass(frozen=True)
class HistoryRecord:
    """Keep a history's ILP coefficients separate from its concrete expansion.

    ``ilp_metrics`` holds (u_q, r_q) in E and (h_q^u, h_q^r) in F.
    ``policy_expansion`` holds the actual branches and achieved utility; it is
    None for an unmaterialized lazy leaf, not an empty terminal expansion.

    / 动作历史 q 的 ILP 系数：E 中为 (u_q, r_q)，F 中为 (h_q^u, h_q^r)。
    policy_expansion 单独保存实际分支和实现效用；None 表示尚未生成，
    不能把它当作没有后继的终止叶。当前 h_q^r 使用一步首次失败概率下界。
    """

    var_id: str
    item: FrontierItem
    ilp_metrics: ExpansionMetrics
    # Flow rows are enabled after F -> E. / F 移入 E 后才加入后续观测 flow 行。
    continues: bool
    policy_expansion: ExpandedAction | None = None


@dataclass(frozen=True)
class _RiskEncodingContext:
    """Risk constants shared by full- and partial-tree encoders.

    / 保存两类编码器共享的风险常数 Delta、R 和 r(b_0)。
    """

    risk_budget: float | None  # Delta / 总预算 Δ。
    remaining_risk_budget: float | None  # R = Delta - r(b_0) / 扣除初态风险后的预算。
    initial_risk: float  # r(b_0) / 初态风险。


def build_full_tree_ilp(
    runtime: PyRDDLGymRuntime,
    interface: ANDORSearchInterface,
    duration_evaluator: HistoryDurationEvaluator,
    *,
    risk_budget: float | None = None,
    root_belief: Mapping[StateKey, float] | None = None,
    max_nodes: int | None = 100_000,
    terminal_heuristic: UtilityHeuristic | None = None,
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
        terminal_heuristic=terminal_heuristic,
    )
    risk_context = _risk_encoding_context(
        runtime,
        interface,
        risk_budget,
        root_belief,
    )
    return _encode_policy_tree_records(
        records,
        remaining_risk_budget=risk_context.remaining_risk_budget,
        risk_budget=risk_context.risk_budget,
        initial_risk=risk_context.initial_risk,
        model_name="darp_full_tree",
    )


def build_partial_tree_ilp(
    *,
    runtime: PyRDDLGymRuntime,
    interface: ANDORSearchInterface,
    expanded_records: Sequence[HistoryRecord],
    frontier_records: Sequence[HistoryRecord],
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
    risk_context = _risk_encoding_context(
        runtime,
        interface,
        risk_budget,
        root_belief,
    )
    return _encode_policy_tree_records(
        records,
        remaining_risk_budget=risk_context.remaining_risk_budget,
        risk_budget=risk_context.risk_budget,
        initial_risk=risk_context.initial_risk,
        model_name="darp_hilp_partial_tree",
        frontier_variable_ids=tuple(record.var_id for record in frontier_records),
    )


class IncrementalPartialTreeILP:
    """Encode only new frontier records and F→E refinements.

    Existing root/flow rows are retained. Snapshots are independent of later
    updates so timeout fallback still uses the policy matching the last solve.

    / 只编码新 frontier 和本轮 F→E 的节点；旧 root/flow 行保持不变。
    快照独立于后续更新，保证超时时仍能返回上一轮解对应的策略。
    """

    def __init__(
        self,
        runtime: PyRDDLGymRuntime,
        interface: ANDORSearchInterface,
        *,
        risk_budget: float | None = None,
        root_belief: Mapping[StateKey, float] | None = None,
    ) -> None:
        self._risk_context = _risk_encoding_context(
            runtime, interface, risk_budget, root_belief,
        )
        self._variables: dict[str, ILPVariable] = {}
        self._objective: dict[str, float] = {}
        self._rows: dict[str, ILPLinearConstraint] = {}
        self._items: dict[str, FrontierItem] = {}
        self._expansions: dict[str, ExpandedAction] = {}
        self._continues: dict[str, bool] = {}
        self._frontier: dict[str, None] = {}
        self._roots: list[str] = []
        self._risk: dict[str, float] = {}
        self._risk_initialized = False
        self._new_variables: list[ILPVariable] = []
        self._new_rows: list[ILPLinearConstraint] = []
        self._objective_updates: dict[str, float] = {}
        self._risk_updates: dict[str, float] = {}

    def update(
        self,
        *,
        expanded_records: Sequence[HistoryRecord],
        frontier_records: Sequence[HistoryRecord],
    ) -> None:
        """Apply changed E records and new F records, not the whole E∪F.

        Declare all children in this batch before adding parent flow rows.
        Replace risk coefficients with their new values rather than adding to
        old values, preserving first-entry risk semantics.

        / 只处理变化的 E 记录与新 F 记录，不遍历整个 E∪F。
        先声明本批次全部 child，再添加父节点 flow；risk 系数按新值替换，
        不累加旧值，也不改变首次进入危险状态的概率语义。
        """
        records = tuple(expanded_records) + tuple(frontier_records)
        for record in records:
            var_id = record.var_id
            if var_id not in self._variables:
                variable = ILPVariable(var_id)
                self._variables[var_id] = variable
                self._new_variables.append(variable)
                if record.item.node.history.depth == 1:
                    if "root_action" in self._rows:
                        raise ValueError("All root actions must be declared in the first update.")
                    self._roots.append(var_id)
            self._items[var_id] = record.item
            self._continues[var_id] = bool(record.continues)
            if record.policy_expansion is not None:
                self._expansions[var_id] = record.policy_expansion
            else:
                self._expansions.pop(var_id, None)
            utility = record.ilp_metrics.utility
            if self._objective.get(var_id) != utility:
                self._objective[var_id] = utility
                self._objective_updates[var_id] = utility
            if self._risk_context.remaining_risk_budget is not None:
                risk = record.ilp_metrics.chance_risk
                if self._risk.get(var_id, 0.0) != risk:
                    if risk == 0.0:
                        self._risk.pop(var_id, None)
                    else:
                        self._risk[var_id] = risk
                    self._risk_updates[var_id] = risk
        for record in expanded_records:
            self._frontier.pop(record.var_id, None)
        for record in frontier_records:
            self._frontier[record.var_id] = None

        if "root_action" not in self._rows:
            if not self._roots:
                raise ValueError("Policy tree has no root action variables.")
            root = ILPLinearConstraint(
                "root_action", dict.fromkeys(self._roots, 1.0), "==", 1.0,
            )
            self._rows[root.name] = root
            self._new_rows.append(root)
        for record in records:
            for row in _definition31_flow_constraints(
                record.var_id,
                record.policy_expansion,
                declared_var_ids=self._variables.keys(),
                should_encode=record.continues,
            ):
                previous = self._rows.get(row.name)
                if previous is None:
                    self._rows[row.name] = row
                    self._new_rows.append(row)
                elif previous != row:
                    raise ValueError(f"Refinement changed an existing flow row: {row.name}")

    def snapshot(self) -> tuple[PolicyTreeILP, ILPModelDelta]:
        """Return a stable policy checkpoint and consume the pending delta.

        Shallow-copy the checkpoint without re-encoding old nodes/flow rows or
        comparing the complete objective and risk row.

        / 返回独立的策略快照及待传差量，并清空已取出的差量记录。
        仅浅拷贝策略检查点；不重编码旧节点/flow，也不比较完整目标与 risk 行。
        """
        rows = tuple(self._rows.values())
        new_rows = tuple(self._new_rows)
        coefficients: dict[str, Mapping[str, float]] = {}
        if self._risk_context.remaining_risk_budget is not None:
            risk_row = ILPLinearConstraint(
                "risk_budget", dict(self._risk), "<=", self._risk_context.remaining_risk_budget,
            )
            rows += (risk_row,)
            if not self._risk_initialized:
                new_rows += (risk_row,)
                self._risk_initialized = True
            elif self._risk_updates:
                coefficients["risk_budget"] = dict(self._risk_updates)
        tree = PolicyTreeILP(
            spec=ILPModelSpec(
                name="darp_hilp_partial_tree",
                variables=tuple(self._variables.values()),
                objective=dict(self._objective),
                constraints=rows,
            ),
            variable_items=dict(self._items),
            root_variable_ids=tuple(self._roots),
            frontier_variable_ids=tuple(self._frontier),
            variable_expansions=dict(self._expansions),
            variable_continues=dict(self._continues),
            risk_budget=self._risk_context.risk_budget,
            initial_risk=self._risk_context.initial_risk,
        )
        model_delta = ILPModelDelta(
            variables=tuple(self._new_variables),
            objective=dict(self._objective_updates),
            constraints=new_rows,
            coefficients=coefficients,
        )
        self._new_variables.clear()
        self._new_rows.clear()
        self._objective_updates.clear()
        self._risk_updates.clear()
        return tree, model_delta


def paper_preprocess(
    *,
    runtime: PyRDDLGymRuntime,
    interface: ANDORSearchInterface,
    duration_evaluator: HistoryDurationEvaluator,
    root_belief: Mapping[StateKey, float] | None,
    max_nodes: int | None = 100_000,
    terminal_heuristic: UtilityHeuristic | None = None,
) -> tuple[HistoryRecord, ...]:
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
    records: list[HistoryRecord] = []
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
        if terminal_heuristic is not None:
            expanded = apply_terminal_heuristic(
                item,
                expanded,
                interface,
                terminal_heuristic,
            )
        # Algorithm 1 lines 7-9: Algorithm 2 Expand creates child frontier entries
        # only for $$qao$$ branches satisfying $$tau(qao) > varsigma$$.
        # 论文第 7-9 行：Algorithm 2 Expand 只为 $$tau(qao)>varsigma$$ 的 $$qao$$ 分支
        # 创建后继 frontier。
        continues = bool(expanded.child_frontier)
        records.append(
            HistoryRecord(
                var_id=var_id,
                item=item,
                ilp_metrics=expanded.metrics,
                continues=continues,
                policy_expansion=expanded,
            )
        )
        if continues:
            queue.extend(expanded.child_frontier)
    return tuple(records)


def validate_risk_budget(risk_budget: float | None) -> None:
    r"""Validate the CC-POMDP chance budget :math:`\Delta`. / 校验概率预算 Delta 的范围。"""
    if risk_budget is None:
        return
    numeric = float(risk_budget)
    if not isfinite(numeric) or not 0.0 <= numeric <= 1.0:
        raise ValueError(
            "risk_budget must be a finite probability in [0, 1] for a chance constraint."
        )


def _risk_encoding_context(
    runtime: PyRDDLGymRuntime,
    interface: ANDORSearchInterface,
    risk_budget: float | None,
    root_belief: Mapping[StateKey, float] | None,
) -> _RiskEncodingContext:
    r"""Resolve the shared Lemma 3.3 encoding constants once.

    Lemma 3.3 rewrites the chance constraint as:

    $$
       \sum_q r_q x_q \le R,\qquad R=\Delta-r(b_0).
    $$

    / Lemma 3.3 先从 $$\Delta$$ 扣除初始 belief 的风险概率 $$r(b_0)$$。
    """
    validate_risk_budget(risk_budget)
    initial_risk = 0.0
    if interface.kernel is not None:
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

    remaining_risk_budget = None if risk_budget is None else float(risk_budget)
    if remaining_risk_budget is not None:
        remaining_risk_budget -= initial_risk

    return _RiskEncodingContext(
        risk_budget=risk_budget,
        remaining_risk_budget=remaining_risk_budget,
        initial_risk=initial_risk,
    )


def _encode_policy_tree_records(
    records: Sequence[HistoryRecord],
    *,
    remaining_risk_budget: float | None,
    risk_budget: float | None,
    initial_risk: float = 0.0,
    model_name: str = "darp_full_tree",
    frontier_variable_ids: tuple[str, ...] = (),
) -> PolicyTreeILP:
    r"""Encode history records as full-ILP or p-ILP.

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

    For p-ILP, frontier coefficients are (h_q^u, h_q^r) instead of (u_q, r_q).
    ``remaining_risk_budget`` is R, not the original budget Delta.

    / 共用 full-ILP/p-ILP 编码器；frontier 系数使用 (h_q^u, h_q^r)。
    remaining_risk_budget 是已扣除初态风险的 R，不是原始预算 Delta。
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

        # Definition 3.1 variable: $$x_q=1$$ means this action-history is selected
        # in the deterministic policy tree. / Definition 3.1 变量：$$x_q=1$$
        # 表示 deterministic policy tree 选择该 action history。
        variables[record.var_id] = ILPVariable(var_id=record.var_id)
        variable_items[record.var_id] = item
        variable_metrics[record.var_id] = record.ilp_metrics
        # Lazy F records have coefficients but no executable expansion yet.
        # / lazy F 只有估值，不能当作已完成的策略叶节点。
        if record.policy_expansion is not None:
            variable_expansions[record.var_id] = record.policy_expansion
        variable_continues[record.var_id] = bool(record.continues)
        objective[record.var_id] = record.ilp_metrics.utility
        if item.node.history.depth == 1:
            root_ids.append(record.var_id)
        constraints.extend(
            _definition31_flow_constraints(
                record.var_id,
                record.policy_expansion,
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
    if remaining_risk_budget is not None:
        # Lemma 3.3 uses safe-flow first-entry coefficients and
        # R=Delta-r(b0). / 使用安全概率流的首次进入风险系数。
        constraints.append(
            ILPLinearConstraint(
                name="risk_budget",
                coefficients={
                    var_id: metrics.chance_risk
                    for var_id, metrics in variable_metrics.items()
                    if metrics.chance_risk != 0.0
                },
                sense="<=",
                rhs=float(remaining_risk_budget),
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
        variable_expansions=variable_expansions,
        variable_continues=variable_continues,
        risk_budget=risk_budget,
        initial_risk=float(initial_risk),
    )


def _definition31_flow_constraints(
    parent_var_id: str,
    expanded: ExpandedAction | None,
    *,
    declared_var_ids: AbstractSet[str],
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
    if expanded is None:
        if should_encode:
            raise ValueError("Cannot encode child flow before expanding the history.")
        return ()
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
