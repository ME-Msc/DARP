"""Define, serialize and extract conditional policies. / 定义、序列化并提取条件策略。"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isclose, isfinite
from typing import TYPE_CHECKING, Any, Literal

from darp.adapter.kernel import ObservationKey

if TYPE_CHECKING:
    from darp.ilp.model import ILPSolveResult
    from darp.planning.ilp_tree import PolicyTreeILP


PolicyInput = Literal["observation", "state"]
ObservationHistory = tuple[ObservationKey, ...]


@dataclass(frozen=True)
class PolicyNode:
    """One action in a finite observation-contingent policy graph. / 有限观测条件策略图中的一个动作。"""

    node_id: str
    stage: int
    action_label: str
    assignment: Mapping[str, Any]
    transitions: Mapping[ObservationKey, str | None]

    def to_dict(self, input_kind: PolicyInput) -> dict[str, Any]:
        return {
            "id": self.node_id,
            "stage": self.stage,
            "action_label": self.action_label,
            "action": json_ready(self.assignment),
            "transitions": [
                {
                    "observation": json_ready(
                        _observation_values(observation, input_kind)
                    ),
                    "next": next_node,
                }
                for observation, next_node in sorted(
                    self.transitions.items(), key=lambda item: repr(item[0])
                )
            ],
        }


@dataclass(frozen=True)
class ConditionalPolicy:
    """A deterministic policy and its essential numeric post-checks. / 确定性策略及必要的数值校验结果。"""

    root: str
    nodes: tuple[PolicyNode, ...]
    input_kind: PolicyInput
    solver_status: str
    duration_complete: bool
    achieved_utility: float | None
    active_constraint_value: float | None
    feasible: bool | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": "darp-policy-graph",
            "version": 1,
            "input": self.input_kind,
            "timing": "act-then-observe",
            "root": self.root,
            "nodes": [node.to_dict(self.input_kind) for node in self.nodes],
            "complete": self.duration_complete,
            "solver_status": self.solver_status,
            "achieved_utility": self.achieved_utility,
            "active_constraint_value": self.active_constraint_value,
            "feasible": self.feasible,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ConditionalPolicy:
        """Restore policy data from :meth:`to_dict` output. / 从序列化结果恢复策略数据。"""
        if (
            value.get("format") != "darp-policy-graph"
            or value.get("version") != 1
        ):
            raise ValueError("Unsupported DARP policy format or version.")
        if value.get("timing") != "act-then-observe":
            raise ValueError("Unsupported DARP policy timing semantics.")
        input_kind = value.get("input")
        if input_kind not in ("observation", "state"):
            raise ValueError("DARP policy input must be 'observation' or 'state'.")
        raw_nodes = value.get("nodes")
        if not isinstance(raw_nodes, Sequence) or isinstance(raw_nodes, str):
            raise TypeError("Policy nodes must be an array.")
        nodes = tuple(_policy_node_from_dict(node, input_kind) for node in raw_nodes)
        return cls(
            root=str(value["root"]),
            nodes=nodes,
            input_kind=input_kind,
            solver_status=str(value["solver_status"]),
            duration_complete=bool(value["complete"]),
            achieved_utility=_optional_float(value.get("achieved_utility")),
            active_constraint_value=_optional_float(
                value.get("active_constraint_value")
            ),
            feasible=(
                None if value.get("feasible") is None else bool(value["feasible"])
            ),
        )


def json_ready(value: Any) -> Any:
    """Convert numpy-backed action values to plain JSON values. / 将 NumPy 动作值转换为普通 JSON 值。"""
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(json_ready(item) for item in value)
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if hasattr(value, "tolist"):
        return json_ready(value.tolist())
    if hasattr(value, "item"):
        return value.item()
    return value


def extract_conditional_policy(
    tree: PolicyTreeILP,
    result: ILPSolveResult,
) -> ConditionalPolicy:
    """Extract and check the policy encoded by selected :math:`x_q`.

    The ILP already enforces policy flow. This small post-check protects the
    public result from a partial HILP frontier and disconnected fake
    incumbents, then sums the selected nodes' utility and constraint values.

    / 验证选中的 x_q 是否形成观测闭合的完整策略，再汇总真实效用与风险；
    不把未展开 frontier 的启发式值当作已经实现的效用。
    """

    selected = _selected_variable_ids(result)
    unresolved: list[str] = []
    selected_roots = selected.intersection(tree.root_variable_ids)
    if len(selected_roots) != 1:
        unresolved.append(
            f"root:expected-one-selected-action:found-{len(selected_roots)}"
        )

    variable_by_node = {
        item.node.node_id: variable_id
        for variable_id, item in tree.variable_items.items()
    }
    actions_by_history: dict[
        ObservationHistory, tuple[str, Mapping[str, Any]]
    ] = {}
    leaf_histories: set[ObservationHistory] = set()
    selected_edges: dict[str, set[str]] = {}
    utility_terms: list[float] = []
    risk_terms: list[float] = []

    for variable_id in sorted(selected):
        item = tree.variable_items.get(variable_id)
        expansion = tree.variable_expansions.get(variable_id)
        if item is None:
            unresolved.append(f"{variable_id}:missing-frontier-item")
            continue

        assignment = item.node.assignment
        if assignment is None:
            unresolved.append(f"{variable_id}:missing-action-assignment")
        else:
            history = item.observation_keys
            action = (item.action_label, dict(assignment))
            previous = actions_by_history.get(history)
            if previous is None:
                actions_by_history[history] = action
            else:
                kind = "duplicate" if previous == action else "conflicting"
                unresolved.append(
                    f"{variable_id}:{kind}-action-for-observations:"
                    f"{history!r}"
                )

        # Preserve the selected action rule even when a time/round limit stops
        # on an unmaterialized HILP frontier. The public decision can then
        # report its incumbent root action while correctly withholding
        # utility, feasibility and duration-completeness certificates.
        # 即使搜索停在尚未 materialize 的 frontier，也先保留 incumbent 动作；
        # 随后将 utility/feasibility/duration 证书明确标记为不完整。
        if expansion is None:
            unresolved.append(f"{variable_id}:missing-expansion")
            continue

        metrics = expansion.metrics
        utility_terms.append(float(metrics.utility))
        risk_terms.append(float(metrics.chance_risk))

        for branch_index, branch in enumerate(expansion.observation_frontiers):
            if not branch.should_expand:
                leaf_histories.add(
                    item.observation_keys + (branch.observation,)
                )
                continue
            branch_name = f"{variable_id}:observation-{branch_index}"
            if not tree.variable_continues.get(variable_id, False):
                unresolved.append(f"{branch_name}:frontier-not-expanded")
                continue
            child_ids = [
                variable_by_node[child.node.node_id]
                for child in branch.child_frontier
                if child.node.node_id in variable_by_node
            ]
            if len(child_ids) != len(branch.child_frontier):
                unresolved.append(f"{branch_name}:undeclared-child-actions")
                continue
            selected_children = selected.intersection(child_ids)
            if len(selected_children) != 1:
                unresolved.append(
                    f"{branch_name}:expected-one-selected-child:"
                    f"found-{len(selected_children)}"
                )
            else:
                selected_edges.setdefault(variable_id, set()).update(
                    selected_children
                )

    reachable = set(selected_roots)
    pending = list(selected_roots)
    while pending:
        parent = pending.pop()
        for child in selected_edges.get(parent, ()):
            if child not in reachable:
                reachable.add(child)
                pending.append(child)
    disconnected = selected - reachable
    if disconnected:
        unresolved.append(
            "policy:disconnected-selected-variables:"
            + ",".join(sorted(disconnected))
        )

    duration_complete = not unresolved
    achieved_utility = sum(utility_terms) if duration_complete else None
    if achieved_utility is not None and not isfinite(achieved_utility):
        raise ValueError("Selected policy utility must be finite.")
    policy_risk = tree.initial_risk + sum(risk_terms)

    if not isfinite(policy_risk) or policy_risk < 0.0:
        raise ValueError(
            f"Constraint coefficient sum must be finite and non-negative: "
            f"{policy_risk!r}"
        )
    active_constraint_value = policy_risk if duration_complete else None
    risk_budget = tree.risk_budget
    feasible = (
        None
        if not duration_complete
        else risk_budget is None
        or policy_risk <= risk_budget
        # Match Gurobi's default absolute row-feasibility tolerance.
        # / 与 Gurobi 默认的线性约束绝对可行性容差保持一致。
        or isclose(policy_risk, risk_budget, rel_tol=0.0, abs_tol=1e-6)
    )
    if () not in actions_by_history:
        raise ValueError("Selected ILP incumbent has no root policy rule.")
    root, nodes, input_kind = _build_policy_graph(
        actions_by_history,
        leaf_histories,
    )
    return ConditionalPolicy(
        root=root,
        nodes=nodes,
        input_kind=input_kind,
        solver_status=result.status,
        duration_complete=duration_complete,
        achieved_utility=achieved_utility,
        active_constraint_value=active_constraint_value,
        feasible=feasible,
    )


def _selected_variable_ids(result: ILPSolveResult) -> set[str]:
    selected = {
        variable_id
        for variable_id, value in result.variable_values.items()
        if float(value) > 0.5
    }
    selected.update(result.selected_variables)
    return selected


def _build_policy_graph(
    actions: Mapping[ObservationHistory, tuple[str, Mapping[str, Any]]],
    leaves: set[ObservationHistory],
) -> tuple[str, tuple[PolicyNode, ...], PolicyInput]:
    """Turn selected history actions into a compact finite policy tree. / 将选中的历史动作转换为紧凑有限策略树。"""
    histories = sorted(actions, key=lambda item: (len(item), repr(item)))
    node_ids = {history: f"n{index}" for index, history in enumerate(histories)}
    transitions: dict[
        ObservationHistory, dict[ObservationKey, str | None]
    ] = defaultdict(dict)
    for history in histories:
        if history:
            transitions[history[:-1]][history[-1]] = node_ids[history]
    for history in leaves:
        if not history:
            raise ValueError("The policy root cannot be a leaf.")
        if history[-1] in transitions[history[:-1]]:
            raise ValueError("A policy outcome cannot both continue and terminate.")
        transitions[history[:-1]][history[-1]] = None

    observations = [
        observation for edges in transitions.values() for observation in edges
    ]
    state_input = bool(observations) and _is_state_key(observations[0])
    if any(_is_state_key(item) != state_input for item in observations):
        raise ValueError("A policy cannot mix state and observation inputs.")
    input_kind: PolicyInput = "state" if state_input else "observation"

    nodes = tuple(
        PolicyNode(
            node_id=node_ids[history],
            stage=len(history),
            action_label=actions[history][0],
            assignment=actions[history][1],
            transitions=dict(transitions.get(history, {})),
        )
        for history in histories
    )
    return node_ids[()], nodes, input_kind


def _policy_node_from_dict(value: Any, input_kind: PolicyInput) -> PolicyNode:
    if not isinstance(value, Mapping):
        raise TypeError("Each policy node must be an object.")
    action = value.get("action")
    transitions = value.get("transitions")
    if not isinstance(action, Mapping):
        raise TypeError("Policy node action must be an object.")
    if not isinstance(transitions, Sequence) or isinstance(transitions, str):
        raise TypeError("Policy transitions must be an array.")
    decoded = tuple(
        _policy_transition_from_dict(item, input_kind) for item in transitions
    )
    decoded_transitions = dict(decoded)
    if len(decoded_transitions) != len(decoded):
        raise ValueError("Policy node repeats an observation transition.")
    return PolicyNode(
        node_id=str(value["id"]),
        stage=int(value["stage"]),
        action_label=str(value["action_label"]),
        assignment=dict(action),
        transitions=decoded_transitions,
    )


def _policy_transition_from_dict(
    value: Any,
    input_kind: PolicyInput,
) -> tuple[ObservationKey, str | None]:
    if not isinstance(value, Mapping):
        raise TypeError("Each policy transition must be an object.")
    child = value.get("next")
    return (
        policy_input_key(value.get("observation"), input_kind),
        None if child is None else str(child),
    )


def _is_state_key(observation: ObservationKey) -> bool:
    return len(observation) == 1 and observation[0][0] == "__state__"


def _observation_values(
    observation: ObservationKey,
    input_kind: PolicyInput,
) -> Mapping[str, Any]:
    if input_kind == "state":
        if not _is_state_key(observation):
            raise ValueError("Expected a state-key policy transition.")
        return dict(observation[0][1])
    return dict(observation)


def policy_input_key(value: Any, input_kind: PolicyInput) -> ObservationKey:
    """Normalize a grounded observation or state for policy lookup. / 规范化 grounded 观测或状态以查找策略动作。"""
    if not isinstance(value, Mapping):
        raise TypeError("Policy transition observation must be an object.")
    key = tuple(
        sorted(
            ((str(name), _freeze(item)) for name, item in value.items()),
            key=lambda item: item[0],
        )
    )
    return (("__state__", key),) if input_kind == "state" else key


def _freeze(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze(item) for item in value)
    return value


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)
