"""Extract an executable conditional policy from selected ILP variables."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isclose, isfinite
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from darp.ilp.model import ILPSolveResult
    from darp.planning.ilp_tree import PolicyTreeILP


@dataclass(frozen=True)
class PolicyRule:
    """One deterministic action indexed by its observation history."""

    observations: tuple[str, ...]
    action_label: str
    assignment: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "observations": list(self.observations),
            "action_label": self.action_label,
            "assignment": json_ready(self.assignment),
        }


@dataclass(frozen=True)
class ConditionalPolicy:
    """A deterministic policy and its essential numeric post-checks."""

    rules: tuple[PolicyRule, ...]
    solver_status: str
    duration_complete: bool
    achieved_utility: float | None
    active_constraint_value: float | None
    feasible: bool | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "complete": self.duration_complete,
            "rule_count": len(self.rules),
            "rules": [rule.to_dict() for rule in self.rules],
            "solver_status": self.solver_status,
            "achieved_utility": self.achieved_utility,
            "active_constraint_value": self.active_constraint_value,
            "feasible": self.feasible,
        }


def json_ready(value: Any) -> Any:
    """Convert numpy-backed action values to plain JSON values."""
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
    rules_by_observations: dict[tuple[str, ...], PolicyRule] = {}
    selected_edges: dict[str, set[str]] = {}
    utility_terms: list[float] = []
    constraint_terms: list[float] = []

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
            rule = PolicyRule(
                observations=tuple(item.node.history.observations),
                action_label=item.action_label,
                assignment=dict(assignment),
            )
            previous = rules_by_observations.get(rule.observations)
            if previous is None:
                rules_by_observations[rule.observations] = rule
            else:
                kind = (
                    "duplicate"
                    if previous.action_label == rule.action_label
                    and dict(previous.assignment) == dict(rule.assignment)
                    else "conflicting"
                )
                unresolved.append(
                    f"{variable_id}:{kind}-action-for-observations:"
                    f"{rule.observations!r}"
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
        constraint_terms.append(float(metrics.chance_risk))

        for branch_index, branch in enumerate(expansion.observation_frontiers):
            if not branch.should_expand:
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
    active_constraint = tree.initial_chance_risk + sum(constraint_terms)

    if not isfinite(active_constraint) or active_constraint < 0.0:
        raise ValueError(
            f"Constraint coefficient sum must be finite and non-negative: "
            f"{active_constraint!r}"
        )
    active_constraint_value = active_constraint if duration_complete else None
    budget = tree.constraint_budget
    feasible = (
        None
        if not duration_complete
        else budget is None
        or active_constraint <= budget
        or isclose(active_constraint, budget, rel_tol=1e-9, abs_tol=1e-9)
    )
    rules = tuple(
        sorted(
            rules_by_observations.values(),
            key=lambda rule: (
                len(rule.observations),
                rule.observations,
                rule.action_label,
            ),
        )
    )
    if not any(rule.observations == () for rule in rules):
        raise ValueError("Selected ILP incumbent has no root policy rule.")
    return ConditionalPolicy(
        rules=rules,
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
