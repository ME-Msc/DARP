"""Extract an executable conditional policy from selected ILP variables."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from math import isfinite, nextafter
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
    """A deterministic policy and its essential exact post-checks."""

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
    """Extract and exactly check the policy encoded by selected :math:`x_q`.

    The ILP already enforces policy flow.  This small post-check protects the
    public result from a partial HILP frontier, disconnected fake incumbents,
    and floating-point rounding of the selected policy's utility or active
    constraint.
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
    utility = Fraction(0)
    active_constraint = Fraction(0)
    utility_exact = True
    constraint_exact = True

    for variable_id in sorted(selected):
        item = tree.variable_items.get(variable_id)
        expansion = tree.variable_expansions.get(variable_id)
        if item is None or expansion is None:
            utility_exact = False
            constraint_exact = False
            unresolved.append(f"{variable_id}:missing-exact-expansion")
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

        metrics = expansion.metrics
        utility_value = getattr(metrics, "utility_exact", None)
        constraint_value = getattr(
            metrics,
            (
                "chance_risk_exact"
                if tree.constraint_type == "chance"
                else "penalty_exact"
            ),
            None,
        )
        if utility_value is None:
            utility_exact = False
        else:
            utility += Fraction(utility_value)
        if constraint_value is None:
            constraint_exact = False
        else:
            active_constraint += Fraction(constraint_value)

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
    achieved_utility = (
        float(utility)
        if duration_complete and utility_exact
        else None
    )
    if tree.constraint_type == "chance":
        active_constraint += Fraction(tree.initial_chance_risk_exact)

    active_constraint_value = (
        _upper_non_negative_float(active_constraint)
        if constraint_exact
        else None
    )
    budget = (
        Fraction.from_float(float(tree.constraint_budget))
        if tree.constraint_budget is not None
        else None
    )
    feasible = (
        None
        if not duration_complete or not constraint_exact
        else budget is None or active_constraint <= budget
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


def _upper_non_negative_float(value: Fraction) -> float:
    if value < 0:
        raise ValueError(
            f"Constraint coefficient sum must be non-negative: {value!r}"
        )
    rounded = float(value)
    if not isfinite(rounded):
        raise ValueError("Constraint coefficient sum is non-finite.")
    if Fraction.from_float(rounded) < value:
        rounded = nextafter(rounded, float("inf"))
    return rounded


def _selected_variable_ids(result: ILPSolveResult) -> set[str]:
    selected = {
        variable_id
        for variable_id, value in result.variable_values.items()
        if float(value) > 0.5
    }
    selected.update(result.selected_variables)
    return selected
