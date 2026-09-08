"""Shared file-based entry point for the CLI and experiments."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Literal

from darp.adapter.duration import build_duration_evaluator
from darp.adapter.kernel import RDDLKernel, StateKey
from darp.adapter.loader import load_rddl
from darp.adapter.runtime import PyRDDLGymRuntime
from darp.model.risk_sidecar import load_risk_sidecar
from darp.planning.decision import ActionDecision
from darp.planning.full_ilp import FullILPPlanner
from darp.planning.heuristic import UtilityHeuristic
from darp.planning.hilp import HILPPlanner

PlannerName = Literal["hilp", "full-ilp"]
RootBeliefFactory = Callable[
    [RDDLKernel],
    Mapping[StateKey, float],
]


@dataclass(frozen=True, slots=True)
class DARPResult:
    """One DARP decision with the common planner-only timing boundary."""

    decision: ActionDecision
    elapsed_s: float
    risk_budget: float | None


def solve_rddl(
    domain: str | Path,
    instance: str | Path,
    *,
    risk_path: str | Path,
    planner: PlannerName = "hilp",
    seed: int = 0,
    risk_budget: float | None = None,
    expansion_rounds: int | None = None,
    frontier_width: int | None = None,
    heuristic: UtilityHeuristic | None = None,
    terminal_heuristic: bool = False,
    timeout_s: float | None = 60.0,
    full_ilp_max_tree_nodes: int | None = 100_000,
    root_belief_factory: RootBeliefFactory | None = None,
) -> DARPResult:
    """Load one RDDL problem, construct DARP, and run one search."""

    _validate_options(planner, heuristic, terminal_heuristic, timeout_s)
    problem = load_rddl(domain, instance)
    runtime = PyRDDLGymRuntime(problem.env)
    runtime.reset(seed=seed)
    constraint = load_risk_sidecar(risk_path)
    interface = problem.build_grounded_view().build_and_or_interface(
        runtime,
        risk=constraint,
    )
    kernel = interface.kernel
    if kernel is None:
        raise ValueError("RDDL duration evaluation requires DARP's RDDL kernel.")
    evaluator = build_duration_evaluator(
        kernel,
        interface.actions,
        horizon=runtime.horizon,
    )
    budget = risk_budget if risk_budget is not None else constraint.budget

    root_belief = None
    if root_belief_factory is not None:
        root_belief = root_belief_factory(kernel)
        if not isinstance(root_belief, Mapping):
            raise TypeError("A root-belief factory must return a mapping.")
    limit_ms = None if timeout_s is None else timeout_s * 1000.0
    selected = (
        FullILPPlanner(
            risk_budget=budget,
            max_tree_nodes=full_ilp_max_tree_nodes,
            solver_time_limit_ms=limit_ms,
            terminal_heuristic=heuristic if terminal_heuristic else None,
        )
        if planner == "full-ilp"
        else HILPPlanner(
            expansion_rounds=expansion_rounds,
            frontier_width=frontier_width,
            frontier_heuristic=heuristic,
            terminal_heuristic=terminal_heuristic,
            risk_budget=budget,
            solver_time_limit_ms=limit_ms,
        )
    )
    started = perf_counter()
    decision = selected.choose_action(
        runtime,
        interface,
        evaluator,
        root_belief=root_belief,
    )
    return DARPResult(
        decision=decision,
        elapsed_s=perf_counter() - started,
        risk_budget=budget,
    )


def _validate_options(
    planner: str,
    heuristic: UtilityHeuristic | None,
    terminal_heuristic: bool,
    timeout_s: float | None,
) -> None:
    if planner not in ("hilp", "full-ilp"):
        raise ValueError(f"Unknown DARP planner: {planner!r}")
    if timeout_s is not None and timeout_s <= 0:
        raise ValueError("timeout_s must be positive when provided")
    if terminal_heuristic and heuristic is None:
        raise ValueError("terminal_heuristic requires an external heuristic")
    if planner == "full-ilp" and heuristic is not None and not terminal_heuristic:
        raise ValueError(
            "Full-ILP accepts an external heuristic only as a terminal value."
        )


__all__ = [
    "DARPResult",
    "PlannerName",
    "RootBeliefFactory",
    "solve_rddl",
]
