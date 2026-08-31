"""Shared file-based entry point for the CLI and experiments."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Literal

from darp.adapter.kernel import RDDLKernel, StateKey
from darp.adapter.loader import load_rddl
from darp.adapter.runtime import PyRDDLGymRuntime
from darp.model.and_or_tree import ANDORSearchInterface
from darp.model.duration import HistoryDurationEvaluator
from darp.model.duration_sidecar import DurationSidecar, load_duration_sidecar
from darp.planning.decision import ActionDecision
from darp.planning.full_ilp import FullILPPlanner
from darp.planning.heuristic import UtilityHeuristic
from darp.planning.hilp import HILPPlanner

PlannerName = Literal["hilp", "full-ilp"]
RootBeliefFactory = Callable[
    [RDDLKernel],
    Mapping[StateKey, float],
]
PreSolveCheck = Callable[
    [ANDORSearchInterface, DurationSidecar, HistoryDurationEvaluator, float | None],
    None,
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
    duration_path: str | Path,
    *,
    planner: PlannerName = "hilp",
    seed: int = 0,
    risk_budget: float | None = None,
    expansion_rounds: int | None = None,
    frontier_width: int | None = None,
    heuristic: UtilityHeuristic | None = None,
    terminal_heuristic: bool = False,
    timeout_s: float | None = 60.0,
    root_belief_factory: RootBeliefFactory | None = None,
    pre_solve_check: PreSolveCheck | None = None,
) -> DARPResult:
    """Load one RDDL problem, construct DARP, and run one search."""

    _validate_options(planner, heuristic, terminal_heuristic, timeout_s)
    problem = load_rddl(domain, instance)
    runtime = PyRDDLGymRuntime(problem.env)
    runtime.reset(seed=seed)
    duration = load_duration_sidecar(duration_path)
    interface = problem.build_grounded_view().build_and_or_interface(
        runtime,
        risk=duration.risk,
    )
    duration.validate_actions([choice.label for choice in interface.actions])
    duration.validate_state_fluents(
        getattr(interface.kernel, "state_names", ())
    )
    evaluator = duration.evaluator(horizon=runtime.horizon)
    budget = risk_budget if risk_budget is not None else duration.risk.budget

    root_belief = None
    if root_belief_factory is not None:
        kernel = interface.kernel
        if kernel is None:
            raise ValueError("An external root belief requires DARP's RDDL kernel.")
        root_belief = root_belief_factory(kernel)
        if not isinstance(root_belief, Mapping):
            raise TypeError("A root-belief factory must return a mapping.")
    if pre_solve_check is not None:
        pre_solve_check(interface, duration, evaluator, budget)

    limit_ms = None if timeout_s is None else timeout_s * 1000.0
    selected = (
        FullILPPlanner(
            risk_budget=budget,
            solver_time_limit_ms=limit_ms,
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
    if planner == "full-ilp" and heuristic is not None:
        raise ValueError(
            "External heuristics apply only to HILP; full-ILP uses the RDDL objective."
        )


__all__ = [
    "DARPResult",
    "PlannerName",
    "PreSolveCheck",
    "RootBeliefFactory",
    "solve_rddl",
]
