"""Configure DARP for the Grid comparison. 配置并运行 Grid 对比中的 DARP。"""

from __future__ import annotations

from pathlib import Path

from darp.planning.heuristic import HeuristicInput, UtilityHeuristic
from darp.solve import DARPResult, solve_rddl

RDDL_DIR = Path(__file__).with_name("rddl")
DOMAIN = RDDL_DIR / "domain.rddl"


def _negative_manhattan(value: HeuristicInput) -> int:
    row = int(value.state["grid_row"])
    column = int(value.state["grid_col"])
    goal_row = int(value.non_fluents["goal_row"])
    goal_column = int(value.non_fluents["goal_col"])
    return -(abs(row - goal_row) + abs(column - goal_column))


MANHATTAN = UtilityHeuristic(
    name="paper-grid-manhattan",
    evaluate=_negative_manhattan,
    upper_bound=True,
)


def run_darp(
    instance: Path,
    *,
    delta: float,
    seed: int,
    timeout_s: float | None,
) -> DARPResult:
    """Run one complete DARP-HILP search."""
    result = solve_rddl(
        DOMAIN,
        instance,
        planner="hilp",
        seed=seed,
        risk_budget=delta,
        heuristic=MANHATTAN,
        terminal_heuristic=True,
        timeout_s=timeout_s,
    )
    decision = result.decision
    if not decision.complete or decision.policy.feasible is not True:
        raise RuntimeError(
            "DARP-HILP did not return a complete feasible policy: "
            f"status={decision.policy.solver_status}"
        )
    if (
        decision.policy.achieved_utility is None
        or decision.policy.active_constraint_value is None
    ):
        raise RuntimeError("DARP policy is missing objective or risk metrics.")
    return result
