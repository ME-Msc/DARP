"""Run DARP and verify that its RDDL model matches the RAO* Grid."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from fractions import Fraction
from math import isclose
from pathlib import Path
from typing import Any

from darp.adapter.exact import ExactRDDLKernel, StateKey
from darp.adapter.loader import load_rddl
from darp.ilp.gurobi import GurobiILPSolver
from darp.ilp.model import ILPLinearConstraint, ILPModelSpec, ILPVariable
from darp.model.and_or_tree import ANDORSearchInterface
from darp.model.duration import FixedDurationModel
from darp.model.duration_sidecar import DurationSidecar, load_duration_sidecar
from darp.planning.heuristic import HeuristicInput, UtilityHeuristic
from darp.solve import solve_rddl

RDDL_DIR = Path(__file__).with_name("rddl")
DOMAIN = RDDL_DIR / "domain.rddl"
DURATION = RDDL_DIR / "duration.json"

# DARP's all-false Boolean action is the paper implementation's action 0 (L).
ACTION_TO_REFERENCE = {
    "noop": 0,
    "move_up": 1,
    "move_right": 2,
    "move_down": 3,
}


def instance_path(size: int, horizon: int) -> Path:
    """Return the checked-in RDDL instance for one paper scenario."""

    path = RDDL_DIR / f"instance_{size}_h{horizon}.rddl"
    if not path.is_file():
        raise ValueError(
            f"No DARP vs RAO* instance for size={size}, horizon={horizon}."
        )
    return path


def read_instance(instance: Path) -> tuple[int, int]:
    """Read and validate Grid size, horizon, initial state, and goal from RDDL."""

    problem = load_rddl(DOMAIN, instance)
    model = problem.env.model
    non_fluents = getattr(model, "non_fluents", {})
    state = getattr(model, "state_fluents", {})
    names = ("grid_row", "grid_col", "row_mod5", "col_mod5", "noise")
    try:
        max_row = int(non_fluents["max_row"])
        max_col = int(non_fluents["max_col"])
        goal = int(non_fluents["goal_row"]), int(non_fluents["goal_col"])
        horizon = int(model.horizon)
        initial = {name: int(state[name]) for name in names}
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Cannot read Grid metadata from {instance}") from exc
    if max_row < 0 or max_row != max_col:
        raise ValueError("The comparison requires a non-empty square Grid.")
    if horizon < 1:
        raise ValueError("RDDL horizon must be positive.")
    expected = {
        "grid_row": max_row,
        "grid_col": 0,
        "row_mod5": max_row % 5,
        "col_mod5": 0,
        "noise": 0,
    }
    if initial != expected or goal != (0, max_col):
        raise ValueError("RDDL initial state or goal does not match the paper Grid.")
    return max_row + 1, horizon


def default_risk_budget() -> float:
    budget = load_duration_sidecar(DURATION).risk.budget
    if budget is None:
        raise ValueError("duration.json must define the single-instance risk budget.")
    return float(budget)


def _negative_manhattan(value: HeuristicInput) -> int:
    """Return the paper Manhattan cost-to-go in DARP utility sign."""

    row = int(value.state["grid_row"])
    column = int(value.state["grid_col"])
    goal_row = int(value.non_fluents["goal_row"])
    goal_column = int(value.non_fluents["goal_col"])
    return -(abs(row - goal_row) + abs(column - goal_column))


MANHATTAN = UtilityHeuristic(
    name="paper-grid-manhattan",
    evaluate=_negative_manhattan,
    # Unit step cost and axis-aligned moves make -Manhattan an upper bound on
    # achievable utility even before transition noise and partial observation.
    upper_bound=True,
)


def initial_belief(kernel: ExactRDDLKernel) -> Mapping[StateKey, Fraction]:
    """Return the original deterministic position with pre-sampled move noise.

    RDDL evaluates different stochastic CPFs independently.  The domain keeps
    row/column motion correlated by storing one hidden categorical ``noise``
    state that is re-sampled for the following action.  This explicit root
    belief gives the first action the same .85/.075/.075 distribution.
    Marginalising ``noise`` yields exactly the paper's position process.
    """

    declared = kernel.initial_belief_from_model()
    if len(declared) != 1:
        raise ValueError("The paper Grid requires one declared initial RDDL state.")
    state_key, probability = next(iter(declared.items()))
    if Fraction.from_float(float(probability)) != 1:
        raise ValueError("The declared Grid initial state must have probability one.")
    base = kernel.state_from_key(state_key)
    if base.pop("noise", None) != 0:
        raise ValueError("The RDDL Grid initial noise placeholder must be zero.")
    intended = Fraction.from_float(
        float(kernel.non_fluents["transition_accuracy"])
    )
    slip = (Fraction(1) - intended) / 2
    return {
        kernel.state_key({**base, "noise": noise}): probability
        for noise, probability in ((0, intended), (1, slip), (2, slip))
    }


def assert_duration_parity(
    duration: DurationSidecar,
    *,
    rddl_horizon: float,
    reference: Any,
) -> None:
    """Require the only duration semantics understood by external RAO*."""

    model = duration.model
    if not isinstance(model, FixedDurationModel):
        raise TypeError(
            "RAO* has no duration-model input; comparison requires fixed unit duration."
        )
    durations = (float(model.default), *(float(value) for value in model.durations.values()))
    if any(value != 1.0 for value in durations) or duration.zeta != 0.0:
        raise ValueError(
            "RAO* action depth matches DARP only for D(s,a)=1 and zeta=0."
        )
    if float(rddl_horizon) != float(reference.horizon):
        raise ValueError(
            "DARP duration horizon and external fixed_horizon do not match: "
            f"{rddl_horizon} != {reference.horizon}"
        )
    depth_bound = duration.evaluator(rddl_horizon).action_depth_upper_bound()
    if depth_bound != int(reference.horizon):
        raise ValueError(
            "Unit-duration continuation does not produce the external action horizon: "
            f"{depth_bound} != {reference.horizon}"
        )
    risk = duration.risk
    if risk.constraint_type != "chance":
        raise ValueError("The Grid comparison requires a chance constraint.")
    if (
        risk.state_fluent_costs
        or risk.next_state_fluent_costs
        or risk.state_action_costs
    ):
        raise ValueError(
            "The Grid comparison requires first-entry risk on the next state only."
        )


def assert_reference_parity(
    interface: ANDORSearchInterface,
    reference: Any,
    *,
    horizon: int,
) -> None:
    """Check every state/action row reachable within the experiment horizon.

    The comparison marginalises DARP's auxiliary noise state, then checks the
    original implementation's transition, observation, reward, risk and
    Manhattan callbacks.  A mismatch aborts before either planner is timed.
    """

    kernel = interface.exact_kernel
    if kernel is None:
        raise ValueError("DARP vs RAO* parity requires DARP's exact kernel.")
    labels = tuple(choice.label for choice in interface.actions)
    if labels != tuple(ACTION_TO_REFERENCE):
        raise ValueError(
            "Unexpected RDDL action order: " + ", ".join(labels)
        )
    choices = {choice.label: choice.assignment for choice in interface.actions}

    root = initial_belief(kernel)
    root_positions: dict[tuple[int, int], Fraction] = defaultdict(Fraction)
    root_noise: dict[int, Fraction] = defaultdict(Fraction)
    for state, probability in root.items():
        root_positions[_position(state)] += probability
        root_noise[int(dict(state)["noise"])] += probability
    _assert_distribution(
        "initial-position",
        root_positions,
        reference.b0,
        tuple(reference.start_state),
        "b0",
    )
    intended = Fraction.from_float(float(kernel.non_fluents["transition_accuracy"]))
    _assert_distribution(
        "initial-noise",
        root_noise,
        {0: intended, 1: (1 - intended) / 2, 2: (1 - intended) / 2},
        tuple(reference.start_state),
        "b0",
    )

    positions = _reachable_positions(reference, horizon)
    noise_weights = {
        0: intended,
        1: (Fraction(1) - intended) / 2,
        2: (Fraction(1) - intended) / 2,
    }
    for position in positions:
        for label, reference_action in ACTION_TO_REFERENCE.items():
            action = choices[label]
            actual_transition: dict[tuple[int, int], Fraction] = defaultdict(Fraction)
            for noise, noise_probability in noise_weights.items():
                source = _state(position, noise)
                for target, probability in kernel.transition_fraction_distribution(
                    dict(source), action
                ).items():
                    target_position = _position(target)
                    actual_transition[target_position] += noise_probability * probability
                    expected_risk = float(reference.risk_model(target_position, reference_action))
                    actual_risk = float(
                        kernel.transition_failure_fraction(
                            kernel.state_key(source), target, action
                        )
                    )
                    _assert_close("risk", actual_risk, expected_risk, position, label)

            expected_transition = reference.trans_model(position, reference_action)
            _assert_distribution(
                "transition",
                actual_transition,
                expected_transition,
                position,
                label,
            )

            source_key = kernel.state_key(_state(position, 0))
            utility, _, _ = kernel.utility_coefficient_for_mass(
                {source_key: Fraction(1)}, action
            )
            _assert_close(
                "reward",
                utility,
                -float(reference.reward_model(position, reference_action)),
                position,
                label,
            )
            heuristic = MANHATTAN.evaluate(
                HeuristicInput(
                    state=dict(source_key),
                    action_label=label,
                    action=action,
                    non_fluents=kernel.non_fluents,
                )
            )
            _assert_close(
                "heuristic",
                float(heuristic),
                -float(reference.reward_heuristic(position, reference_action)),
                position,
                label,
            )

            for target_position in expected_transition:
                target = kernel.state_key(_state(target_position, 0))
                expected_observation = reference.obs_model(
                    target_position, reference_action
                )
                actual_observation = {
                    observation: kernel.observation_fraction_probability(
                        (("obs", observation),), target, action
                    )
                    for observation in expected_observation
                }
                if sum(actual_observation.values(), start=Fraction(0)) != 1:
                    raise ValueError("RDDL observation support does not sum to one.")
                _assert_distribution(
                    "observation",
                    actual_observation,
                    expected_observation,
                    target_position,
                    label,
                )

    _assert_goal_parity(kernel, choices, reference)


def _assert_goal_parity(
    kernel: ExactRDDLKernel,
    choices: Mapping[str, Mapping[str, Any]],
    reference: Any,
) -> None:
    """Check terminal semantics even when the short benchmark cannot reach goal."""

    goal = tuple(reference.goal_state)
    for noise in (0, 1, 2):
        state = _state(goal, noise)
        state_key = kernel.state_key(state)
        if not kernel.belief_is_terminal({state_key: Fraction(1)}):
            raise ValueError(f"RDDL goal is not terminal: {goal}")
        for label, action in choices.items():
            actual: dict[tuple[int, int], Fraction] = defaultdict(Fraction)
            for target, probability in kernel.transition_fraction_distribution(
                state, action
            ).items():
                actual[_position(target)] += probability
            _assert_distribution(
                "goal-transition",
                actual,
                {goal: 1.0},
                goal,
                label,
            )
            utility, _, _ = kernel.utility_coefficient_for_mass(
                {state_key: Fraction(1)}, action
            )
            _assert_close("goal-reward", utility, 0.0, goal, label)
            heuristic = MANHATTAN.evaluate(
                HeuristicInput(
                    state=dict(state_key),
                    action_label=label,
                    action=action,
                    non_fluents=kernel.non_fluents,
                )
            )
            _assert_close("goal-heuristic", float(heuristic), 0.0, goal, label)


def _reachable_positions(reference: Any, horizon: int) -> tuple[tuple[int, int], ...]:
    reached = {tuple(reference.start_state)}
    frontier = set(reached)
    for _ in range(horizon):
        following = {
            tuple(target)
            for state in frontier
            for action in reference.actions
            for target in reference.trans_model(state, action)
        } - reached
        if not following:
            break
        reached.update(following)
        frontier = following
    return tuple(sorted(reached))


def _state(position: tuple[int, int], noise: int) -> dict[str, int]:
    row, column = position
    return {
        "grid_row": row,
        "grid_col": column,
        "row_mod5": row % 5,
        "col_mod5": column % 5,
        "noise": noise,
    }


def _position(state: StateKey) -> tuple[int, int]:
    values = dict(state)
    return int(values["grid_row"]), int(values["grid_col"])


def _assert_distribution(
    name: str,
    actual: Mapping[Any, float | Fraction],
    expected: Mapping[Any, float],
    state: tuple[int, int],
    action: str,
) -> None:
    keys = set(actual) | set(expected)
    for key in keys:
        _assert_close(
            f"{name}[{key!r}]",
            float(actual.get(key, 0.0)),
            float(expected.get(key, 0.0)),
            state,
            action,
        )


def _assert_close(
    name: str,
    actual: float,
    expected: float,
    state: tuple[int, int],
    action: str,
) -> None:
    if not isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(
            f"DARP vs RAO* parity failed for {name}, state={state}, "
            f"action={action}: DARP={actual:.17g}, reference={expected:.17g}"
        )


def run_darp(
    instance: Path,
    *,
    delta: float,
    seed: int,
    timeout_s: float | None,
    reference_grid: Any | None = None,
) -> dict[str, Any]:
    """Run one complete DARP-HILP search and return comparison metrics."""

    def check_model(interface: Any, duration: Any, evaluator: Any, budget: Any) -> None:
        if budget != delta:
            raise ValueError(f"DARP risk budget mismatch: {budget} != {delta}")
        if reference_grid is not None:
            assert_duration_parity(
                duration,
                rddl_horizon=evaluator.horizon,
                reference=reference_grid,
            )
            assert_reference_parity(
                interface,
                reference_grid,
                horizon=int(reference_grid.horizon),
            )

    result = solve_rddl(
        DOMAIN,
        instance,
        DURATION,
        planner="hilp",
        seed=seed,
        risk_budget=delta,
        heuristic=MANHATTAN,
        terminal_heuristic=True,
        timeout_s=timeout_s,
        root_belief_factory=initial_belief,
        pre_solve_check=check_model,
    )
    decision = result.decision
    timing = decision.timing
    complete = bool(
        decision.policy.duration_complete
        and decision.policy.solver_status == "optimal"
        and not timing.get("solver_time_limit_hit", 0.0)
        and timing.get("frontier_refinement_exhausted", 0.0)
    )
    if not complete or decision.policy.feasible is not True:
        raise RuntimeError(
            "DARP-HILP did not return a complete feasible policy: "
            f"status={decision.policy.solver_status}"
        )
    utility = decision.policy.achieved_utility
    risk = decision.policy.active_constraint_value
    if utility is None or risk is None:
        raise RuntimeError("DARP policy is missing exact objective or risk metrics.")
    return {
        "objective": -float(utility),
        "risk": float(risk),
        "time_s": result.elapsed_s,
        "n": int(timing["expanded_nodes"] + timing["frontier_nodes"]),
        "iterations": int(timing["partial_ilp_solves"]),
        "complete": True,
        "certified": bool(decision.complete),
    }


def warm_up() -> None:
    """Initialize Gurobi before measured searches."""

    variable = ILPVariable(var_id="warmup")
    result = GurobiILPSolver().solve(
        ILPModelSpec(
            name="darp_vs_raostar_warmup",
            variables=(variable,),
            objective={variable.var_id: 1.0},
            constraints=(
                ILPLinearConstraint(
                    name="select",
                    coefficients={variable.var_id: 1.0},
                    sense="==",
                    rhs=1.0,
                ),
            ),
        ),
        time_limit_ms=30_000.0,
    )
    if result.status != "optimal":
        raise RuntimeError(f"Gurobi warm-up failed: {result.status}")


__all__ = [
    "ACTION_TO_REFERENCE",
    "DOMAIN",
    "DURATION",
    "MANHATTAN",
    "default_risk_budget",
    "initial_belief",
    "instance_path",
    "read_instance",
    "run_darp",
    "warm_up",
]
