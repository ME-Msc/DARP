"""Run DARP on the explicitly specified, reduced power-supply benchmark."""

from __future__ import annotations

from collections.abc import Mapping
from math import isclose
from pathlib import Path
from typing import Any

from darp.adapter.kernel import RDDLKernel, StateKey
from darp.adapter.loader import load_rddl
from darp.adapter.runtime import PyRDDLGymRuntime
from darp.ilp.gurobi import GurobiILPSolver
from darp.ilp.model import ILPLinearConstraint, ILPModelSpec, ILPVariable
from darp.model.and_or_tree import ANDORSearchInterface
from darp.model.duration import FixedDurationModel
from darp.model.duration_sidecar import load_duration_sidecar
from darp.model.risk_sidecar import load_risk_sidecar
from darp.planning.heuristic import UtilityHeuristic
from darp.solve import solve_rddl

RDDL_DIR = Path(__file__).with_name("rddl")
DOMAIN = RDDL_DIR / "domain.rddl"
DURATION = RDDL_DIR / "duration.json"
RISK = RDDL_DIR / "risk.json"

STATE_NAMES = (
    "fault",
    "cb1_closed",
    "sd1_closed",
    "sd2_closed",
    "cb2_closed",
    "unsafe",
    "done",
    "depth",
)
ACTION_LABELS = (
    "noop",
    "open_cb1",
    "close_cb1",
    "open_sd1",
    "close_sd1",
    "open_sd2",
    "close_sd2",
    "open_cb2",
    "close_cb2",
)


def instance_path(sensor_count: int) -> Path:
    """Return the checked-in one- or two-breaker-sensor instance."""

    path = RDDL_DIR / f"instance_h4_s{sensor_count}.rddl"
    if not path.is_file():
        raise ValueError(f"No reduced PSR instance for sensor_count={sensor_count}.")
    return path


def read_instance(instance: Path) -> tuple[int, int]:
    """Read the sensor count and horizon from one RDDL instance."""

    problem = load_rddl(DOMAIN, instance)
    try:
        sensors = int(problem.env.model.non_fluents["sensor_count"])
        horizon = int(problem.env.model.horizon)
    except (AttributeError, KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Cannot read reduced PSR metadata from {instance}.") from error
    if sensors not in (1, 2) or horizon != 4:
        raise ValueError("The paired reduced PSR experiment requires s=1/2 and h=4.")
    return sensors, horizon


def initial_belief(kernel: RDDLKernel) -> Mapping[StateKey, float]:
    """Use the experiment's explicit uniform single-fault prior."""

    declared = kernel.initial_belief_from_model()
    if len(declared) != 1:
        raise ValueError("Reduced PSR requires one declared RDDL initial state.")
    state = kernel.state_from_key(next(iter(declared)))
    if int(state["fault"]) != 1:
        raise ValueError("The RDDL fault value must be the documented placeholder 1.")
    return {
        kernel.state_key({**state, "fault": fault}): 0.5
        for fault in (1, 2)
    }


PSR_HEURISTIC = UtilityHeuristic(
    name="paper-psr-only-faulty-lines-unpowered",
    # In this reduced topology every known-fault state can ultimately leave
    # only the faulty line unpowered, whose final penalty is zero.
    evaluate=lambda _: 0.0,
    upper_bound=True,
)


def run_darp(
    instance: Path,
    *,
    delta: float,
    seed: int,
    timeout_s: float | None,
    reference_model: Any | None = None,
    reference_belief: Mapping[Any, Any] | None = None,
) -> dict[str, Any]:
    """Run one complete HILP search and return comparison-table metrics."""

    if (reference_model is None) != (reference_belief is None):
        raise ValueError("Reference model and belief must be supplied together.")
    if reference_model is not None and reference_belief is not None:
        _validate_reference(instance, reference_model, reference_belief)

    result = solve_rddl(
        DOMAIN,
        instance,
        DURATION,
        risk_path=RISK,
        planner="hilp",
        seed=seed,
        risk_budget=delta,
        heuristic=PSR_HEURISTIC,
        # Finish is terminal but has an immediate breakdown cost, so its exact
        # RDDL reward must not be replaced by the zero continuation heuristic.
        terminal_heuristic=False,
        timeout_s=timeout_s,
        root_belief_factory=initial_belief,
    )
    decision = result.decision
    if not decision.complete or decision.policy.feasible is not True:
        raise RuntimeError(
            "DARP-HILP did not return a complete feasible policy: "
            f"status={decision.policy.solver_status}"
        )
    utility = decision.policy.achieved_utility
    risk = decision.policy.active_constraint_value
    if utility is None or risk is None:
        raise RuntimeError("DARP policy is missing objective or risk metrics.")
    timing = decision.timing
    return {
        "cost": -float(utility),
        "risk": float(risk),
        "time_s": result.elapsed_s,
        "expanded_nodes": int(timing["expanded_nodes"]),
        # RAO* counts evaluated belief particles; DARP has no identical counter.
        "evaluated_states": None,
        "iterations": int(timing["partial_ilp_solves"]),
        "complete": True,
    }


def warm_up() -> None:
    """Initialize the Gurobi environment outside measured planner time."""

    variable = ILPVariable(var_id="warmup")
    result = GurobiILPSolver().solve(
        ILPModelSpec(
            name="power_supply_warmup",
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


def _validate_reference(
    instance: Path,
    model: Any,
    reference_belief: Mapping[Any, Any],
) -> None:
    """Validate duration and every reachable T/O/U/risk row before timing."""

    sensors, horizon = read_instance(instance)
    if int(model.sensor_count) != sensors or int(model.horizon) != horizon:
        raise ValueError("DARP and RAO* reduced PSR metadata do not match.")

    problem = load_rddl(DOMAIN, instance)
    runtime = PyRDDLGymRuntime(problem.env)
    runtime.reset(seed=0)
    constraint = load_risk_sidecar(RISK)
    interface = problem.build_grounded_view().build_and_or_interface(
        runtime,
        risk=constraint,
    )
    duration = load_duration_sidecar(DURATION)
    duration.validate_actions([choice.label for choice in interface.actions])
    if not isinstance(duration.model, FixedDurationModel):
        raise TypeError("The paired RAO* comparison requires fixed durations.")
    values = (duration.model.default, *duration.model.durations.values())
    if any(float(value) != 1.0 for value in values) or duration.zeta != 0.0:
        raise ValueError("DARP and RAO* action depth match only for unit duration.")
    if duration.evaluator(horizon).action_depth_upper_bound() != horizon:
        raise ValueError("DARP duration horizon does not match RAO* action depth.")
    assert_reference_parity(interface, initial_belief(interface.kernel), model, reference_belief)


def assert_reference_parity(
    interface: ANDORSearchInterface,
    root: Mapping[StateKey, float],
    model: Any,
    reference_belief: Mapping[Any, Any],
) -> None:
    """Check all finite-horizon rows against the model passed to pinned RAO*."""

    kernel = interface.kernel
    if kernel is None:
        raise ValueError("Reduced PSR parity requires DARP's finite RDDL kernel.")
    choices = {choice.label: choice.assignment for choice in interface.actions}
    if tuple(choices) != ACTION_LABELS:
        raise ValueError("Unexpected reduced PSR RDDL action order.")

    reference_states = {
        _state_tuple(state): state
        for state, _ in (_belief_entry(entry) for entry in reference_belief.values())
    }
    _assert_distribution(
        "initial belief",
        {_state_tuple(dict(state)): probability for state, probability in root.items()},
        {
            _state_tuple(state): probability
            for state, probability in (_belief_entry(entry) for entry in reference_belief.values())
        },
    )

    frontier = set(root)
    seen: set[StateKey] = set()
    while frontier:
        source = frontier.pop()
        if source in seen:
            continue
        seen.add(source)
        state = dict(source)
        signature = _state_tuple(state)
        reference_state = reference_states[signature]
        actual_terminal = kernel.belief_is_terminal({source: 1.0})
        expected_terminal = bool(model.is_terminal(reference_state))
        if actual_terminal != expected_terminal:
            raise ValueError(f"Reduced PSR terminal parity failed at {signature!r}.")
        if actual_terminal:
            _assert_close("terminal value", 0.0, float(model.terminal_value(reference_state)))
            continue

        reference_actions = {
            model.action_label(action): action
            for action in model.actions(reference_state)
        }
        if set(reference_actions) != set(choices):
            raise ValueError(f"Reduced PSR action parity failed at {signature!r}.")
        _assert_close(
            "heuristic",
            0.0,
            float(model.heuristic(reference_state)),
        )

        for label, assignment in choices.items():
            action = reference_actions[label]
            _assert_close(
                "reward",
                kernel.utility_coefficient_for_mass({source: 1.0}, assignment),
                -float(model.value(reference_state, action)),
            )

            actual_joint: dict[tuple[tuple[Any, ...], int], float] = {}
            expansion = kernel.expand_ordinary_mass({source: 1.0}, assignment)
            for outcome in expansion.observations:
                observation = int(dict(outcome.observation)["obs"])
                for target, probability in outcome.state_mass.items():
                    key = (_state_tuple(dict(target)), observation)
                    actual_joint[key] = actual_joint.get(key, 0.0) + float(probability)
                    frontier.add(target)

            expected_joint: dict[tuple[tuple[Any, ...], int], float] = {}
            expected_risk = 0.0
            for target, transition_probability in model.state_transitions(
                reference_state,
                action,
            ):
                target_signature = _state_tuple(target)
                reference_states[target_signature] = target
                expected_risk += float(transition_probability) * float(model.state_risk(target))
                for observation, observation_probability in model.observations(target):
                    key = (target_signature, int(observation))
                    expected_joint[key] = (
                        expected_joint.get(key, 0.0)
                        + float(transition_probability) * float(observation_probability)
                    )
            _assert_distribution("transition/observation", actual_joint, expected_joint)
            _assert_close(
                "risk",
                kernel.safe_constraint_coefficient_for_mass({source: 1.0}, assignment),
                expected_risk,
            )


def _belief_entry(entry: Any) -> tuple[Mapping[str, Any], float]:
    state, probability = entry
    return state, float(probability)


def _state_tuple(state: Mapping[str, Any]) -> tuple[Any, ...]:
    return tuple(
        int(state[name]) if name in ("fault", "depth") else bool(state[name])
        for name in STATE_NAMES
    )


def _assert_distribution(
    name: str,
    actual: Mapping[Any, float],
    expected: Mapping[Any, float],
) -> None:
    if set(actual) != set(expected):
        raise ValueError(f"Reduced PSR {name} support parity failed.")
    for key in actual:
        _assert_close(name, float(actual[key]), float(expected[key]))


def _assert_close(name: str, actual: float, expected: float) -> None:
    if not isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(
            f"Reduced PSR {name} parity failed: {actual:.17g} != {expected:.17g}"
        )
