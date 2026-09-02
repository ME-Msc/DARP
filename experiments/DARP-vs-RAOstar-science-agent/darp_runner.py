"""Run DARP on the source-equivalent non-scheduling Science Agent model."""

from __future__ import annotations

from collections.abc import Mapping
from itertools import product
from math import isclose
from pathlib import Path
from time import perf_counter
from typing import Any

from darp.adapter.kernel import RDDLKernel, StateKey
from darp.adapter.loader import load_rddl
from darp.adapter.runtime import PyRDDLGymRuntime
from darp.ilp.gurobi import GurobiILPSolver
from darp.ilp.model import ILPLinearConstraint, ILPModelSpec, ILPVariable
from darp.model.and_or_tree import ANDORSearchInterface
from darp.model.duration_sidecar import load_duration_sidecar
from darp.model.risk_sidecar import load_risk_sidecar
from darp.planning.heuristic import HeuristicInput, UtilityHeuristic
from darp.planning.hilp import HILPPlanner

RDDL_DIR = Path(__file__).with_name("rddl")
DOMAIN = RDDL_DIR / "domain.rddl"
INSTANCE = RDDL_DIR / "instance_h5.rddl"
DURATION = RDDL_DIR / "duration.json"
RISK = RDDL_DIR / "risk.json"

SITE_NAMES = {
    0: "_start_",
    1: "minerals",
    2: "funny_rock",
    3: "geiser",
    4: "alien_lair",
    5: "relay",
    6: "_crash_",
}
VISITED = {
    1: "visited_minerals",
    2: "visited_funny_rock",
    3: "visited_geiser",
    4: "visited_alien_lair",
}
DISCOVERY = {
    1: "has_minerals",
    2: "has_funny_rock",
    3: "has_geiser",
    4: "has_alien_lair",
}
VALUES = {1: 10.0, 2: 5.0, 3: 4.0, 4: 100.0}

# RDDL always includes an all-false action; it represents relay/low-risk.
ACTION = {
    "noop": (5, 0.001),
    "go_minerals_low": (1, 0.001),
    "go_minerals_high": (1, 0.01),
    "go_funny_rock_low": (2, 0.001),
    "go_funny_rock_high": (2, 0.01),
    "go_geiser_low": (3, 0.001),
    "go_geiser_high": (3, 0.01),
    "go_alien_lair_low": (4, 0.001),
    "go_alien_lair_high": (4, 0.01),
    "go_relay_high": (5, 0.01),
}


class ScienceAgentKernel(RDDLKernel):
    """Expose the original model's state-dependent no-revisit action set."""

    def available_action_labels(
        self,
        belief: Mapping[StateKey, float],
    ) -> tuple[str, ...]:
        signatures = {
            (
                int(state["site"]),
                *(bool(state[name]) for name in VISITED.values()),
            )
            for key, probability in belief.items()
            if probability > 0
            for state in (dict(key),)
        }
        if len(signatures) != 1:
            raise ValueError("Science Agent belief must have known site/visited facts.")
        site, *visited = next(iter(signatures))
        if site in (5, 6):
            return ()
        visited_by_site = dict(zip(VISITED, visited, strict=True))
        return tuple(
            label
            for label, (target, _) in ACTION.items()
            if target != site and not visited_by_site.get(target, False)
        )


def initial_belief(kernel: RDDLKernel) -> Mapping[StateKey, float]:
    """Enumerate the four independent discovery priors from the source test."""

    declared = next(iter(kernel.initial_belief_from_model()))
    base = kernel.state_from_key(declared)
    priors = tuple(
        float(kernel.non_fluents[f"prior_{SITE_NAMES[site]}"])
        for site in DISCOVERY
    )
    belief: dict[StateKey, float] = {}
    for values in product((False, True), repeat=len(DISCOVERY)):
        probability = 1.0
        state = dict(base)
        for site, present, prior in zip(DISCOVERY, values, priors, strict=True):
            state[DISCOVERY[site]] = present
            probability *= prior if present else 1.0 - prior
        if probability > 0:
            belief[kernel.state_key(state)] = probability
    return belief


def _science_heuristic(value: HeuristicInput) -> float:
    """Return CCRockSample.heuristic() for one hidden state."""

    if int(value.state["site"]) in (5, 6):
        return 0.0
    return sum(
        reward
        for site, reward in VALUES.items()
        if not bool(value.state[VISITED[site]])
        and bool(value.state[DISCOVERY[site]])
    )


SCIENCE_HEURISTIC = UtilityHeuristic(
    name="raostar-science-unvisited-discoveries",
    evaluate=_science_heuristic,
    upper_bound=True,
)


def build_problem(seed: int = 0) -> tuple[Any, PyRDDLGymRuntime, ANDORSearchInterface, Any]:
    """Build the normal DARP inputs with only a model-local kernel subclass."""

    problem = load_rddl(DOMAIN, INSTANCE)
    runtime = PyRDDLGymRuntime(problem.env)
    runtime.reset(seed=seed)
    view = problem.build_grounded_view()
    view.validate_supported()
    risk = load_risk_sidecar(RISK)
    kernel = ScienceAgentKernel.from_grounded_model(
        problem.build_grounded_model(),
        risk=risk,
    )
    interface = ANDORSearchInterface.from_actions_and_observations(
        actions=view.action_choices(runtime),
        observation_scope=view.observation_scope(),
        kernel=kernel,
    )
    duration = load_duration_sidecar(DURATION)
    duration.validate_actions([choice.label for choice in interface.actions])
    return problem, runtime, interface, duration


def run_darp(
    *,
    delta: float,
    seed: int,
    timeout_s: float | None,
    reference_model: Any | None = None,
    reference_belief: Mapping[Any, Any] | None = None,
) -> dict[str, Any]:
    """Run one complete HILP search and return paired-table metrics."""

    _, runtime, interface, duration = build_problem(seed)
    root = initial_belief(interface.kernel)
    if reference_model is not None or reference_belief is not None:
        if reference_model is None or reference_belief is None:
            raise ValueError("Reference model and belief must be supplied together.")
        assert_reference_parity(interface, root, reference_model, reference_belief)
        # Parity enumerates the finite reachable model.  Rebuild so its caches
        # cannot warm the measured planner or inflate evaluated-state counts.
        _, runtime, interface, duration = build_problem(seed)
        root = initial_belief(interface.kernel)

    planner = HILPPlanner(
        frontier_heuristic=SCIENCE_HEURISTIC,
        risk_budget=delta,
        solver_time_limit_ms=None if timeout_s is None else timeout_s * 1000.0,
    )
    started = perf_counter()
    decision = planner.choose_action(
        runtime,
        interface,
        duration.evaluator(runtime.horizon),
        root_belief=root,
    )
    elapsed = perf_counter() - started
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
        "objective": float(utility),
        "risk": float(risk),
        "time_s": elapsed,
        "expanded_nodes": int(timing["expanded_nodes"]),
        # DARP has no paper-style particle counter.  This is the number of
        # distinct grounded states lazily compiled by this search.
        "evaluated_states": len(interface.kernel._state_index.id_to_key),
        "iterations": int(timing["partial_ilp_solves"]),
        "complete": True,
    }


def warm_up() -> None:
    """Initialize the Gurobi environment outside measured planner time."""

    variable = ILPVariable(var_id="warmup")
    result = GurobiILPSolver().solve(
        ILPModelSpec(
            name="science_agent_warmup",
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


def assert_reference_parity(
    interface: ANDORSearchInterface,
    root: Mapping[StateKey, float],
    model: Any,
    reference_belief: Mapping[Any, Any],
) -> None:
    """Check all reachable T/O/U/risk/action rows against Pedro Santana's model."""

    kernel = interface.kernel
    choices = {choice.label: choice.assignment for choice in interface.actions}
    if set(choices) != set(ACTION):
        raise ValueError("Science Agent RDDL action labels do not match the adapter.")
    _assert_distribution(
        "initial belief",
        {_hidden_tuple(key): probability for key, probability in root.items()},
        {
            _hidden_tuple_from_reference(entry[0]): float(entry[1])
            for entry in reference_belief.values()
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
        reference_state = _reference_state(model, source)
        actual_terminal = kernel.belief_is_terminal({source: 1.0})
        expected_terminal = bool(model.is_terminal(reference_state))
        if actual_terminal != expected_terminal:
            raise ValueError("Science Agent terminal parity failed.")
        if actual_terminal:
            _assert_close(
                "terminal value",
                0.0,
                float(model.terminal_value(reference_state)),
            )
            continue
        _assert_close(
            "heuristic",
            _science_heuristic(
                HeuristicInput(
                    state=state,
                    action_label="parity",
                    action={},
                    non_fluents=kernel.non_fluents,
                )
            ),
            float(model.heuristic(reference_state)),
        )
        reference_actions = {
            _action_label(action.goal_site, float(action.risk)): action
            for action in model.actions(reference_state)
        }
        available = kernel.available_action_labels({source: 1.0})
        if set(available) != set(reference_actions):
            raise ValueError("Science Agent available-action parity failed.")
        for label in available:
            assignment = choices[label]
            reference_action = reference_actions[label]
            actual_reward = kernel.utility_coefficient_for_mass(
                {source: 1.0}, assignment
            )
            expected_reward = float(model.value(reference_state, reference_action))
            _assert_close("reward", actual_reward, expected_reward)

            actual_joint: dict[tuple[Any, int], float] = {}
            expansion = kernel.expand_ordinary_mass({source: 1.0}, assignment)
            for outcome in expansion.observations:
                observation = int(dict(outcome.observation)["obs"])
                for target, probability in outcome.state_mass.items():
                    key = (_abstract_darp(target), observation)
                    actual_joint[key] = actual_joint.get(key, 0.0) + probability
                    frontier.add(target)

            expected_joint: dict[tuple[Any, int], float] = {}
            expected_risk = 0.0
            for target, transition_probability in model.state_transitions(
                reference_state, reference_action
            ):
                expected_risk += transition_probability * float(model.state_risk(target))
                for observation, observation_probability in model.observations(target):
                    key = (_abstract_reference(target), _observation_code(observation))
                    expected_joint[key] = (
                        expected_joint.get(key, 0.0)
                        + transition_probability * observation_probability
                    )
            _assert_distribution("transition/observation", actual_joint, expected_joint)
            actual_risk = kernel.safe_constraint_coefficient_for_mass(
                {source: 1.0}, assignment
            )
            _assert_close("risk", actual_risk, expected_risk)


def _reference_state(model: Any, key: StateKey) -> Mapping[str, Any]:
    state = dict(key)
    site = int(state["site"])
    if site == 6:
        return model.get_state((), "_crash_", True, set(), {}, tcs=[])
    visited = {
        SITE_NAMES[index]
        for index, fluent in VISITED.items()
        if bool(state[fluent])
    }
    if site != 0:
        visited.add("_start_")
    discoveries = {
        SITE_NAMES[index]: bool(state[fluent])
        for index, fluent in DISCOVERY.items()
    }
    discoveries["relay"] = False
    return model.get_state(
        model.sites[SITE_NAMES[site]]["coords"],
        SITE_NAMES[site],
        False,
        visited,
        discoveries,
        tcs=[],
    )


def _action_label(goal_site: str, risk: float) -> str:
    if goal_site == "relay" and isclose(risk, 0.001):
        return "noop"
    suffix = "high" if isclose(risk, 0.01) else "low"
    return f"go_{goal_site}_{suffix}"


def _abstract_darp(key: StateKey) -> Any:
    state = dict(key)
    if int(state["site"]) == 6:
        return (6,)
    return (
        int(state["site"]),
        *(bool(state[name]) for name in VISITED.values()),
        *(bool(state[name]) for name in DISCOVERY.values()),
    )


def _abstract_reference(state: Mapping[str, Any]) -> Any:
    if bool(state["crashed"]):
        return (6,)
    inverse = {name: index for index, name in SITE_NAMES.items()}
    visited = state["visited"]
    discoveries = state["new_discovery"]
    return (
        inverse[state["site"]],
        *(SITE_NAMES[index] in visited for index in VISITED),
        *(bool(discoveries[SITE_NAMES[index]]) for index in DISCOVERY),
    )


def _hidden_tuple(key: StateKey) -> tuple[bool, ...]:
    state = dict(key)
    return tuple(bool(state[name]) for name in DISCOVERY.values())


def _hidden_tuple_from_reference(state: Mapping[str, Any]) -> tuple[bool, ...]:
    discoveries = state["new_discovery"]
    return tuple(bool(discoveries[SITE_NAMES[index]]) for index in DISCOVERY)


def _observation_code(observation: tuple[str, str]) -> int:
    kind, site = observation
    if kind == "obstacle":
        return 12
    inverse = {name: index for index, name in SITE_NAMES.items()}
    return 2 * inverse[site] + (1 if kind == "new_discovery" else 0)


def _assert_distribution(
    name: str,
    actual: Mapping[Any, float],
    expected: Mapping[Any, float],
) -> None:
    if set(actual) != set(expected):
        raise ValueError(f"Science Agent {name} support parity failed.")
    for key in actual:
        _assert_close(name, float(actual[key]), float(expected[key]))


def _assert_close(name: str, actual: float, expected: float) -> None:
    if not isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(
            f"Science Agent {name} parity failed: {actual:.17g} != {expected:.17g}"
        )
