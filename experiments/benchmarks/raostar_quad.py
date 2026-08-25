"""Shared-model bridge for the pinned upstream RAO* Quad scenario.

The model equations live exclusively in ``models/quad_model.py`` in the
external checkout.  This module validates and calls that object's callbacks;
it only supplies DARP's finite-kernel protocol, objective-sign conversion, and
an observation-history policy evaluator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
import hashlib
import importlib.util
import json
from math import isfinite, nextafter
from pathlib import Path
import sys
from typing import Any, Mapping

from darp.adapter.exact import (
    ExactConstraintMassExpansion,
    ExactConstraintMassOutcome,
    RiskConstraintSpec,
)
from darp.model.and_or_tree import ActionChoice, ANDORSearchInterface, ObservationScope
from darp.model.duration import FixedDurationModel, HistoryDurationEvaluator


WORLD_SIZE = (7, 7)
GOAL_STATE = (5, 5, 90)
QUAD_INITIAL = (1, 1, 90, 0)
GUEST_INITIAL = (3, 1, 90, 0)
INITIAL_STATE = (QUAD_INITIAL, GUEST_INITIAL)
QUAD_ACTIONS = ("forward", "turn-right-45", "turn-left-45")


def load_upstream_quad_model(checkout: Path) -> Any:
    """Import ``QuadModel`` directly from one validated external checkout."""

    source = Path(checkout).resolve() / "models" / "quad_model.py"
    if not source.is_file():
        raise FileNotFoundError(f"upstream QuadModel source does not exist: {source}")
    name = "_darp_upstream_quad_" + hashlib.sha256(str(source).encode()).hexdigest()[:16]
    spec = importlib.util.spec_from_file_location(name, source)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import upstream QuadModel from {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    model = module.QuadModel(WORLD_SIZE, GOAL_STATE)
    _validate_model(model)
    return model


def _validate_model(model: Any) -> None:
    required = (
        "actions",
        "state_transitions",
        "observations",
        "values",
        "heuristic",
        "state_risk",
        "is_terminal",
    )
    missing = [name for name in required if not callable(getattr(model, name, None))]
    if missing:
        raise TypeError(f"upstream QuadModel is missing callbacks: {missing}")
    if getattr(model, "optimization", None) != "minimize":
        raise ValueError("upstream QuadModel optimization must be 'minimize'")
    if tuple(getattr(model, "envSize", ())) != WORLD_SIZE:
        raise ValueError("upstream QuadModel world differs from quad_raos.py")
    if tuple(getattr(model, "goal", ())) != GOAL_STATE:
        raise ValueError("upstream QuadModel goal differs from quad_raos.py")


def _fraction(value: Any, *, probability: bool = False) -> Fraction:
    number = float(value)
    if not isfinite(number) or (probability and not 0.0 <= number <= 1.0):
        raise ValueError(f"invalid upstream numeric callback result: {value!r}")
    return Fraction.from_float(number)


def _normalize(values: Mapping[Any, float]) -> dict[Any, float]:
    cleaned = {state: float(value) for state, value in values.items() if float(value) > 0.0}
    total = sum(cleaned.values())
    if not isfinite(total) or total <= 0.0:
        raise ValueError("distribution has no positive finite mass")
    return {state: value / total for state, value in cleaned.items()}


def _upward(value: Fraction) -> float:
    rounded = float(value)
    if Fraction.from_float(rounded) < value:
        rounded = nextafter(rounded, float("inf"))
    return rounded


def observation_label(observation: Any) -> str:
    return json.dumps(observation, separators=(",", ":"), allow_nan=False)


def _observation_key(label: str) -> tuple[tuple[str, str], ...]:
    return (("quad_observation", label),)


@dataclass
class UpstreamQuadKernel:
    """DARP exact-kernel facade delegating every model callback upstream."""

    model: Any
    horizon: int
    risk: RiskConstraintSpec = field(
        default_factory=lambda: RiskConstraintSpec(constraint_type="chance")
    )

    def __post_init__(self) -> None:
        _validate_model(self.model)
        self.horizon = int(self.horizon)
        if self.horizon <= 0:
            raise ValueError("horizon must be positive")

    def initial_belief_from_model(self) -> Mapping[Any, float]:
        return {INITIAL_STATE: 1.0}

    def available_action_labels(self, belief: Mapping[Any, float]) -> tuple[str, ...]:
        """Match upstream RAO*'s union of positive-support state actions.

        Quad observations are deterministic, so every reachable posterior is
        a point mass and upstream's union agrees with per-state availability.
        The transition callback retains its fail-closed availability check for
        any noncanonical mixed belief.
        """

        states = tuple(_normalize(belief))
        available: set[str] = set()
        for state in states:
            labels = tuple(map(str, self.model.actions(state)))
            unknown = sorted(set(labels).difference(QUAD_ACTIONS))
            if unknown:
                raise ValueError(
                    f"upstream QuadModel returned unknown actions {unknown!r} "
                    f"at {state!r}"
                )
            available.update(labels)
        return tuple(label for label in QUAD_ACTIONS if label in available)

    def belief_is_terminal(self, belief: Mapping[Any, float]) -> bool:
        """Match RAO*'s ``terminal_prob=1.0`` stopping rule."""

        states = tuple(_normalize(belief))
        return all(bool(self.model.is_terminal(state)) for state in states)

    def initial_constraint_mass(self, belief: Mapping[Any, Any]) -> Mapping[Any, Fraction]:
        total = sum((_fraction(value, probability=True) for value in belief.values()), Fraction())
        if total <= 0:
            raise ValueError("initial belief has no positive mass")
        return {state: _fraction(value, probability=True) / total for state, value in belief.items() if float(value) > 0.0}

    def initial_safe_mass(self, belief: Mapping[Any, Any]) -> Mapping[Any, Fraction]:
        return {
            state: probability
            for state, probability in self.initial_constraint_mass(belief).items()
            if not self.state_is_unsafe(state)
        }

    @staticmethod
    def constraint_mass_belief(mass: Mapping[Any, Fraction]) -> Mapping[Any, float]:
        total = sum(mass.values(), Fraction())
        return {} if total <= 0 else {state: float(value / total) for state, value in mass.items()}

    @staticmethod
    def parse_action(action: Mapping[str, Any] | str) -> str:
        label = str(action.get("quad_action")) if isinstance(action, Mapping) else str(action)
        if label not in QUAD_ACTIONS:
            raise ValueError(f"unsupported Quad action: {label!r}")
        return label

    def _assert_available(self, state: Any, action: str) -> None:
        if action not in tuple(self.model.actions(state)):
            raise ValueError(f"upstream QuadModel action {action!r} unavailable at {state!r}")

    def transition_fraction_distribution(self, state: Any, action: Mapping[str, Any] | str) -> Mapping[Any, Fraction]:
        label = self.parse_action(action)
        self._assert_available(state, label)
        result: dict[Any, Fraction] = {}
        for successor, probability in self.model.state_transitions(state, label):
            exact = _fraction(probability, probability=True)
            if exact > 0:
                result[successor] = result.get(successor, Fraction()) + exact
        total = sum(result.values(), Fraction())
        if total != 1:
            raise ValueError(f"upstream transition probabilities sum to {float(total)!r}, not 1")
        return result

    def _observation_fraction_distribution(self, state: Any) -> Mapping[str, Fraction]:
        result: dict[str, Fraction] = {}
        for observation, probability in self.model.observations(state):
            label = observation_label(observation)
            exact = _fraction(probability, probability=True)
            if exact > 0:
                result[label] = result.get(label, Fraction()) + exact
        total = sum(result.values(), Fraction())
        if total != 1:
            raise ValueError(f"upstream observation probabilities sum to {float(total)!r}, not 1")
        return result

    def _mass_observations(self, mass: Mapping[Any, Fraction]) -> tuple[ExactConstraintMassOutcome, ...]:
        grouped: dict[str, dict[Any, Fraction]] = {}
        for state, state_mass in mass.items():
            for label, probability in self._observation_fraction_distribution(state).items():
                branch = grouped.setdefault(label, {})
                branch[state] = branch.get(state, Fraction()) + state_mass * probability
        outcomes = []
        for label in sorted(grouped):
            branch = grouped[label]
            outcomes.append(ExactConstraintMassOutcome(
                observation=_observation_key(label), label=label, state_mass=branch,
            ))
        return tuple(outcomes)

    def expand_ordinary_mass(self, mass: Mapping[Any, Fraction], action: Mapping[str, Any] | str) -> ExactConstraintMassExpansion:
        post: dict[Any, Fraction] = {}
        for state, state_mass in mass.items():
            for successor, probability in self.transition_fraction_distribution(state, action).items():
                post[successor] = post.get(successor, Fraction()) + state_mass * probability
        return ExactConstraintMassExpansion(
            coefficient=0.0, post_action_mass=post,
            observations=self._mass_observations(post),
        )

    def expand_safe_constraint_mass(self, mass: Mapping[Any, Fraction], action: Mapping[str, Any] | str) -> ExactConstraintMassExpansion:
        ordinary = self.expand_ordinary_mass(mass, action)
        failure = sum((value for state, value in ordinary.post_action_mass.items() if self.state_is_unsafe(state)), Fraction())
        surviving = {state: value for state, value in ordinary.post_action_mass.items() if not self.state_is_unsafe(state)}
        return ExactConstraintMassExpansion(
            coefficient=_upward(failure), coefficient_exact=failure,
            post_action_mass=surviving,
            observations=self._mass_observations(surviving),
        )

    def utility_coefficient_for_mass(self, mass: Mapping[Any, Fraction], action: Mapping[str, Any] | str) -> tuple[float, bool, Fraction]:
        label = self.parse_action(action)
        exact = Fraction()
        for state, probability in mass.items():
            self._assert_available(state, label)
            native = _fraction(self.model.values(state, label))
            # RAO* assigns model.heuristic to both fixed-horizon and domain-
            # terminal leaves.  Fold exactly those successor leaf values into
            # the parent coefficient because DARP terminal OR nodes have no
            # action variable of their own.
            native += sum(
                (
                    transition_probability * _fraction(self.model.heuristic(successor))
                    for successor, transition_probability in
                    self.transition_fraction_distribution(state, label).items()
                    if int(successor[0][3]) == self.horizon
                    or bool(self.model.is_terminal(successor))
                ),
                Fraction(),
            )
            exact -= probability * native
        rounded = float(exact)
        return rounded, Fraction.from_float(rounded) == exact, exact

    def belief_state_risk_fraction(self, belief: Mapping[Any, Any]) -> Fraction:
        mass = self.initial_constraint_mass(belief)
        return sum(
            (
                probability
                for state, probability in mass.items()
                if self.state_is_unsafe(state)
            ),
            Fraction(),
        )

    def state_is_unsafe(self, state: Any) -> bool:
        risk = float(self.model.state_risk(state))
        if not isfinite(risk) or not 0.0 <= risk <= 1.0:
            raise ValueError(f"invalid upstream state_risk: {risk!r}")
        if risk not in (0.0, 1.0):
            raise ValueError("first-entry bridge requires binary upstream state_risk")
        return risk == 1.0


@dataclass
class UpstreamQuadRuntime:
    state: Any = INITIAL_STATE


def build_upstream_quad_problem(model: Any, horizon: int) -> tuple[UpstreamQuadRuntime, ANDORSearchInterface, HistoryDurationEvaluator]:
    kernel = UpstreamQuadKernel(model=model, horizon=horizon)
    runtime = UpstreamQuadRuntime()
    interface = ANDORSearchInterface.from_actions_and_observations(
        actions=tuple(ActionChoice(label=action, assignment={"quad_action": action}) for action in QUAD_ACTIONS),
        observation_scope=ObservationScope(mode="pomdp-observation"),
        exact_kernel=kernel,
    )
    duration = HistoryDurationEvaluator(model=FixedDurationModel(durations={}, default=1.0), horizon=float(horizon))
    return runtime, interface, duration


@dataclass(frozen=True)
class QuadPolicyEvaluation:
    native_cost_exact: Fraction
    first_entry_risk_exact: Fraction

    @property
    def native_cost(self) -> float:
        return float(self.native_cost_exact)

    @property
    def darp_utility(self) -> float:
        return -self.native_cost

    @property
    def first_entry_risk(self) -> float:
        return float(self.first_entry_risk_exact)


def evaluate_quad_policy(model: Any, policy: Mapping[tuple[str, ...], str], horizon: int) -> QuadPolicyEvaluation:
    """Recompute cost and first-entry risk solely through upstream callbacks."""

    _validate_model(model)
    normalized = {tuple(map(str, history)): str(action) for history, action in policy.items()}
    def recurse(mass: Mapping[Any, Fraction], safe_mass: Mapping[Any, Fraction], history: tuple[str, ...], depth: int) -> tuple[Fraction, Fraction]:
        positive_states = tuple(state for state, probability in mass.items() if probability > 0)
        if not positive_states:
            raise ValueError("policy evaluator reached an empty belief")
        if depth == horizon or all(bool(model.is_terminal(state)) for state in positive_states):
            terminal_heuristic = sum(
                (
                    probability * _fraction(model.heuristic(state))
                    for state, probability in mass.items()
                ),
                Fraction(),
            )
            return terminal_heuristic, Fraction()
        action = normalized.get(history)
        if action is None:
            raise ValueError(f"policy is missing reachable history {history!r}")
        if action not in QUAD_ACTIONS:
            raise ValueError(f"policy contains invalid action {action!r}")
        cost = sum((probability * _fraction(model.values(state, action)) for state, probability in mass.items()), Fraction())
        next_by_observation: dict[str, dict[Any, Fraction]] = {}
        safe_by_observation: dict[str, dict[Any, Fraction]] = {}
        failure = Fraction()
        for source, source_mass in mass.items():
            if action not in tuple(model.actions(source)):
                raise ValueError(
                    f"upstream QuadModel action {action!r} unavailable at {source!r}"
                )
            for successor, transition_probability in model.state_transitions(source, action):
                tp = _fraction(transition_probability, probability=True)
                for observation, observation_probability in model.observations(successor):
                    label = observation_label(observation)
                    op = _fraction(observation_probability, probability=True)
                    branch = next_by_observation.setdefault(label, {})
                    branch[successor] = branch.get(successor, Fraction()) + source_mass * tp * op
        for source, source_mass in safe_mass.items():
            if action not in tuple(model.actions(source)):
                raise ValueError(
                    f"upstream QuadModel action {action!r} unavailable at {source!r}"
                )
            for successor, transition_probability in model.state_transitions(source, action):
                tp = _fraction(transition_probability, probability=True)
                unsafe = float(model.state_risk(successor))
                if unsafe not in (0.0, 1.0):
                    raise ValueError("first-entry evaluator requires binary upstream state_risk")
                if unsafe == 1.0:
                    failure += source_mass * tp
                    continue
                for observation, observation_probability in model.observations(successor):
                    label = observation_label(observation)
                    op = _fraction(observation_probability, probability=True)
                    branch = safe_by_observation.setdefault(label, {})
                    branch[successor] = branch.get(successor, Fraction()) + source_mass * tp * op
        risk = failure
        for label, child_mass in next_by_observation.items():
            child_cost, child_risk = recurse(child_mass, safe_by_observation.get(label, {}), history + (label,), depth + 1)
            cost += child_cost
            risk += child_risk
        return cost, risk

    cost, risk = recurse({INITIAL_STATE: Fraction(1)}, {INITIAL_STATE: Fraction(1)}, (), 0)
    if not 0 <= risk <= 1:
        raise RuntimeError(f"invalid first-entry risk {risk}")
    return QuadPolicyEvaluation(cost, risk)


def serialize_quad_policy(policy: Mapping[tuple[str, ...], str], *, horizon: int) -> dict[str, Any]:
    rules = [{"observations": list(history), "action": action} for history, action in sorted(policy.items(), key=lambda item: (len(item[0]), item[0], item[1]))]
    payload: dict[str, Any] = {"schema_version": 1, "scenario": "upstream-quad", "horizon": int(horizon), "rules": rules}
    payload["sha256"] = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return payload


def deserialize_quad_policy(payload: Mapping[str, Any]) -> dict[tuple[str, ...], str]:
    unsigned = {key: payload[key] for key in ("schema_version", "scenario", "horizon", "rules")}
    if unsigned["schema_version"] != 1 or unsigned["scenario"] != "upstream-quad":
        raise ValueError("unsupported Quad policy protocol")
    digest = hashlib.sha256(json.dumps(unsigned, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    if payload.get("sha256") != digest:
        raise ValueError("Quad policy protocol digest mismatch")
    result: dict[tuple[str, ...], str] = {}
    for rule in unsigned["rules"]:
        history = tuple(map(str, rule["observations"]))
        if history in result:
            raise ValueError(f"duplicate Quad policy history: {history!r}")
        result[history] = str(rule["action"])
    return result
