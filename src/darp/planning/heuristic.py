"""Small public interface for HILP utility heuristics."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from importlib import import_module
from math import isfinite
from numbers import Real
from typing import Any

from darp.adapter.kernel import StateKey


@dataclass(frozen=True, slots=True)
class HeuristicInput:
    """One state/action point passed to a user heuristic.

    ``non_fluents`` exposes model constants without coupling the planner to a
    particular domain.  A callback returns a utility-to-go value for DARP's
    maximization objective; a cost-to-go heuristic must therefore be negated.
    """

    state: Mapping[str, Any]
    action_label: str
    action: Mapping[str, Any]
    non_fluents: Mapping[str, Any]


HeuristicFunction = Callable[[HeuristicInput], Real]


@dataclass(frozen=True, slots=True)
class UtilityHeuristic:
    """Describe an external state utility heuristic used at HILP frontiers.

    The planner applies the paper's history-probability weighting itself:

    ``h_q = sum_s ordinary_mass_q[s] * value(s, a_q)``.

    Set ``upper_bound`` only when the callback is an admissible upper bound for
    DARP's maximization objective.  The flag affects optimality certification,
    never the ILP solution itself.
    """

    name: str
    evaluate: HeuristicFunction
    upper_bound: bool = False

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("A utility heuristic must have a non-empty name.")
        if not callable(self.evaluate):
            raise TypeError("UtilityHeuristic.evaluate must be callable.")


def load_utility_heuristic(spec: str) -> UtilityHeuristic:
    """Load ``module:attribute`` as a :class:`UtilityHeuristic`.

    A bare callable is accepted as a convenient, non-certifying heuristic.
    Exporting ``UtilityHeuristic`` explicitly is recommended because it records
    a stable name and whether the bound is admissible.
    """

    module_name, separator, attribute = spec.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("Heuristic must use the form 'module:attribute'.")
    value = getattr(import_module(module_name), attribute)
    if isinstance(value, UtilityHeuristic):
        return value
    if callable(value):
        return UtilityHeuristic(name=spec, evaluate=value)
    raise TypeError(
        f"{spec!r} must resolve to UtilityHeuristic or a callable, "
        f"not {type(value).__name__}."
    )


def history_heuristic_coefficient(
    heuristic: UtilityHeuristic,
    *,
    state_mass: Mapping[StateKey, float],
    action_label: str,
    action: Mapping[str, Any],
    non_fluents: Mapping[str, Any],
) -> float:
    """Return the paper coefficient ``rho(q) E[h(S,a) | q]``.

    ``state_mass`` is already the unnormalised ordinary history mass, so no
    second probability scale or belief normalisation is applied here.
    """

    terms: list[float] = []
    for state_key, probability in state_mass.items():
        value = heuristic.evaluate(
            HeuristicInput(
                state=dict(state_key),
                action_label=action_label,
                action=action,
                non_fluents=non_fluents,
            )
        )
        terms.append(float(probability) * _finite_float(value))
    coefficient = sum(terms)
    if not isfinite(coefficient):
        raise ValueError(f"Heuristic {heuristic.name!r} returned a non-finite value.")
    return coefficient


def _finite_float(value: Real) -> float:
    """Validate and convert one user heuristic value."""

    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("A utility heuristic must return a finite real number.")
    numeric = float(value)
    if not isfinite(numeric):
        raise ValueError("A utility heuristic must return a finite value.")
    return numeric


__all__ = [
    "HeuristicInput",
    "UtilityHeuristic",
    "history_heuristic_coefficient",
    "load_utility_heuristic",
]
