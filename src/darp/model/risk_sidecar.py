"""Load CC-POMDP risk-state sets from JSON sidecars."""

from __future__ import annotations

import json
from collections.abc import Mapping
from math import isfinite
from pathlib import Path
from typing import Any

from darp.adapter.kernel import RiskConstraintSpec, StateSelector


class RiskSpecError(ValueError):
    """Raised when a risk sidecar violates the canonical schema."""


_FIELDS = frozenset({"budget", "risky_states"})


def load_risk_sidecar(path: str | Path) -> RiskConstraintSpec:
    """Load one paper CC-POMDP risk budget and risky-state set."""

    sidecar_path = Path(path).expanduser()
    if sidecar_path.suffix.lower() != ".json":
        raise RiskSpecError("Risk sidecar must be a .json file.")
    try:
        raw = json.loads(
            sidecar_path.read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
        )
    except OSError as error:
        raise RiskSpecError(f"Cannot read risk sidecar {sidecar_path}: {error}") from error
    except json.JSONDecodeError as error:
        raise RiskSpecError(f"Invalid risk sidecar JSON: {error}") from error
    if not isinstance(raw, Mapping):
        raise RiskSpecError("Risk sidecar root must be an object.")

    unknown = sorted(str(key) for key in set(raw) - _FIELDS)
    if unknown:
        raise RiskSpecError("Unknown risk sidecar fields: " + ", ".join(unknown))
    if "budget" not in raw:
        raise RiskSpecError("Risk sidecar requires a budget.")
    if "risky_states" not in raw:
        raise RiskSpecError("Risk sidecar requires risky_states.")
    budget = _number(raw["budget"], "budget")
    if not isfinite(budget) or not 0.0 <= budget <= 1.0:
        raise RiskSpecError("Risk sidecar budget must be finite and in [0, 1].")
    return RiskConstraintSpec(
        budget=budget,
        risky_states=_risky_states(raw["risky_states"]),
    )


def _risky_states(value: Any) -> tuple[StateSelector, ...]:
    if not isinstance(value, list):
        raise RiskSpecError("risky_states must be an array.")
    selectors: list[StateSelector] = []
    for index, selector in enumerate(value):
        if not isinstance(selector, Mapping) or not selector:
            raise RiskSpecError(f"risky_states[{index}] must be a non-empty object.")
        entries: list[tuple[str, bool | int]] = []
        for raw_name, expected in selector.items():
            if not isinstance(raw_name, str) or not raw_name:
                raise RiskSpecError(
                    f"risky_states[{index}] fluent names must be non-empty strings."
                )
            if type(expected) not in (bool, int):
                raise RiskSpecError(
                    f"risky_states[{index}].{raw_name} must be a Boolean or integer."
                )
            entries.append((raw_name, expected))
        canonical = tuple(sorted(entries))
        if canonical in selectors:
            raise RiskSpecError(f"risky_states contains duplicate selector {selector!r}.")
        selectors.append(canonical)
    return tuple(selectors)


def _number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RiskSpecError(f"{field} must be a JSON number.")
    return float(value)


def _reject_json_constant(value: str) -> None:
    raise RiskSpecError(f"Risk sidecar JSON does not allow {value}.")
