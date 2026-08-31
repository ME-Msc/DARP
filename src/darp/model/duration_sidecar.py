"""Load canonical JSON duration and constraint sidecars."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any

from darp.adapter.kernel import RiskConstraintSpec, RiskConstraintType
from darp.model.duration import (
    ActionName,
    ChanceConstrainedDurationModel,
    DurationModel,
    FixedDurationModel,
    GaussianDurationModel,
    HistoryDurationEvaluator,
    StateDependentDurationModel,
)


class DurationSpecError(ValueError):
    """Raised when a duration sidecar violates the canonical schema."""


@dataclass(frozen=True)
class DurationSidecar:
    """Parsed duration model, stopping threshold, and paper constraint."""

    model: DurationModel
    zeta: float
    risk: RiskConstraintSpec

    def evaluator(self, horizon: float) -> HistoryDurationEvaluator:
        return HistoryDurationEvaluator(
            model=self.model,
            horizon=float(horizon),
            zeta=self.zeta,
        )

    def validate_actions(
        self,
        action_names: set[str] | tuple[str, ...] | list[str],
    ) -> None:
        available = {str(action) for action in action_names}
        unknown = sorted(
            action for action in _duration_action_names(self.model) if action not in available
        )
        if unknown:
            raise DurationSpecError(
                "Duration sidecar references unknown actions: " + ", ".join(unknown)
            )

    def validate_state_fluents(
        self,
        state_fluent_names: set[str] | tuple[str, ...] | list[str],
    ) -> None:
        available = {str(name) for name in state_fluent_names}
        unknown = sorted(_duration_state_selector_names(self.model) - available)
        if unknown:
            raise DurationSpecError(
                "Duration sidecar references unknown state fluents: "
                + ", ".join(unknown)
            )


def load_duration_sidecar(path: str | Path) -> DurationSidecar:
    """Load a canonical JSON sidecar."""

    sidecar_path = Path(path).expanduser()
    if sidecar_path.suffix.lower() != ".json":
        raise DurationSpecError("Duration sidecar must be a .json file.")
    try:
        text = sidecar_path.read_text(encoding="utf-8")
        raw = json.loads(text, parse_constant=_reject_json_constant)
    except OSError as error:
        raise DurationSpecError(f"Cannot read duration sidecar {sidecar_path}: {error}") from error
    except json.JSONDecodeError as error:
        raise DurationSpecError(f"Invalid duration sidecar JSON: {error}") from error
    if not isinstance(raw, Mapping):
        raise DurationSpecError("Duration sidecar root must be an object.")
    return build_duration_sidecar(raw)


def build_duration_sidecar(raw: Mapping[str, Any]) -> DurationSidecar:
    """Build a sidecar from the same canonical mapping accepted in JSON."""

    if not isinstance(raw, Mapping):
        raise DurationSpecError("Duration sidecar root must be an object.")
    kind = _validate_sidecar_schema(raw)
    try:
        model = _build_duration_model(kind, raw)
        # The paper defines fixed duration with varsigma = 0. Other duration
        # models may supply their percentile threshold explicitly.
        zeta = 0.0 if kind == "fixed" else float(raw.get("zeta", 0.0))
        if not isfinite(zeta) or zeta < 0.0:
            raise DurationSpecError("Duration sidecar zeta must be finite and non-negative.")
        if isinstance(model, (GaussianDurationModel, ChanceConstrainedDurationModel)) and zeta > 1.0:
            raise DurationSpecError(
                "Probabilistic duration sidecar zeta must be in [0, 1]."
            )
        risk = _build_risk_spec(raw.get("risk"))
    except DurationSpecError:
        raise
    except (TypeError, ValueError) as error:
        raise DurationSpecError(str(error)) from error
    return DurationSidecar(model=model, zeta=zeta, risk=risk)


_COMMON_FIELDS = frozenset({"kind", "zeta", "risk"})
_KIND_FIELDS = {
    "fixed": {"kind", "default", "actions", "risk"},
    "expected": _COMMON_FIELDS | {"default", "state_actions"},
    "chance": _COMMON_FIELDS | {"default", "state_actions"},
    "gaussian": _COMMON_FIELDS
    | {"default_mean", "default_variance", "state_actions"},
}


def _validate_sidecar_schema(raw: Mapping[str, Any]) -> str:
    kind = raw.get("kind")
    if not isinstance(kind, str) or kind not in _KIND_FIELDS:
        raise DurationSpecError(
            "Duration sidecar kind must be 'fixed', 'expected', 'chance', or 'gaussian'."
        )
    unknown = sorted(str(key) for key in set(raw) - _KIND_FIELDS[kind])
    if unknown:
        raise DurationSpecError("Unknown duration sidecar fields: " + ", ".join(unknown))
    if kind == "fixed" and "default" not in raw:
        raise DurationSpecError("Fixed duration sidecar requires a default duration.")
    return kind


def _build_duration_model(kind: str, config: Mapping[str, Any]) -> DurationModel:
    if kind == "fixed":
        return FixedDurationModel(
            durations=_number_mapping(config.get("actions", {}), field_name="actions"),
            default=float(config["default"]),
        )
    if kind == "expected":
        return StateDependentDurationModel(
            durations=_state_action_numbers(config.get("state_actions", {})),
            default=float(config.get("default", 1.0)),
        )
    if kind == "chance":
        return ChanceConstrainedDurationModel(
            durations=_state_action_numbers(config.get("state_actions", {})),
            default=float(config.get("default", 1.0)),
        )
    means, variances = _state_action_gaussians(config.get("state_actions", {}))
    return GaussianDurationModel(
        means=means,
        variances=variances,
        default_mean=float(config.get("default_mean", 1.0)),
        default_variance=float(config.get("default_variance", 0.0)),
    )


_RISK_FIELDS = frozenset(
    {
        "budget",
        "constraint_type",
        "state_fluents",
        "next_state_fluents",
        "state_actions",
        "next_state_actions",
    }
)


def _build_risk_spec(value: Any) -> RiskConstraintSpec:
    if value is None:
        return RiskConstraintSpec()
    if not isinstance(value, Mapping):
        raise DurationSpecError("risk must be an object when present.")
    unknown = sorted(str(key) for key in set(value) - _RISK_FIELDS)
    if unknown:
        raise DurationSpecError("Unknown risk sidecar fields: " + ", ".join(unknown))
    if "constraint_type" not in value:
        raise DurationSpecError(
            "risk.constraint_type is required and must be 'chance' or 'expected'."
        )
    return RiskConstraintSpec(
        budget=(
            float(value["budget"])
            if "budget" in value and value["budget"] is not None
            else None
        ),
        state_fluent_costs=_number_mapping(
            value.get("state_fluents", {}),
            field_name="risk.state_fluents",
        ),
        next_state_fluent_costs=_number_mapping(
            value.get("next_state_fluents", {}),
            field_name="risk.next_state_fluents",
        ),
        state_action_costs=_state_action_numbers(
            value.get("state_actions", {}),
            field_name="risk.state_actions",
        ),
        next_state_action_costs=_state_action_numbers(
            value.get("next_state_actions", {}),
            field_name="risk.next_state_actions",
        ),
        constraint_type=_risk_constraint_type(value["constraint_type"]),
    )


def _risk_constraint_type(value: Any) -> RiskConstraintType:
    if value == "chance":
        return "chance"
    if value == "expected":
        return "expected"
    raise DurationSpecError(
        "risk.constraint_type must be 'chance' (CC-POMDP) or "
        "'expected' (C-POMDP)."
    )


def _duration_action_names(model: DurationModel) -> tuple[ActionName, ...]:
    actions: set[str] = set()
    if isinstance(model, FixedDurationModel):
        actions.update(str(action) for action in model.durations)
    elif isinstance(model, (StateDependentDurationModel, ChanceConstrainedDurationModel)):
        actions.update(str(action) for _, action in model.durations)
    elif isinstance(model, GaussianDurationModel):
        actions.update(str(action) for _, action in model.means)
        actions.update(str(action) for _, action in model.variances)
    return tuple(sorted(actions))


def _duration_state_selector_names(model: DurationModel) -> set[str]:
    selectors: set[object] = set()
    if isinstance(model, (StateDependentDurationModel, ChanceConstrainedDurationModel)):
        selectors.update(state for state, _ in model.durations)
    elif isinstance(model, GaussianDurationModel):
        selectors.update(state for state, _ in model.means)
        selectors.update(state for state, _ in model.variances)
    names: set[str] = set()
    for selector in selectors:
        if not isinstance(selector, str):
            continue
        for clause in selector.split("&"):
            name = clause.partition("=")[0].strip()
            if name:
                names.add(name)
    return names


def _number_mapping(value: Any, *, field_name: str) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise DurationSpecError(f"{field_name} must be an object.")
    return {str(key): float(item) for key, item in value.items()}


def _state_action_numbers(
    value: Any,
    *,
    field_name: str = "state_actions",
) -> dict[tuple[str, str], float]:
    if not isinstance(value, Mapping):
        raise DurationSpecError(f"{field_name} must be an object.")
    result: dict[tuple[str, str], float] = {}
    for state, actions in value.items():
        if not isinstance(actions, Mapping):
            raise DurationSpecError(f"{field_name}[{state!r}] must be an object.")
        for action, duration in actions.items():
            result[(str(state), str(action))] = float(duration)
    return result


def _state_action_gaussians(
    value: Any,
) -> tuple[dict[tuple[str, str], float], dict[tuple[str, str], float]]:
    if not isinstance(value, Mapping):
        raise DurationSpecError("state_actions must be an object.")
    means: dict[tuple[str, str], float] = {}
    variances: dict[tuple[str, str], float] = {}
    for state, actions in value.items():
        if not isinstance(actions, Mapping):
            raise DurationSpecError(f"state_actions[{state!r}] must be an object.")
        for action, entry in actions.items():
            if not isinstance(entry, Mapping):
                raise DurationSpecError(
                    f"state_actions[{state!r}][{action!r}] must contain mean and variance."
                )
            unknown = sorted(str(key) for key in set(entry) - {"mean", "variance"})
            if unknown:
                raise DurationSpecError(
                    f"Unknown Gaussian fields for {state!r}/{action!r}: "
                    + ", ".join(unknown)
                )
            missing = sorted({"mean", "variance"} - set(entry))
            if missing:
                raise DurationSpecError(
                    f"Missing Gaussian fields for {state!r}/{action!r}: "
                    + ", ".join(missing)
                )
            key = (str(state), str(action))
            means[key] = float(entry["mean"])
            variances[key] = float(entry["variance"])
    return means, variances


def _reject_json_constant(value: str) -> None:
    raise DurationSpecError(f"Duration sidecar JSON does not allow {value}.")
