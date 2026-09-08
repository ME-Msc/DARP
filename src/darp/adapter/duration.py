"""Build the paper's duration models from one grounded RDDL expression."""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping
from math import isfinite
from typing import Any

from darp.adapter.kernel import RDDLKernel
from darp.model.duration import (
    ChanceConstrainedDurationModel,
    DurationEstimate,
    FixedDurationModel,
    GaussianDurationModel,
    HistoryDurationEvaluator,
    StateDependentDurationModel,
)


class DurationExpressionError(ValueError):
    """Raised when the RDDL duration clause is invalid or unsupported."""


def build_duration_evaluator(
    kernel: RDDLKernel,
    actions: Iterable[Any],
    *,
    horizon: float,
) -> HistoryDurationEvaluator:
    """Compile ``duration = D(s,a);`` without changing the planning algorithm."""
    expression = getattr(kernel.grounded_model, "duration", None)
    if expression is None:
        raise DurationExpressionError(
            "RDDL domain must define `duration = <expression>;`."
        )

    action_map = {str(choice.label): dict(choice.assignment) for choice in actions}
    if not action_map:
        raise DurationExpressionError("The RDDL problem exposes no supported actions.")
    features = _features(expression, kernel)
    moments = _memoized_moments(expression, kernel, action_map)
    threshold = getattr(
        kernel.grounded_model,
        "max_duration_shortfall_probability",
        None,
    )

    if threshold is None:
        if "state" not in features:
            fixed = {
                action: moments((), action).mean
                for action in action_map
            }
            model = FixedDurationModel(
                durations=fixed,
                default=next(iter(fixed.values())),
            )
        else:
            model = StateDependentDurationModel(
                duration=lambda state, action: moments(state, action).mean
            )
        return HistoryDurationEvaluator(model=model, horizon=horizon)

    zeta = float(threshold)
    if not isfinite(zeta) or not 0.0 <= zeta <= 1.0:
        raise DurationExpressionError(
            "max-duration-shortfall-probability must be in [0, 1]."
        )
    if "normal" in features:
        model = GaussianDurationModel(moments=moments)
    else:
        model = ChanceConstrainedDurationModel(
            duration=lambda state, action: moments(state, action).mean
        )
    return HistoryDurationEvaluator(model=model, horizon=horizon, zeta=zeta)


def _memoized_moments(
    expression: Any,
    kernel: RDDLKernel,
    actions: Mapping[str, Mapping[str, Any]],
):
    cache: dict[tuple[Hashable, str], DurationEstimate] = {}

    def evaluate(state: Hashable, action: str) -> DurationEstimate:
        key = (state, action)
        cached = cache.get(key)
        if cached is not None:
            return cached
        try:
            assignment = actions[action]
        except KeyError as error:
            raise DurationExpressionError(
                f"Unknown duration action: {action!r}."
            ) from error
        state_mapping = _state_mapping(state)
        result = _moments(expression, kernel, state_mapping, assignment)
        cache[key] = result
        return result

    return evaluate


def _moments(
    expression: Any,
    kernel: RDDLKernel,
    state: Mapping[str, Any],
    action: Mapping[str, Any],
) -> DurationEstimate:
    if not _is_expression(expression):
        return DurationEstimate(mean=_number(expression, "duration"))
    expression_type, operator = expression.etype
    if expression_type == "control" and operator == "if":
        condition, true_value, false_value = expression.args
        selected = (
            true_value
            if bool(kernel.deterministic_value(condition, state, action))
            else false_value
        )
        return _moments(selected, kernel, state, action)
    if expression_type == "randomvar":
        if operator != "Normal":
            raise DurationExpressionError(
                f"DARP stochastic duration supports Normal, not {operator}."
            )
        mean_expression, variance_expression = expression.args
        variance = _number(
            kernel.deterministic_value(variance_expression, state, action),
            "Normal duration variance",
        )
        if variance < 0.0:
            raise DurationExpressionError(
                "Normal duration variance must be non-negative."
            )
        return DurationEstimate(
            mean=_number(
                kernel.deterministic_value(mean_expression, state, action),
                "Normal duration mean",
            ),
            variance=variance,
        )
    if _has_random(expression):
        raise DurationExpressionError(
            "A stochastic duration must place Normal directly in an if-branch."
        )
    return DurationEstimate(
        mean=_number(
            kernel.deterministic_value(expression, state, action),
            "duration",
        )
    )


def _features(expression: Any, kernel: RDDLKernel) -> set[str]:
    """Return active state/action/Normal dependencies after non-fluent branching."""
    if not _is_expression(expression):
        return set()
    expression_type, operator = expression.etype
    if expression_type == "pvar":
        name, parameters = expression.args
        if parameters not in (None, []):
            raise DurationExpressionError(
                f"Duration expression is not grounded: {name}."
            )
        if name in kernel.state_names:
            return {"state"}
        if name in kernel.action_names:
            return {"action"}
        if name in kernel.non_fluents:
            return set()
        if name in kernel.intermediate_names:
            return _features(kernel.cpf_expression(name), kernel)
        if name in kernel.cpfs:
            raise DurationExpressionError(
                f"Duration D(s,a) cannot reference non-intermediate CPF {name!r}."
            )
        raise DurationExpressionError(
            f"Duration D(s,a) cannot reference {name!r}; use state, action, or non-fluents."
        )
    if expression_type == "control" and operator == "if":
        condition, true_value, false_value = expression.args
        condition_features = _features(condition, kernel)
        if not condition_features:
            selected = (
                true_value
                if bool(kernel.deterministic_value(condition, {}, {}))
                else false_value
            )
            return _features(selected, kernel)
        result = set(condition_features)
        result.update(_features(true_value, kernel))
        result.update(_features(false_value, kernel))
        return result
    result = {"normal"} if expression_type == "randomvar" and operator == "Normal" else set()
    for argument in _arguments(expression.args):
        result.update(_features(argument, kernel))
    if expression_type == "randomvar" and operator != "Normal":
        raise DurationExpressionError(
            f"Duration distributions currently support Normal, not {operator}."
        )
    return result


def _has_random(expression: Any) -> bool:
    if not _is_expression(expression):
        return False
    if expression.etype[0] == "randomvar":
        return True
    return any(_has_random(argument) for argument in _arguments(expression.args))


def _arguments(value: Any) -> tuple[Any, ...]:
    if isinstance(value, (tuple, list)):
        return tuple(value)
    return (value,)


def _is_expression(value: Any) -> bool:
    return hasattr(value, "etype") and hasattr(value, "args")


def _state_mapping(state: Hashable) -> Mapping[str, Any]:
    if state == ():
        return {}
    if isinstance(state, Mapping):
        return state
    if isinstance(state, tuple):
        try:
            return dict(state)
        except (TypeError, ValueError):
            pass
    raise DurationExpressionError(f"Unsupported duration state key: {state!r}.")


def _number(value: Any, label: str) -> float:
    """Convert one RDDL numeric value without accepting Boolean durations."""
    if isinstance(value, bool):
        raise DurationExpressionError(f"{label} must be numeric, not Boolean.")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise DurationExpressionError(
            f"{label} must be numeric, got {value!r}."
        ) from error
    if not isfinite(number):
        raise DurationExpressionError(f"{label} must be finite, got {value!r}.")
    return number


__all__ = ["DurationExpressionError", "build_duration_evaluator"]
