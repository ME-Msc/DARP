"""Gurobi adapter for DARP binary ILP models."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import importlib
from math import frexp, fsum, isfinite, ldexp
from time import perf_counter
from typing import Any, Mapping

from darp.ilp.model import ILPLinearConstraint, ILPModelSpec, ILPSolveResult


class GurobiUnavailableError(RuntimeError):
    """Raised when gurobipy is not installed. / gurobipy 未安装时抛出。"""


@dataclass(frozen=True)
class _Solve:
    """One solve after exact validation of its binary incumbent."""

    status: str
    has_incumbent: bool
    values: Mapping[str, float]
    objective_value: float | None
    objective_bound: float | None
    mip_gap: float | None


class GurobiILPSolver:
    """Solve DARP binary ILP models with one standard Gurobi objective."""

    def solve(
        self,
        spec: ILPModelSpec,
        *,
        time_limit_ms: float | None = None,
        warm_start: Mapping[str, float] | None = None,
    ) -> ILPSolveResult:
        """Build and solve a Gurobi model from an ``ILPModelSpec``."""
        spec.validate()
        if time_limit_ms is not None and (
            not isfinite(float(time_limit_ms)) or float(time_limit_ms) < 0.0
        ):
            raise ValueError("time_limit_ms must be finite and non-negative when provided.")
        objective, objective_scale = _scaled_objective(spec.objective)
        gp = _gurobipy()
        grb = gp.GRB
        started_at = perf_counter()
        deadline = (
            started_at + float(time_limit_ms) / 1000.0
            if time_limit_ms is not None
            else None
        )

        model = gp.Model(spec.name)
        _set_param(model, "OutputFlag", 0)
        _set_param(model, "Threads", 1)
        # Disable both independent early-stopping tolerances.  This is a
        # numerical zero-gap request, not a claim of exact objective arithmetic.
        _set_param(model, "MIPGap", 0.0)
        _set_param(model, "MIPGapAbs", 0.0)

        variables = {
            variable.var_id: model.addVar(
                vtype=grb.BINARY,
                name=_safe_name(variable.var_id),
            )
            for variable in spec.variables
        }
        if warm_start:
            for var_id, value in warm_start.items():
                if var_id in variables:
                    _set_start(variables[var_id], float(value))
        if hasattr(model, "update"):
            model.update()
        for constraint in spec.constraints:
            model.addConstr(
                _linear_expr(gp, variables, _scaled_for_solver(constraint)),
                name=_safe_name(constraint.name),
            )

        expression = gp.LinExpr()
        for var_id, coefficient in objective.items():
            expression.addTerms(coefficient, variables[var_id])
        model.setObjective(
            expression,
            grb.MAXIMIZE,
        )
        solved = _solve_valid_incumbent(
            gp,
            grb,
            model,
            spec,
            variables,
            deadline=deadline,
        )

        values = dict(solved.values)
        selected = tuple(
            var_id for var_id, value in values.items() if value > 0.5
        )
        objective_value = (
            _original_objective_value(spec, values)
            if solved.has_incumbent
            else None
        )
        objective_bound = _original_objective_bound(
            solved.objective_bound,
            scale=objective_scale,
            solver_value=solved.objective_value,
            original_value=objective_value,
        )
        return ILPSolveResult(
            status=solved.status,
            objective_value=objective_value,
            variable_values=values,
            selected_variables=selected,
            runtime_ms=(perf_counter() - started_at) * 1000.0,
            mip_gap=solved.mip_gap,
            objective_bound=objective_bound,
        )


def _gurobipy() -> Any:
    """Import gurobipy lazily."""
    try:
        return importlib.import_module("gurobipy")
    except ImportError as exc:
        raise GurobiUnavailableError(
            "gurobipy is required for DARP Phase 8 ILP solving."
        ) from exc


def _scaled_objective(
    objective: Mapping[str, float],
) -> tuple[dict[str, float], float]:
    """Apply one order-preserving power-of-two scale to the objective.

    Keeping the largest coefficient near one avoids losing a small objective
    to Gurobi's absolute optimality tolerance.  Scaling is used only when
    every multiplication is exact in binary64; an extreme dynamic range can
    otherwise round distinct subnormal coefficients to the same value.
    """
    coefficients = {var_id: float(value) for var_id, value in objective.items()}
    for var_id, value in coefficients.items():
        if not isfinite(value):
            raise ValueError(f"Objective coefficient for {var_id!r} must be finite.")
    largest = max((abs(value) for value in coefficients.values()), default=0.0)
    if largest == 0.0:
        return coefficients, 1.0
    _, exponent = frexp(largest)
    try:
        scale = ldexp(1.0, 1 - exponent)
    except OverflowError:
        return coefficients, 1.0
    if not isfinite(scale) or scale == 0.0:
        return coefficients, 1.0
    scaled = {var_id: value * scale for var_id, value in coefficients.items()}
    if any(
        not isfinite(value) or (coefficients[var_id] != 0.0 and value == 0.0)
        for var_id, value in scaled.items()
    ):
        return coefficients, 1.0
    scale_exact = Fraction.from_float(scale)
    if any(
        Fraction.from_float(scaled[var_id])
        != Fraction.from_float(value) * scale_exact
        for var_id, value in coefficients.items()
    ):
        return coefficients, 1.0
    return scaled, scale


def _solve_valid_incumbent(
    gp: Any,
    grb: Any,
    model: Any,
    spec: ILPModelSpec,
    variables: Mapping[str, Any],
    *,
    deadline: float | None,
) -> _Solve:
    """Optimize, excluding every binary incumbent that fails an exact row check."""
    rejected = 0
    zeros = {var_id: 0.0 for var_id in variables}
    while True:
        if deadline is not None:
            remaining = deadline - perf_counter()
            if remaining <= 0.0:
                return _Solve(
                    status="time_limit",
                    has_incumbent=False,
                    values=zeros,
                    objective_value=None,
                    objective_bound=None,
                    mip_gap=None,
                )
            # Re-optimization after a no-good cut shares the original deadline.
            _set_param(model, "TimeLimit", remaining)
        model.optimize()
        status = _status_name(grb, _optional_attr(model, "Status"))
        solution_count = _optional_float(_optional_attr(model, "SolCount"))
        has_incumbent = status not in {
            "infeasible",
            "infeasible_or_unbounded",
            "unbounded",
        } and (solution_count is None or solution_count > 0.0)
        if not has_incumbent:
            return _Solve(
                status=status,
                has_incumbent=False,
                values=zeros,
                objective_value=None,
                objective_bound=None,
                mip_gap=None,
            )
        values = {
            var_id: 1.0 if _variable_value(variable) > 0.5 else 0.0
            for var_id, variable in variables.items()
        }
        if not _strictly_violated_constraints(spec, values):
            return _Solve(
                status=status,
                has_incumbent=True,
                values=values,
                objective_value=_optional_float(_optional_attr(model, "ObjVal")),
                objective_bound=_optional_float(_optional_attr(model, "ObjBound")),
                mip_gap=_optional_float(_optional_attr(model, "MIPGap")),
            )
        rejected += 1
        model.addConstr(
            _no_good_cut(gp, variables, values),
            name=f"darp_exact_feasibility_cut_{rejected}",
        )


def _original_objective_value(
    spec: ILPModelSpec,
    values: Mapping[str, float],
) -> float:
    """Recompute an incumbent in the caller's original objective units."""
    return fsum(
        float(coefficient) * float(values.get(var_id, 0.0))
        for var_id, coefficient in spec.objective.items()
    )


def _original_objective_bound(
    solver_bound: float | None,
    *,
    scale: float,
    solver_value: float | None,
    original_value: float | None,
) -> float | None:
    """Convert the numerical Gurobi bound to original objective units."""
    if solver_bound is None:
        return None
    if original_value is not None and solver_value == solver_bound:
        return original_value
    converted = solver_bound / scale
    return converted if isfinite(converted) else None


def _linear_expr(
    gp: Any,
    variables: Mapping[str, Any],
    constraint: ILPLinearConstraint,
) -> Any:
    """Convert one sparse constraint to a Gurobi expression."""
    expr = gp.LinExpr()
    for var_id, coefficient in constraint.coefficients.items():
        expr.addTerms(float(coefficient), variables[var_id])
    if constraint.sense == "==":
        return expr == float(constraint.rhs)
    if constraint.sense == "<=":
        return expr <= float(constraint.rhs)
    if constraint.sense == ">=":
        return expr >= float(constraint.rhs)
    raise ValueError(f"Unsupported ILP constraint sense: {constraint.sense}")


def _scaled_for_solver(constraint: ILPLinearConstraint) -> ILPLinearConstraint:
    """Scale a uniformly tiny row up before handing it to Gurobi."""
    magnitude = max(
        (abs(float(value)) for value in constraint.coefficients.values()),
        default=0.0,
    )
    magnitude = max(magnitude, abs(float(constraint.rhs)))
    if magnitude <= 0.0 or magnitude >= 1.0:
        return constraint
    factor = 1.0 / magnitude
    if not isfinite(factor):
        return constraint
    return ILPLinearConstraint(
        name=constraint.name,
        coefficients={
            var_id: float(coefficient) * factor
            for var_id, coefficient in constraint.coefficients.items()
        },
        sense=constraint.sense,
        rhs=float(constraint.rhs) * factor,
    )


def _strictly_violated_constraints(
    spec: ILPModelSpec,
    assignment: Mapping[str, float],
) -> tuple[str, ...]:
    """Return rows violated by a binary incumbent in exact rational arithmetic."""
    violated: list[str] = []
    for constraint in spec.constraints:
        lhs = sum(
            (
                Fraction.from_float(float(coefficient))
                * Fraction.from_float(float(assignment.get(var_id, 0.0)))
                for var_id, coefficient in constraint.coefficients.items()
            ),
            start=Fraction(0),
        )
        rhs = Fraction.from_float(float(constraint.rhs))
        if constraint.sense == "==":
            satisfied = lhs == rhs
        elif constraint.sense == "<=":
            satisfied = lhs <= rhs
        elif constraint.sense == ">=":
            satisfied = lhs >= rhs
        else:
            raise ValueError(f"Unsupported ILP constraint sense: {constraint.sense}")
        if not satisfied:
            violated.append(constraint.name)
    return tuple(violated)


def _no_good_cut(
    gp: Any,
    variables: Mapping[str, Any],
    assignment: Mapping[str, float],
) -> Any:
    """Exclude exactly one binary assignment rejected by strict validation."""
    expr = gp.LinExpr()
    selected_count = 0
    for var_id, variable in variables.items():
        if float(assignment.get(var_id, 0.0)) > 0.5:
            expr.addTerms(-1.0, variable)
            selected_count += 1
        else:
            expr.addTerms(1.0, variable)
    return expr >= 1.0 - float(selected_count)


def _set_param(model: Any, name: str, value: float | int) -> None:
    """Set a Gurobi parameter across real and test-double APIs."""
    if hasattr(model, "Params") and hasattr(model.Params, name):
        setattr(model.Params, name, value)
        return
    if hasattr(model, "setParam"):
        model.setParam(name, value)


def _set_start(variable: Any, value: float) -> None:
    """Set a binary MIP start when the adapter supports it."""
    try:
        setattr(variable, "Start", 1.0 if value > 0.5 else 0.0)
    except Exception:
        pass


def _status_name(grb: Any, status: object) -> str:
    """Map a Gurobi status code to a stable string."""
    names = {
        getattr(grb, "OPTIMAL", None): "optimal",
        getattr(grb, "INFEASIBLE", None): "infeasible",
        getattr(grb, "INF_OR_UNBD", None): "infeasible_or_unbounded",
        getattr(grb, "UNBOUNDED", None): "unbounded",
        getattr(grb, "TIME_LIMIT", None): "time_limit",
        getattr(grb, "INTERRUPTED", None): "interrupted",
    }
    return names.get(status, f"status_{status}")


def _safe_name(value: str) -> str:
    """Return a Gurobi-safe name."""
    return "".join(
        char if char.isalnum() or char == "_" else "_" for char in value
    )[:240]


def _optional_float(value: object) -> float | None:
    """Return a finite ``float(value)`` or ``None``."""
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if isfinite(numeric) else None


def _optional_attr(obj: object, name: str) -> object | None:
    """Read an optional solver attribute without leaking solver errors."""
    try:
        return getattr(obj, name)
    except Exception:
        return None


def _variable_value(variable: object) -> float:
    """Return a solved binary variable value, or zero when unavailable."""
    return _optional_float(_optional_attr(variable, "X")) or 0.0
