"""Gurobi adapter for DARP binary ILP models."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from time import perf_counter
from typing import Any, Mapping

from darp.ilp.model import ILPLinearConstraint, ILPModelSpec, ILPSolveResult


class GurobiUnavailableError(RuntimeError):
    """Raised when gurobipy is not installed. / gurobipy 未安装时抛出。"""


@dataclass(frozen=True)
class GurobiILPSolver:
    """Solve DARP binary ILP models with Gurobi. / 使用 Gurobi 求解 DARP 二元 ILP 模型。"""

    output: bool = False

    def solve(
        self,
        spec: ILPModelSpec,
        *,
        time_limit_ms: float | None = None,
        mip_gap: float | None = None,
    ) -> ILPSolveResult:
        """Build and solve a Gurobi model from an ILPModelSpec. / 从 ILPModelSpec 构建并求解 Gurobi 模型。"""
        spec.validate()
        gp = _gurobipy()
        grb = gp.GRB
        started_at = perf_counter()
        model = gp.Model(spec.name)
        _set_param(model, "OutputFlag", 1 if self.output else 0)
        if time_limit_ms is not None:
            _set_param(model, "TimeLimit", max(0.0, float(time_limit_ms) / 1000.0))
        if mip_gap is not None:
            _set_param(model, "MIPGap", float(mip_gap))

        variables = {
            variable.var_id: model.addVar(vtype=grb.BINARY, name=_safe_name(variable.var_id))
            for variable in spec.variables
        }
        if hasattr(model, "update"):
            model.update()
        for constraint in spec.constraints:
            model.addConstr(
                _linear_expr(gp, variables, constraint),
                name=_safe_name(constraint.name),
            )
        objective = gp.LinExpr()
        for var_id, coeff in spec.objective.items():
            objective.addTerms(float(coeff), variables[var_id])
        model.setObjective(objective, grb.MAXIMIZE if spec.maximize else grb.MINIMIZE)
        model.optimize()

        values = {var_id: _variable_value(variable) for var_id, variable in variables.items()}
        selected = tuple(var_id for var_id, value in values.items() if value > 0.5)
        elapsed_ms = (perf_counter() - started_at) * 1000.0
        status = _status_name(grb, _optional_attr(model, "Status"))
        return ILPSolveResult(
            solver="gurobi",
            status=status,
            objective_value=_optional_float(_optional_attr(model, "ObjVal")),
            variable_values=values,
            selected_variables=selected,
            runtime_ms=elapsed_ms,
            mip_gap=_optional_float(_optional_attr(model, "MIPGap")),
            message=None if status == "optimal" else status,
        )


def gurobi_available() -> bool:
    """Return whether `gurobipy` can be imported. / 返回当前环境是否可导入 `gurobipy`。"""
    try:
        _gurobipy()
    except GurobiUnavailableError:
        return False
    return True


def _gurobipy() -> Any:
    """Import gurobipy lazily. / 惰性导入 gurobipy。"""
    try:
        return importlib.import_module("gurobipy")
    except ImportError as exc:
        raise GurobiUnavailableError("gurobipy is required for DARP Phase 8 ILP solving.") from exc


def _linear_expr(gp: Any, variables: Mapping[str, Any], constraint: ILPLinearConstraint) -> Any:
    """Convert one sparse constraint to a Gurobi expression. / 将稀疏约束转成 Gurobi 表达式。"""
    expr = gp.LinExpr()
    for var_id, coeff in constraint.coefficients.items():
        expr.addTerms(float(coeff), variables[var_id])
    if constraint.sense == "==":
        return expr == float(constraint.rhs)
    if constraint.sense == "<=":
        return expr <= float(constraint.rhs)
    if constraint.sense == ">=":
        return expr >= float(constraint.rhs)
    raise ValueError(f"Unsupported ILP constraint sense: {constraint.sense}")


def _set_param(model: Any, name: str, value: float | int) -> None:
    """Set a Gurobi parameter across real/fake APIs. / 兼容真实和 fake API 设置 Gurobi 参数。"""
    if hasattr(model, "Params") and hasattr(model.Params, name):
        setattr(model.Params, name, value)
        return
    if hasattr(model, "setParam"):
        model.setParam(name, value)


def _status_name(grb: Any, status: object) -> str:
    """Map Gurobi status code to a stable string. / 将 Gurobi status code 映射为稳定字符串。"""
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
    """Return a Gurobi-safe name. / 返回适合 Gurobi 的名称。"""
    return "".join(char if char.isalnum() or char == "_" else "_" for char in value)[:240]


def _optional_float(value: object) -> float | None:
    """Return float(value) or None. / 返回 float(value) 或 None。"""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_attr(obj: object, name: str) -> object | None:
    """Return an optional solver attribute without leaking solver errors. / 安全读取可选 solver 属性。"""
    try:
        return getattr(obj, name)
    except Exception:
        return None


def _variable_value(variable: object) -> float:
    """Return a solved binary variable value or zero if unavailable. / 返回变量解值，不可用时返回零。"""
    return _optional_float(_optional_attr(variable, "X")) or 0.0
