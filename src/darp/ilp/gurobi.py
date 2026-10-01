"""Gurobi adapter for DARP binary ILP models. / DARP 二元 ILP 的 Gurobi 适配层。"""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from math import isfinite
from time import perf_counter
from typing import Any, Self

from darp.ilp.model import (
    ILPLinearConstraint,
    ILPModelDelta,
    ILPModelSpec,
    ILPSolveResult,
)

DEFAULT_MIP_GAP = 1e-6


class GurobiUnavailableError(RuntimeError):
    """Raised when gurobipy is not installed. / gurobipy 未安装时抛出。"""


class GurobiILPSession:
    """Incrementally solve a monotone sequence of binary ILP specifications.

    HILP grows one partial policy tree over several refinements. Variables and
    structural rows are retained in one Gurobi model; a refinement adds child
    variables/flow rows and updates the objective and global budget row.
    Every update uses ``ILPModelDelta``. A fresh session can turn a complete
    specification into the initial delta; later solves require explicit deltas
    and never rediscover changes by scanning the complete specification.

    / 增量求解一系列只增长的二元 ILP。HILP 的多轮 refinement 共用同一个
    Gurobi model：保留已有变量和结构约束，只加入 child/flow，并更新目标与
    全局预算行。所有更新统一使用 ``ILPModelDelta``；新会话可将完整模型描述
    转为首次差量，后续求解必须显式提供差量，不再扫描完整描述来查找变化。
    """

    def __init__(self) -> None:
        self._gp: Any | None = None
        self._grb: Any | None = None
        self._model: Any | None = None
        self._model_name: str | None = None
        self._variables: dict[str, Any] = {}
        self._constraints: dict[str, Any] = {}
        self._row_coefficients: dict[str, dict[str, float]] = {}
        self._solver_objective: dict[str, float] = {}
        self._objective_initialized = False
        self._start_values: dict[str, float] = {}
        self.last_model_update_ms = 0.0
        self.last_optimize_ms = 0.0

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def close(self) -> None:
        """Release the persistent model. / 结束会话并释放持久 Gurobi model。"""
        model = self._model
        self._model = None
        try:
            if model is not None and hasattr(model, "dispose"):
                model.dispose()
        finally:
            self._gp = None
            self._grb = None
            self._model_name = None
            self._variables.clear()
            self._constraints.clear()
            self._row_coefficients.clear()
            self._solver_objective.clear()
            self._objective_initialized = False
            self._start_values.clear()

    def solve(
        self,
        spec: ILPModelSpec,
        *,
        time_limit_ms: float | None = None,
        warm_start: Mapping[str, float] | None = None,
        delta: ILPModelDelta | None = None,
    ) -> ILPSolveResult:
        """Initialize from a full spec or apply an explicit delta, then optimize.

        / 首次可从完整描述初始化；后续应用显式差量，再调用求解器。
        """
        if delta is None:
            if self._model is not None:
                raise ValueError("An initialized ILP session requires an explicit delta.")
            delta = ILPModelDelta(
                variables=spec.variables,
                objective=spec.objective,
                constraints=spec.constraints,
            )
        if time_limit_ms is not None and (
            not isfinite(float(time_limit_ms)) or float(time_limit_ms) < 0.0
        ):
            raise ValueError("time_limit_ms must be finite and non-negative when provided.")

        started_at = perf_counter()
        deadline = (
            started_at + float(time_limit_ms) / 1000.0
            if time_limit_ms is not None
            else None
        )
        self._synchronize_delta(spec, delta, warm_start=warm_start)
        self.last_model_update_ms = (perf_counter() - started_at) * 1000.0

        grb = self._grb
        model = self._model
        if grb is None or model is None:
            raise RuntimeError("Gurobi session failed to initialize its model.")
        if deadline is not None:
            remaining = deadline - perf_counter()
            if remaining <= 0.0:
                self.last_optimize_ms = 0.0
                return _time_limit_result(
                    spec,
                    runtime_ms=(perf_counter() - started_at) * 1000.0,
                )
            _set_param(model, "TimeLimit", remaining)
        else:
            _set_param(model, "TimeLimit", getattr(grb, "INFINITY", 1e100))

        optimize_started_at = perf_counter()
        model.optimize()
        self.last_optimize_ms = (perf_counter() - optimize_started_at) * 1000.0

        status = _status_name(grb, _optional_attr(model, "Status"))
        solution_count = _optional_float(_optional_attr(model, "SolCount"))
        has_incumbent = status not in {
            "infeasible",
            "infeasible_or_unbounded",
            "unbounded",
        } and (solution_count is None or solution_count > 0.0)
        variable_ids = spec.variable_ids()
        if has_incumbent:
            handles = [self._variables[var_id] for var_id in variable_ids]
            raw_values = (
                model.getAttr("X", handles)
                if hasattr(model, "getAttr")
                else [_variable_value(variable) for variable in handles]
            )
            values = {
                var_id: _optional_float(value) or 0.0
                for var_id, value in zip(variable_ids, raw_values, strict=True)
            }
        else:
            values = dict.fromkeys(variable_ids, 0.0)
        selected = tuple(
            var_id for var_id, value in values.items() if value > 0.5
        )
        return ILPSolveResult(
            status=status,
            objective_value=(
                _optional_float(_optional_attr(model, "ObjVal"))
                if has_incumbent
                else None
            ),
            variable_values=values,
            selected_variables=selected,
            runtime_ms=(perf_counter() - started_at) * 1000.0,
        )

    def _synchronize_delta(
        self,
        spec: ILPModelSpec,
        delta: ILPModelDelta,
        *,
        warm_start: Mapping[str, float] | None,
    ) -> None:
        """Validate and apply only declared changes, without scanning old rows.

        / 只验证和同步本轮新增或改变的项，不扫描旧行。
        """
        new_ids: set[str] = set()
        for variable in delta.variables:
            if variable.var_id in self._variables or variable.var_id in new_ids:
                raise ValueError(f"ILP delta adds an existing variable: {variable.var_id!r}.")
            new_ids.add(variable.var_id)
        rows = _constraints_by_name(delta.constraints)
        if len(spec.variables) != len(self._variables) + len(new_ids):
            raise ValueError("ILP delta does not account for the full specification's variables.")
        if len(spec.constraints) != len(self._constraints) + len(rows):
            raise ValueError("ILP delta does not account for the full specification's constraints.")

        def validate_coefficients(coefficients: Mapping[str, float]) -> None:
            for var_id, coefficient in coefficients.items():
                if var_id not in self._variables and var_id not in new_ids:
                    raise ValueError(f"ILP delta references unknown variable: {var_id!r}.")
                if not isfinite(float(coefficient)):
                    raise ValueError(f"ILP coefficient must be finite: {var_id!r}={coefficient!r}.")

        validate_coefficients(delta.objective)
        for name, row in rows.items():
            if name in self._constraints:
                raise ValueError(f"ILP delta adds an existing constraint: {name!r}.")
            if row.sense not in {"==", "<=", ">="} or not isfinite(float(row.rhs)):
                raise ValueError(f"Invalid ILP delta constraint: {name!r}.")
            validate_coefficients(row.coefficients)
        for name, coefficients in delta.coefficients.items():
            if name not in self._constraints:
                raise ValueError(f"ILP delta updates unknown constraint: {name!r}.")
            validate_coefficients(coefficients)

        self._ensure_model(spec.name)
        gp, grb, model = self._gp, self._grb, self._model
        if gp is None or grb is None or model is None:
            raise RuntimeError("Gurobi session failed to initialize its model.")
        for variable in delta.variables:
            self._variables[variable.var_id] = model.addVar(
                vtype=grb.BINARY, name=_safe_name(variable.var_id)
            )
        if hasattr(model, "update"):
            model.update()
        for name, row in rows.items():
            self._constraints[name] = model.addConstr(
                _linear_expr(gp, self._variables, row), name=_safe_name(name)
            )
            self._row_coefficients[name] = dict(row.coefficients)
        for name, coefficients in delta.coefficients.items():
            # Private coefficient dictionaries allow updates and zero removals
            # without copying/scanning the whole row or mutating old snapshots.
            # 缓存持有独立的系数字典，只更新变化项或删除零值项；不复制、遍历
            # 整行，也不修改旧快照。
            cached = self._row_coefficients[name]
            for var_id, coefficient in coefficients.items():
                value = float(coefficient)
                if float(cached.get(var_id, 0.0)) != value:
                    model.chgCoeff(self._constraints[name], self._variables[var_id], value)
                if value:
                    cached[var_id] = value
                else:
                    cached.pop(var_id, None)
        if not self._objective_initialized:
            model.setObjective(gp.LinExpr(), grb.MAXIMIZE)
            self._objective_initialized = True
        for var_id, coefficient in delta.objective.items():
            value = float(coefficient)
            if float(self._solver_objective.get(var_id, 0.0)) != value:
                _set_objective_coefficient(self._variables[var_id], value)
            self._solver_objective[var_id] = value
        self._apply_warm_start(warm_start)
        if hasattr(model, "update"):
            model.update()

    def _apply_warm_start(self, warm_start: Mapping[str, float] | None) -> None:
        """Retain the original partial MIP-start semantics. / 保持原 partial MIP start 语义。"""
        # Seed existing variables only, clearing stale starts; leave new children
        # undefined so Gurobi can complete them under the added flow constraints.
        # 上一轮解只给已有变量提供 partial MIP start；不再提供的旧 Start 被
        # 清除，新 child 保持 UNDEFINED，让 Gurobi 根据新增 flow 自动补全。
        next_start = {
            var_id: 1.0 if float(value) > 0.5 else 0.0
            for var_id, value in (warm_start or {}).items()
            if var_id in self._variables
        }
        undefined = getattr(self._grb, "UNDEFINED", None)
        if undefined is not None:
            for var_id in set(self._start_values) - set(next_start):
                _clear_start(self._variables[var_id], undefined)
        for var_id, value in next_start.items():
            if self._start_values.get(var_id) != value:
                _set_start(self._variables[var_id], value)
        self._start_values = next_start

    def _ensure_model(self, name: str) -> None:
        """Create the model once per session. / 每个 session 只创建一个 model。"""
        if self._model is not None:
            if name != self._model_name:
                raise ValueError(
                    "One incremental Gurobi session cannot mix model names: "
                    f"{self._model_name!r} and {name!r}."
                )
            return
        self._gp = _gurobipy()
        self._grb = self._gp.GRB
        self._model_name = name
        self._model = self._gp.Model(name)
        _set_param(self._model, "OutputFlag", 0)
        # Use a tighter relative gap so reported two-decimal objectives remain
        # stable across Gurobi versions; threads and the absolute gap stay at
        # their defaults. / 收紧相对 gap 以稳定跨版本结果；线程数和绝对 gap
        # 仍使用 Gurobi 默认值。
        _set_param(self._model, "MIPGap", DEFAULT_MIP_GAP)

class GurobiILPSolver:
    """Solve one ILP in a fresh model. / 使用一次性新 model 求解一个 DARP ILP。"""

    def solve(
        self,
        spec: ILPModelSpec,
        *,
        time_limit_ms: float | None = None,
        warm_start: Mapping[str, float] | None = None,
    ) -> ILPSolveResult:
        """Build, solve, then release. / 创建、求解并释放一次性 Gurobi model。"""
        with GurobiILPSession() as session:
            return session.solve(
                spec,
                time_limit_ms=time_limit_ms,
                warm_start=warm_start,
            )


def _gurobipy() -> Any:
    """Import gurobipy lazily. / 延迟导入 gurobipy。"""
    try:
        return importlib.import_module("gurobipy")
    except ImportError as exc:
        raise GurobiUnavailableError(
            "gurobipy is required for DARP Phase 8 ILP solving."
        ) from exc


def _time_limit_result(
    spec: ILPModelSpec,
    *,
    runtime_ms: float,
) -> ILPSolveResult:
    """Return an empty result when the wall budget expires before optimize(). / optimize 前超时时返回空解。"""
    values = {var_id: 0.0 for var_id in spec.variable_ids()}
    return ILPSolveResult(
        status="time_limit",
        objective_value=None,
        variable_values=values,
        selected_variables=(),
        runtime_ms=runtime_ms,
    )


def _linear_expr(
    gp: Any,
    variables: Mapping[str, Any],
    constraint: ILPLinearConstraint,
) -> Any:
    """Convert a sparse row to Gurobi. / 将一条稀疏约束转换成 Gurobi 表达式。"""
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


def _constraints_by_name(
    constraints: tuple[ILPLinearConstraint, ...],
) -> dict[str, ILPLinearConstraint]:
    """Index rows by stable name. / 按稳定原始名称索引约束，并拒绝重名。"""
    indexed: dict[str, ILPLinearConstraint] = {}
    for constraint in constraints:
        if constraint.name in indexed:
            raise ValueError(f"ILP model contains duplicate constraint name: {constraint.name!r}.")
        indexed[constraint.name] = constraint
    return indexed


def _set_param(model: Any, name: str, value: float) -> None:
    """Set a solver parameter. / 兼容真实 Gurobi 与测试替身地设置参数。"""
    if hasattr(model, "Params") and hasattr(model.Params, name):
        setattr(model.Params, name, value)
        return
    if hasattr(model, "setParam"):
        model.setParam(name, value)


def _set_start(variable: Any, value: float) -> None:
    """Set a binary MIP start. / 在适配器支持时设置二元 MIP 初始值。"""
    try:
        variable.Start = 1.0 if value > 0.5 else 0.0
    except Exception:
        pass


def _set_objective_coefficient(variable: Any, value: float) -> None:
    """Update one Obj coefficient. / 更新当前目标中一个二元变量的系数。"""
    try:
        variable.Obj = value
    except Exception as exc:
        if hasattr(variable, "setAttr"):
            variable.setAttr("Obj", value)
            return
        raise RuntimeError("Gurobi variable adapter cannot update objective.") from exc


def _clear_start(variable: Any, undefined: object) -> None:
    """Clear a stale MIP start. / 应用最新 partial start 前清除过期初始值。"""
    try:
        variable.Start = undefined
    except Exception:
        pass


def _status_name(grb: Any, status: object) -> str:
    """Normalize a Gurobi status. / 将 Gurobi 状态码映射为稳定字符串。"""
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
    """Return a Gurobi-safe name. / 返回符合 Gurobi 规则的名称。"""
    return "".join(
        char if char.isalnum() or char == "_" else "_" for char in value
    )[:240]


def _optional_float(value: object) -> float | None:
    """Return a finite float or None. / 返回有限浮点数，否则返回 ``None``。"""
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if isfinite(numeric) else None


def _optional_attr(obj: object, name: str) -> object | None:
    """Read an optional solver attribute. / 安全读取可选求解器属性。"""
    try:
        return getattr(obj, name)
    except Exception:
        return None


def _variable_value(variable: object) -> float:
    """Read a solved binary value. / 读取二元变量解；不可用时返回 0。"""
    return _optional_float(_optional_attr(variable, "X")) or 0.0
