"""Gurobi adapter for DARP binary ILP models. / DARP 二元 ILP 的 Gurobi 适配层。"""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from math import isfinite
from time import perf_counter
from typing import Any, Self

from darp.ilp.model import ILPLinearConstraint, ILPModelSpec, ILPSolveResult

DEFAULT_MIP_GAP = 1e-4


class GurobiUnavailableError(RuntimeError):
    """Raised when gurobipy is not installed. / gurobipy 未安装时抛出。"""


class GurobiILPSession:
    """Incrementally solve a monotone sequence of binary ILP specifications.

    HILP grows one partial policy tree over several refinements. Variables and
    structural rows are retained in one Gurobi model; a refinement adds child
    variables/flow rows and updates the objective and global budget row. The
    latest complete ``ILPModelSpec`` remains the source of truth for model
    synchronization.

    / 增量求解一系列只增长的二元 ILP。HILP 的多轮 refinement 共用同一个
    Gurobi model：保留已有变量和结构约束，只加入 child/flow，并更新目标与
    全局预算行。
    """

    def __init__(self) -> None:
        self._gp: Any | None = None
        self._grb: Any | None = None
        self._model: Any | None = None
        self._model_name: str | None = None
        self._variables: dict[str, Any] = {}
        self._constraints: dict[str, Any] = {}
        self._solver_rows: dict[str, ILPLinearConstraint] = {}
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
            self._solver_rows.clear()
            self._solver_objective.clear()
            self._objective_initialized = False
            self._start_values.clear()

    def solve(
        self,
        spec: ILPModelSpec,
        *,
        time_limit_ms: float | None = None,
        warm_start: Mapping[str, float] | None = None,
    ) -> ILPSolveResult:
        """Apply the ``spec`` delta and re-optimize. / 同步 ``spec`` 差量并重新求解。"""
        spec.validate()
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
        self._synchronize(spec, warm_start=warm_start)
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
        values = (
            {
                var_id: _variable_value(self._variables[var_id])
                for var_id in spec.variable_ids()
            }
            if has_incumbent
            else {var_id: 0.0 for var_id in spec.variable_ids()}
        )
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

    def _synchronize(
        self,
        spec: ILPModelSpec,
        *,
        warm_start: Mapping[str, float] | None,
    ) -> None:
        """Apply a full spec as a model delta. / 将完整 spec 以差量方式同步到模型。"""
        self._ensure_model(spec.name)
        gp = self._gp
        grb = self._grb
        model = self._model
        if gp is None or grb is None or model is None:
            raise RuntimeError("Gurobi session failed to initialize its model.")

        # HILP 的 E∪F 只增长：已有变量直接复用，只对新 child 调用 addVar；
        # 若新 spec 删除旧变量，说明 refinement 结构不再等价，应立即报错。
        variable_ids = spec.variable_ids()
        if len(set(variable_ids)) != len(variable_ids):
            raise ValueError("ILP model contains duplicate variable ids.")
        removed_variables = set(self._variables) - set(variable_ids)
        if removed_variables:
            raise ValueError(
                "Incremental ILP specifications cannot remove variables: "
                + ", ".join(sorted(removed_variables))
            )
        for variable in spec.variables:
            if variable.var_id not in self._variables:
                self._variables[variable.var_id] = model.addVar(
                    vtype=grb.BINARY,
                    name=_safe_name(variable.var_id),
                )
        if hasattr(model, "update"):
            model.update()

        # 约束按稳定原始名称匹配：旧 root/flow 保留，新 flow 追加；全局
        # risk 行通过同一个 handle 原位更新。
        rows = _constraints_by_name(spec.constraints)
        removed_rows = set(self._constraints) - set(rows)
        if removed_rows:
            raise ValueError(
                "Incremental ILP specifications cannot remove constraints: "
                + ", ".join(sorted(removed_rows))
            )
        for name, row in rows.items():
            if name not in self._constraints:
                self._constraints[name] = model.addConstr(
                    _linear_expr(gp, self._variables, row),
                    name=_safe_name(name),
                )
            elif self._solver_rows[name] != row:
                self._update_row(name, row)
            self._solver_rows[name] = row

        # Frontier refinement changes h_q to u_q and appends child objective
        # coefficients. Keep the original floating-point units used by the
        # paper implementation and update only coefficients whose values
        # changed. / frontier 展开时原位更新 h_q→u_q，新 child
        # 直接追加；目标保持论文实现的原始浮点单位。
        objective = {
            var_id: float(coefficient)
            for var_id, coefficient in spec.objective.items()
        }
        non_finite = {
            var_id: coefficient
            for var_id, coefficient in objective.items()
            if not isfinite(coefficient)
        }
        if non_finite:
            raise ValueError(
                f"Objective coefficients must be finite: {non_finite!r}"
            )
        if not self._objective_initialized:
            expression = gp.LinExpr()
            for var_id, coefficient in objective.items():
                expression.addTerms(coefficient, self._variables[var_id])
            model.setObjective(expression, grb.MAXIMIZE)
            self._objective_initialized = True
        else:
            for var_id in set(self._solver_objective) | set(objective):
                previous = float(self._solver_objective.get(var_id, 0.0))
                current = float(objective.get(var_id, 0.0))
                if previous != current:
                    _set_objective_coefficient(self._variables[var_id], current)
        self._solver_objective = objective

        # 上一轮解只给已有变量提供 partial MIP start；不再提供的旧 Start 被
        # 清除，新 child 保持 UNDEFINED，让 Gurobi 根据新增 flow 自动补全。
        next_start = {
            var_id: 1.0 if float(value) > 0.5 else 0.0
            for var_id, value in (warm_start or {}).items()
            if var_id in self._variables
        }
        undefined = getattr(grb, "UNDEFINED", None)
        if undefined is not None:
            for var_id in set(self._start_values) - set(next_start):
                _clear_start(self._variables[var_id], undefined)
        for var_id, value in next_start.items():
            if self._start_values.get(var_id) != value:
                _set_start(self._variables[var_id], value)
        self._start_values = next_start
        if hasattr(model, "update"):
            model.update()

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
        # Match the reference implementation's standard Gurobi tolerance.
        # Threads and the absolute gap keep their Gurobi defaults.
        # 采用参考实现的标准 Gurobi 数值设置；线程数和绝对 gap
        # 使用 Gurobi 默认值。
        _set_param(self._model, "MIPGap", DEFAULT_MIP_GAP)

    def _update_row(self, name: str, row: ILPLinearConstraint) -> None:
        """Update one existing row in place. / 原位更新已存在的约束行。"""
        model = self._model
        if model is None:
            raise RuntimeError("Cannot update a row before creating the model.")
        previous = self._solver_rows[name]
        if previous.sense != row.sense:
            raise ValueError(
                f"Incremental ILP constraint {name!r} changed sense "
                f"from {previous.sense!r} to {row.sense!r}."
            )
        handle = self._constraints[name]
        for var_id in set(previous.coefficients) | set(row.coefficients):
            old = float(previous.coefficients.get(var_id, 0.0))
            new = float(row.coefficients.get(var_id, 0.0))
            if old != new:
                model.chgCoeff(handle, self._variables[var_id], new)
        if float(previous.rhs) != float(row.rhs):
            _set_constraint_rhs(handle, float(row.rhs))


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
    """Return an empty result when the wall budget expires before optimize()."""
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


def _set_constraint_rhs(constraint: Any, rhs: float) -> None:
    """Update an existing row RHS. / 兼容不同适配器地更新现有约束右端项。"""
    try:
        constraint.RHS = rhs
    except Exception as exc:
        if hasattr(constraint, "setAttr"):
            constraint.setAttr("RHS", rhs)
            return
        raise RuntimeError("Gurobi constraint adapter cannot update RHS.") from exc


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
