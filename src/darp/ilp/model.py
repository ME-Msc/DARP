"""Small binary ILP model schema used before calling Gurobi."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from typing import Literal

ConstraintSense = Literal["==", "<=", ">="]


@dataclass(frozen=True)
class ILPVariable:
    """Describe one binary policy variable. / 描述一个二元 policy 变量。"""

    var_id: str


@dataclass(frozen=True)
class ILPLinearConstraint:
    """Describe one sparse linear constraint. / 描述一个稀疏线性约束。"""

    name: str
    coefficients: Mapping[str, float]
    sense: ConstraintSense
    rhs: float


@dataclass(frozen=True)
class ILPModelSpec:
    """Describe a binary linear optimization model. / 描述一个二元线性优化模型。"""

    name: str
    variables: tuple[ILPVariable, ...]
    objective: Mapping[str, float]
    constraints: tuple[ILPLinearConstraint, ...]

    def variable_ids(self) -> tuple[str, ...]:
        """Return variable ids in declaration order. / 按声明顺序返回变量 id。"""
        return tuple(variable.var_id for variable in self.variables)

    def validate(self) -> None:
        """Validate that objective and constraints reference known variables. / 验证目标和约束只引用已知变量。"""
        known = set(self.variable_ids())
        unknown = set(self.objective) - known
        for constraint in self.constraints:
            unknown.update(set(constraint.coefficients) - known)
        if unknown:
            raise ValueError(f"ILP model references unknown variables: {', '.join(sorted(unknown))}")


@dataclass(frozen=True)
class ILPSolveResult:
    """Store a Gurobi solve result in a solver-neutral shape. / 以 solver-neutral 形式保存 Gurobi 求解结果。"""

    status: str
    objective_value: float | None
    variable_values: Mapping[str, float]
    selected_variables: tuple[str, ...]
    runtime_ms: float
    mip_gap: float | None = None
    objective_bound: float | None = None

    @property
    def numerically_optimal(self) -> bool:
        """Return whether Gurobi closed its floating-point objective gap."""
        return self.status == "optimal" and self.has_numerical_zero_gap

    @property
    def has_numerical_zero_gap(self) -> bool:
        """Return whether the reported floating-point incumbent and bound coincide."""
        absolute_gap = self.absolute_gap
        if absolute_gap is None or self.mip_gap is None:
            return False
        relative_gap = float(self.mip_gap)
        return (
            isfinite(relative_gap)
            and relative_gap == 0.0
            and isfinite(absolute_gap)
            and absolute_gap == 0.0
        )

    @property
    def absolute_gap(self) -> float | None:
        """Return the incumbent-to-bound display gap in original objective units."""
        if self.objective_value is None or self.objective_bound is None:
            return None
        return abs(float(self.objective_bound) - float(self.objective_value))
