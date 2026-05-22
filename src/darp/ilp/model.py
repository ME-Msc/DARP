"""Small binary ILP model schema used before calling Gurobi."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Mapping

ConstraintSense = Literal["==", "<=", ">="]


@dataclass(frozen=True)
class ILPVariable:
    """Describe one binary policy variable. / 描述一个二元 policy 变量。"""

    var_id: str
    label: str
    metadata: Mapping[str, object] = field(default_factory=dict)


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
    maximize: bool = True

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

    solver: str
    status: str
    objective_value: float | None
    variable_values: Mapping[str, float]
    selected_variables: tuple[str, ...]
    runtime_ms: float
    mip_gap: float | None = None
    message: str | None = None

    @property
    def is_optimal(self) -> bool:
        """Return whether Gurobi reported an optimal solution. / 返回 Gurobi 是否报告最优解。"""
        return self.status == "optimal"
