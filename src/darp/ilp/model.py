"""Small binary ILP model schema used before calling Gurobi. / 提交 Gurobi 前的最小二元 ILP 数据结构。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
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
class ILPModelDelta:
    """Append variables/rows and set changed coefficients, without re-encoding old rows.

    ``variables`` and ``constraints`` contain only new entries. ``objective``
    and ``coefficients`` specify absolute replacement values, not increments.
    ``coefficients`` updates existing rows only; zero removes a term.

    / 追加变量和约束，设置变化的系数，不重新编码旧行。
    variables、constraints 仅包含新增项；objective 与 coefficients 中的值
    是新的绝对系数，不是增量。coefficients 只能更新已有约束，0 表示移除该项。
    """

    variables: tuple[ILPVariable, ...] = ()
    objective: Mapping[str, float] = field(default_factory=dict)
    constraints: tuple[ILPLinearConstraint, ...] = ()
    coefficients: Mapping[str, Mapping[str, float]] = field(default_factory=dict)


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
