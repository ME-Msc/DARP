"""Small public interface for HILP utility heuristics. / HILP 效用启发式的最小公开接口。"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from importlib import import_module
from math import isfinite
from numbers import Real
from typing import Any

from darp.adapter.kernel import StateKey


@dataclass(frozen=True, slots=True)
class HeuristicInput:
    """One state/action point passed to a user heuristic.

    ``non_fluents`` exposes model constants without coupling the planner to a
    particular domain.  A callback returns a utility-to-go value for DARP's
    maximization objective; a cost-to-go heuristic must therefore be negated.

    / 回调接收单个状态、动作和模型常量，返回从当前动作开始的剩余效用；
    核心采用最大化目标，因此剩余代价估计需要取负。
    """

    state: Mapping[str, Any]
    action_label: str
    action: Mapping[str, Any]
    non_fluents: Mapping[str, Any]


HeuristicFunction = Callable[[HeuristicInput], Real]


@dataclass(frozen=True, slots=True)
class UtilityHeuristic:
    r"""Describe an external state utility heuristic used at HILP frontiers.

    The planner applies the paper's history-probability weighting itself:

    ``h_qa = sum_s ordinary_mass_q[s] * value(s, a)``.

    The frontier action history is qa; its ordinary mass is the paper's
    :math:`\rho^*(q)\tilde b^*_q(s)` at preceding observation history q,
    with the continuing-event restriction retained in the mass.

    Set ``upper_bound`` only when the callback is an admissible upper bound for
    DARP's maximization objective.  The flag affects optimality certification,
    never the ILP solution itself.

    / frontier 是动作历史 qa；ordinary_mass_q 是此前观测历史 q 的
    rho*(q)·tilde b*_q，保留继续执行事件的权重；核心负责乘历史概率。
    upper_bound 表示调用者确认启发式是可采纳上界，
    仅影响最优性认证，不改变传给 ILP 的数值。
    """

    name: str
    evaluate: HeuristicFunction
    upper_bound: bool = False

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("A utility heuristic must have a non-empty name.")
        if not callable(self.evaluate):
            raise TypeError("UtilityHeuristic.evaluate must be callable.")


def load_utility_heuristic(spec: str) -> UtilityHeuristic:
    """Load ``module:attribute`` as a :class:`UtilityHeuristic`.

    A bare callable is accepted as a convenient, non-certifying heuristic.
    Exporting ``UtilityHeuristic`` explicitly is recommended because it records
    a stable name and whether the bound is admissible.

    / 从 module:attribute 加载启发式；普通函数可调用但不提供上界认证，
    显式 UtilityHeuristic 还记录名称和可采纳性声明。
    """

    module_name, separator, attribute = spec.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("Heuristic must use the form 'module:attribute'.")
    value = getattr(import_module(module_name), attribute)
    if isinstance(value, UtilityHeuristic):
        return value
    if callable(value):
        return UtilityHeuristic(name=spec, evaluate=value)
    raise TypeError(
        f"{spec!r} must resolve to UtilityHeuristic or a callable, "
        f"not {type(value).__name__}."
    )


def history_heuristic_coefficient(
    heuristic: UtilityHeuristic,
    *,
    state_mass: Mapping[StateKey, float],
    action_label: str,
    action: Mapping[str, Any],
    non_fluents: Mapping[str, Any],
) -> float:
    r"""Return the history-weighted heuristic coefficient for action qa.

    ``state_mass[s]`` already stores :math:`\rho^*(q)\tilde b^*_q(s)`
    with the continuing-event restriction, so no second probability scale or
    belief normalisation is applied here.

    / 返回候选动作 qa 的启发系数；state_mass[s] 已保存 rho*(q)·tilde b*_q(s)，
    保留继续执行事件的权重，不再次加权或归一化。
    """

    terms: list[float] = []
    for state_key, probability in state_mass.items():
        value = heuristic.evaluate(
            HeuristicInput(
                state=dict(state_key),
                action_label=action_label,
                action=action,
                non_fluents=non_fluents,
            )
        )
        terms.append(float(probability) * _finite_float(value))
    coefficient = sum(terms)
    if not isfinite(coefficient):
        raise ValueError(f"Heuristic {heuristic.name!r} returned a non-finite value.")
    return coefficient


def _finite_float(value: Real) -> float:
    """Validate and convert one user heuristic value. / 校验并转换用户返回的有限实数启发值。"""

    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("A utility heuristic must return a finite real number.")
    numeric = float(value)
    if not isfinite(numeric):
        raise ValueError("A utility heuristic must return a finite value.")
    return numeric


__all__ = [
    "HeuristicInput",
    "UtilityHeuristic",
    "history_heuristic_coefficient",
    "load_utility_heuristic",
]
