"""DARP view over pyRDDLGym grounded models."""

# TODO(phase-9.1): Expose risk/cost fluent selectors once benchmark-scale
# constrained CC-POMDP rows are implemented.

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping

from darp.model.and_or_tree import ANDORSearchInterface, ActionChoice, ObservationScope

if TYPE_CHECKING:
    from pyRDDLGym.core.compiler.model import RDDLGroundedModel


@dataclass(frozen=True)
class UnsupportedRDDLFeature:
    """Describe one unsupported grounded RDDL feature. / 描述一个当前不支持的 grounded RDDL 功能。"""

    feature: str
    detail: str


class UnsupportedRDDLFeatureError(ValueError):
    """Raised when a grounded model uses unsupported features. / grounded model 使用暂不支持功能时抛出。"""

    def __init__(self, features: tuple[UnsupportedRDDLFeature, ...]) -> None:
        """Format unsupported features into one readable error. / 将不支持的功能格式化为一条可读错误。"""
        self.features = features
        message = "; ".join(f"{item.feature}: {item.detail}" for item in features)
        super().__init__(f"Unsupported RDDL features for current DARP search interface: {message}")


@dataclass(frozen=True)
class GroundedVariable:
    """Describe one grounded variable exposed by pyRDDLGym. / 描述 pyRDDLGym 暴露的一个 grounded variable。"""

    name: str
    range_name: str | None
    default: Any = None


@dataclass(frozen=True)
class GroundedCPF:
    """Describe one grounded CPF expression. / 描述一个 grounded CPF 表达式。"""

    fluent: str
    dependencies: tuple[Any, ...]
    expression: Any


@dataclass(frozen=True)
class GroundedRDDLView:
    """Expose stable DARP accessors for a pyRDDLGym grounded model. / 为 pyRDDLGym grounded model 暴露稳定的 DARP 读取接口。"""

    grounded_model: "RDDLGroundedModel"

    @property
    def domain_name(self) -> str | None:
        """Return the grounded domain name. / 返回 grounded domain 名称。"""
        return getattr(self.grounded_model, "domain_name", None)

    @property
    def instance_name(self) -> str | None:
        """Return the grounded instance name. / 返回 grounded instance 名称。"""
        return getattr(self.grounded_model, "instance_name", None)

    @property
    def horizon(self) -> int:
        """Return the fixed planning horizon. / 返回 fixed planning horizon。"""
        return int(getattr(self.grounded_model, "horizon", 0) or 0)

    @property
    def discount(self) -> float:
        """Return the reward discount factor. / 返回 reward discount factor。"""
        return float(getattr(self.grounded_model, "discount", 1.0) or 1.0)

    def state_fluents(self) -> tuple[str, ...]:
        """Return grounded state fluent names. / 返回 grounded state fluent 名称。"""
        return _sorted_keys(getattr(self.grounded_model, "state_fluents", None))

    def action_fluents(self) -> tuple[str, ...]:
        """Return grounded action fluent names. / 返回 grounded action fluent 名称。"""
        return _sorted_keys(getattr(self.grounded_model, "action_fluents", None))

    def observation_fluents(self) -> tuple[str, ...]:
        """Return grounded observation fluent names. / 返回 grounded observation fluent 名称。"""
        return _sorted_keys(getattr(self.grounded_model, "observ_fluents", None))

    def non_fluents(self) -> tuple[str, ...]:
        """Return grounded non-fluent names. / 返回 grounded non-fluent 名称。"""
        return _sorted_keys(getattr(self.grounded_model, "non_fluents", None))

    def cpfs(self) -> Mapping[str, Any]:
        """Return grounded CPF expressions keyed by fluent. / 返回按 fluent 索引的 grounded CPF 表达式。"""
        cpfs = getattr(self.grounded_model, "cpfs", None)
        return dict(cpfs) if isinstance(cpfs, Mapping) else {}

    def reward_expression(self) -> Any:
        """Return the pyRDDLGym reward expression. / 返回 pyRDDLGym reward 表达式。"""
        return getattr(self.grounded_model, "reward", None)

    def variable_ranges(self) -> Mapping[str, Any]:
        """Return grounded variable ranges. / 返回 grounded variable range。"""
        ranges = getattr(self.grounded_model, "variable_ranges", None)
        return dict(ranges) if isinstance(ranges, Mapping) else {}

    def action_variables(self) -> tuple[GroundedVariable, ...]:
        """Return grounded action variables with ranges and defaults. / 返回带 range 和默认值的 grounded action 变量。"""
        return _variables(
            getattr(self.grounded_model, "action_fluents", None),
            getattr(self.grounded_model, "action_ranges", None),
        )

    def state_variables(self) -> tuple[GroundedVariable, ...]:
        """Return grounded state variables with ranges and defaults. / 返回带 range 和默认值的 grounded state 变量。"""
        return _variables(
            getattr(self.grounded_model, "state_fluents", None),
            getattr(self.grounded_model, "state_ranges", None),
        )

    def observation_variables(self) -> tuple[GroundedVariable, ...]:
        """Return explicit POMDP observation variables. / 返回显式 POMDP observation 变量。"""
        return _variables(
            getattr(self.grounded_model, "observ_fluents", None),
            getattr(self.grounded_model, "observ_ranges", None),
        )

    def cpf_expressions(self) -> tuple[GroundedCPF, ...]:
        """Return grounded CPF expressions in deterministic order. / 按确定性顺序返回 grounded CPF 表达式。"""
        cpfs = self.cpfs()
        result: list[GroundedCPF] = []
        for fluent in sorted(cpfs):
            raw = cpfs[fluent]
            if isinstance(raw, tuple) and len(raw) == 2:
                dependencies, expression = raw
                deps = tuple(dependencies) if isinstance(dependencies, (list, tuple)) else (dependencies,)
            else:
                deps = ()
                expression = raw
            result.append(GroundedCPF(fluent=fluent, dependencies=deps, expression=expression))
        return tuple(result)

    def observation_scope(self) -> ObservationScope:
        """Return observation scope for AND-OR histories. / 返回 AND-OR history 使用的 observation scope。"""
        observations = self.observation_fluents()
        if observations:
            return ObservationScope(mode="pomdp-observation", variables=observations)
        return ObservationScope(mode="mdp-state", variables=self.state_fluents())

    def action_choices(self, runtime: Any | None = None) -> tuple[ActionChoice, ...]:
        """Return concrete action choices for the current search interface. / 返回当前搜索接口的具体 action choice。"""
        self.validate_supported()
        raw_actions = runtime.action_candidates() if runtime is not None else self._default_action_candidates()
        return tuple(
            ActionChoice(label=_action_label(action), assignment=dict(action))
            for action in raw_actions
        )

    def build_and_or_interface(self, runtime: Any | None = None) -> ANDORSearchInterface:
        """Build the action/observation interface consumed by AND-OR search. / 构建 AND-OR 搜索消费的 action/observation 接口。"""
        return ANDORSearchInterface.from_actions_and_observations(
            actions=self.action_choices(runtime),
            observation_scope=self.observation_scope(),
        )

    def validate_supported(self) -> None:
        """Raise a clear error for features outside the current interface. / 对当前接口外的功能抛出清晰错误。"""
        unsupported: list[UnsupportedRDDLFeature] = []
        non_bool_actions = [
            variable.name
            for variable in self.action_variables()
            if variable.range_name != "bool"
        ]
        if non_bool_actions:
            unsupported.append(
                UnsupportedRDDLFeature(
                    feature="non-bool action fluents",
                    detail=", ".join(non_bool_actions),
                )
            )
        max_actions = getattr(self.grounded_model, "max_allowed_actions", 1)
        if isinstance(max_actions, int) and max_actions > 1:
            unsupported.append(
                UnsupportedRDDLFeature(
                    feature="concurrent action combinations",
                    detail=f"max_allowed_actions={max_actions}; current interface enumerates noop and one-active bool actions",
                )
            )
        if self.reward_expression() is None:
            unsupported.append(
                UnsupportedRDDLFeature(
                    feature="missing reward expression",
                    detail="pyRDDLGym grounded model did not expose reward",
                )
            )
        if not isinstance(getattr(self.grounded_model, "cpfs", None), Mapping):
            unsupported.append(
                UnsupportedRDDLFeature(
                    feature="missing CPF mapping",
                    detail="pyRDDLGym grounded model did not expose cpfs as a mapping",
                )
            )
        if unsupported:
            raise UnsupportedRDDLFeatureError(tuple(unsupported))

    def _default_action_candidates(self) -> tuple[dict[str, Any], ...]:
        """Return noop plus single-active bool action assignments. / 返回 noop 和单个 bool action 赋值。"""
        base = {variable.name: variable.default for variable in self.action_variables()}
        candidates = [base]
        for variable in self.action_variables():
            if variable.range_name == "bool":
                candidate = dict(base)
                candidate[variable.name] = True
                candidates.append(candidate)
        return tuple(candidates)

    def to_summary_dict(self) -> dict[str, Any]:
        """Return a compact JSON-friendly model view summary. / 返回紧凑的 JSON 友好 model view 摘要。"""
        return {
            "domain_name": self.domain_name,
            "instance_name": self.instance_name,
            "horizon": self.horizon,
            "discount": self.discount,
            "state_fluents": list(self.state_fluents()),
            "action_fluents": list(self.action_fluents()),
            "observation_fluents": list(self.observation_fluents()),
            "observation_mode": self.observation_scope().mode,
            "non_fluent_count": len(self.non_fluents()),
            "cpf_count": len(self.cpfs()),
        }


def _sorted_keys(value: object) -> tuple[str, ...]:
    """Return deterministic string keys from a pyRDDLGym mapping. / 从 pyRDDLGym mapping 返回确定性字符串键。"""
    if isinstance(value, Mapping):
        return tuple(sorted(str(key) for key in value))
    return ()


def _variables(values: object, ranges: object) -> tuple[GroundedVariable, ...]:
    """Return grounded variable descriptors from pyRDDLGym mappings. / 从 pyRDDLGym mapping 返回 grounded variable 描述。"""
    value_map = values if isinstance(values, Mapping) else {}
    range_map = ranges if isinstance(ranges, Mapping) else {}
    return tuple(
        GroundedVariable(
            name=str(name),
            range_name=str(range_map[name]) if name in range_map else None,
            default=default,
        )
        for name, default in sorted(value_map.items(), key=lambda item: str(item[0]))
    )


def _action_label(action: Mapping[str, Any]) -> str:
    """Return a compact deterministic action label. / 返回紧凑且确定性的 action 标签。"""
    active: list[str] = []
    for name, value in sorted(action.items(), key=lambda item: str(item[0])):
        python_value = _plain_value(value)
        if python_value is True:
            active.append(str(name))
        elif python_value not in (False, 0, None):
            active.append(f"{name}={python_value}")
    return "+".join(active) if active else "noop"


def _plain_value(value: Any) -> Any:
    """Convert numpy scalar values into plain Python values. / 将 numpy scalar 转成普通 Python 值。"""
    if hasattr(value, "item"):
        return value.item()
    return value
