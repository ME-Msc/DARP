"""DARP view over pyRDDLGym grounded models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping

from darp.model.and_or_tree import ANDORSearchInterface, ActionChoice, ObservationScope

if TYPE_CHECKING:
    from pyRDDLGym.core.compiler.model import RDDLGroundedModel


class UnsupportedRDDLFeatureError(ValueError):
    """Raised when a grounded model uses unsupported features. / grounded model 使用暂不支持功能时抛出。"""


@dataclass(frozen=True)
class GroundedRDDLView:
    """Expose stable DARP accessors for a pyRDDLGym grounded model. / 为 pyRDDLGym grounded model 暴露稳定的 DARP 读取接口。"""

    grounded_model: "RDDLGroundedModel"

    @property
    def discount(self) -> float:
        """Return the reward discount factor. / 返回 reward discount factor。"""
        value = getattr(self.grounded_model, "discount", 1.0)
        return 1.0 if value is None else float(value)

    def action_fluents(self) -> tuple[str, ...]:
        """Return grounded action fluent names. / 返回 grounded action fluent 名称。"""
        return _sorted_keys(getattr(self.grounded_model, "action_fluents", None))

    def observation_fluents(self) -> tuple[str, ...]:
        """Return grounded observation fluent names. / 返回 grounded observation fluent 名称。"""
        return _sorted_keys(getattr(self.grounded_model, "observ_fluents", None))

    def observation_scope(self) -> ObservationScope:
        """Return observation scope for AND-OR histories. / 返回 AND-OR history 使用的 observation scope。"""
        observations = self.observation_fluents()
        if observations:
            return ObservationScope(mode="pomdp-observation")
        return ObservationScope(mode="mdp-state")

    def action_choices(self, runtime: Any) -> tuple[ActionChoice, ...]:
        """Return concrete action choices for the current search interface. / 返回当前搜索接口的具体 action choice。"""
        self.validate_supported()
        return tuple(
            ActionChoice(label=_action_label(action), assignment=dict(action))
            for action in runtime.action_candidates()
        )

    def build_and_or_interface(self, runtime: Any, risk: Any | None = None) -> ANDORSearchInterface:
        """Build the action/observation interface consumed by AND-OR search. / 构建 AND-OR 搜索消费的 action/observation 接口。"""
        from darp.adapter.exact import ExactRDDLKernel

        return ANDORSearchInterface.from_actions_and_observations(
            actions=self.action_choices(runtime),
            observation_scope=self.observation_scope(),
            exact_kernel=ExactRDDLKernel.from_grounded_model(self.grounded_model, risk=risk),
        )

    def validate_supported(self) -> None:
        """Raise a clear error for features outside the current interface. / 对当前接口外的功能抛出清晰错误。"""
        unsupported: list[str] = []
        action_ranges = getattr(self.grounded_model, "action_ranges", None)
        ranges = action_ranges if isinstance(action_ranges, Mapping) else {}
        non_bool_actions = [
            action
            for action in self.action_fluents()
            if str(ranges.get(action)) != "bool"
        ]
        if non_bool_actions:
            unsupported.append(
                "non-bool action fluents: " + ", ".join(non_bool_actions)
            )
        max_actions = getattr(self.grounded_model, "max_allowed_actions", 1)
        if isinstance(max_actions, int) and max_actions > 1:
            unsupported.append(
                "concurrent action combinations: "
                f"max_allowed_actions={max_actions}; current interface enumerates noop and one-active bool actions"
            )
        for attribute, label in (
            ("preconditions", "action preconditions"),
            ("invariants", "state invariants"),
            ("terminations", "termination conditions"),
        ):
            expressions = getattr(self.grounded_model, attribute, ()) or ()
            if expressions:
                unsupported.append(
                    f"{label}: {len(expressions)} expression(s); current exact search does not "
                    "evaluate them while generating actions/histories"
                )
        if abs(self.discount - 1.0) > 1e-12:
            unsupported.append(
                "discounted objective: "
                f"discount={self.discount}; policy-tree coefficients currently assume discount=1"
            )
        if getattr(self.grounded_model, "reward", None) is None:
            unsupported.append(
                "missing reward expression: pyRDDLGym grounded model did not expose reward"
            )
        if not isinstance(getattr(self.grounded_model, "cpfs", None), Mapping):
            unsupported.append(
                "missing CPF mapping: pyRDDLGym grounded model did not expose cpfs as a mapping"
            )
        if unsupported:
            raise UnsupportedRDDLFeatureError(
                "Unsupported RDDL features for current DARP search interface: "
                + "; ".join(unsupported)
            )

def _sorted_keys(value: object) -> tuple[str, ...]:
    """Return deterministic string keys from a pyRDDLGym mapping. / 从 pyRDDLGym mapping 返回确定性字符串键。"""
    if isinstance(value, Mapping):
        return tuple(sorted(str(key) for key in value))
    return ()


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
