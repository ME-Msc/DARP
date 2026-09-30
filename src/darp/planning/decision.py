"""Planner result shared by full-ILP and HILP. / 两类规划器共享的求解结果。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from darp.planning.policy import ConditionalPolicy, json_ready


@dataclass(frozen=True)
class ActionDecision:
    """A selected root action and its conditional policy. / 选中的根动作及对应条件策略。"""

    action: Mapping[str, Any]
    label: str
    value: float
    complete: bool
    value_kind: str
    timing: Mapping[str, float]
    policy: ConditionalPolicy

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": json_ready(self.action),
            "label": self.label,
            "value": self.value,
            "timing": dict(self.timing),
            "complete": self.complete,
            "value_kind": self.value_kind,
            "policy": self.policy.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ActionDecision:
        """Restore a decision from :meth:`to_dict` output. / 从序列化结果恢复动作决策。"""
        action = value["action"]
        timing = value["timing"]
        policy = value["policy"]
        if not isinstance(action, Mapping) or not isinstance(timing, Mapping):
            raise TypeError("Decision action and timing must be objects.")
        if not isinstance(policy, Mapping):
            raise TypeError("Decision policy must be an object.")
        return cls(
            action=dict(action),
            label=str(value["label"]),
            value=float(value["value"]),
            complete=bool(value["complete"]),
            value_kind=str(value["value_kind"]),
            timing={str(name): float(item) for name, item in timing.items()},
            policy=ConditionalPolicy.from_dict(policy),
        )
