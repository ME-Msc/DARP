"""Planner result shared by full-ILP and HILP."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from darp.planning.policy import ConditionalPolicy, json_ready


@dataclass(frozen=True)
class ActionDecision:
    """A selected root action and its conditional policy."""

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
