"""Shared replay-frame schema for experiment visualizations."""

# TODO(visualization): Add typed event fields when replay supports belief/risk overlays.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ReplayFrame:
    """One visual replay frame derived from simulator state and the next action."""

    step: int
    agent: str
    obstacles: tuple[str, ...] = ()
    action: str = ""
    reward: str = ""
    next_agent: str = ""
    raw_state: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return a browser-friendly JSON object."""
        return {
            "step": self.step,
            "agent": self.agent,
            "obstacles": list(self.obstacles),
            "action": self.action,
            "reward": self.reward,
            "next_agent": self.next_agent,
            "raw_state": self.raw_state,
        }


def frame_from_mapping(data: dict[str, Any]) -> ReplayFrame:
    """Parse one JSON frame while tolerating missing optional fields."""
    return ReplayFrame(
        step=int(data.get("step", 0) or 0),
        agent=str(data.get("agent", "") or ""),
        obstacles=tuple(str(item) for item in data.get("obstacles", []) if item),
        action=str(data.get("action", "") or ""),
        reward=str(data.get("reward", "") or ""),
        next_agent=str(data.get("next_agent", "") or ""),
        raw_state=str(data.get("raw_state", "") or ""),
    )
