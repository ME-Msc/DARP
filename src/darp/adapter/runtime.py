"""Minimal pyRDDLGym runtime facade required by Algorithm 1."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass
class PyRDDLGymRuntime:
    env: Any

    @property
    def horizon(self) -> int:
        return int(getattr(self.env, "horizon", 0) or 0)

    @property
    def state(self) -> Mapping[str, Any]:
        return dict(getattr(self.env, "state", {}) or {})

    def reset(self, seed: int | None = None) -> None:
        self.env.reset(seed=seed)

    def action_candidates(self) -> tuple[dict[str, Any], ...]:
        """Enumerate the supported noop and one-active Boolean actions."""

        base = dict(getattr(self.env, "_noop_actions", None) or {})
        if not base:
            model = getattr(self.env, "model", None)
            base = dict(getattr(model, "action_fluents", {}) or {})
        candidates = [base]
        for name, range_name in (getattr(self.env, "_action_ranges", {}) or {}).items():
            if str(range_name) == "bool":
                candidates.append({**base, str(name): True})
        return tuple(candidates)
