"""Baseline planners over pyRDDLGym runtime copies."""

# TODO(phase-5.1): Move RolloutPlanner behind a shared planner registry before
# adding AND-OR tree, full ILP, and HILP planners.

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Mapping

from darp.adapter.runtime import ActionDict, PyRDDLGymRuntime, _json_ready


@dataclass(frozen=True)
class ActionDecision:
    """Store one pyRDDLGym-runtime action choice. / 保存一次 pyRDDLGym runtime 动作选择。"""

    action: ActionDict
    label: str
    value: float
    action_values: Mapping[str, float]
    remaining_depth: int
    elapsed_ms: float
    time_budget_ms: float | None = None
    complete: bool = True
    timed_out: bool = False
    fallback_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly decision record. / 返回适合 JSON 的决策记录。"""
        return {
            "action": _json_ready(self.action),
            "label": self.label,
            "value": self.value,
            "action_values": dict(self.action_values),
            "remaining_depth": self.remaining_depth,
            "elapsed_ms": self.elapsed_ms,
            "time_budget_ms": self.time_budget_ms,
            "complete": self.complete,
            "timed_out": self.timed_out,
            "fallback_reason": self.fallback_reason,
            "over_time_budget": self.timed_out
            or (self.time_budget_ms is not None and self.elapsed_ms > self.time_budget_ms),
        }


@dataclass
class RolloutPlanner:
    """Choose actions by cloned pyRDDLGym lookahead rollouts. / 通过克隆 pyRDDLGym rollout 选择动作。"""

    lookahead_depth: int = 4
    name: str = "pyrddlgym-rollout"

    def choose_action(
        self,
        runtime: PyRDDLGymRuntime,
        *,
        remaining_depth: int,
        time_budget_ms: float | None = None,
    ) -> ActionDecision:
        """Choose one action for the current pyRDDLGym state. / 为当前 pyRDDLGym state 选择动作。"""
        started_at = perf_counter()
        if time_budget_ms is not None and time_budget_ms < 0.0:
            raise ValueError("time_budget_ms must be non-negative.")
        deadline = None if time_budget_ms is None else started_at + time_budget_ms / 1000.0
        depth = max(1, min(self.lookahead_depth, remaining_depth))
        candidates = runtime.action_candidates()
        action_values: dict[str, float] = {}
        best_action = candidates[0]
        best_label = action_label(best_action)
        best_value = float("-inf")
        complete = True
        fallback_reason = None
        cache: dict[tuple[tuple[tuple[str, str], ...], int], float] = {}
        try:
            for action in candidates:
                _raise_if_deadline_expired(deadline)
                label = action_label(action)
                value = self._rollout_value(runtime.clone(), action, depth, deadline, cache)
                action_values[label] = value
                if value > best_value:
                    best_action = action
                    best_label = label
                    best_value = value
        except _RuntimeDeadlineExceeded as exc:
            complete = False
            fallback_reason = str(exc)
            if best_value == float("-inf"):
                best_value = 0.0
                action_values.setdefault(best_label, best_value)
        elapsed_ms = (perf_counter() - started_at) * 1000.0
        timed_out = deadline is not None and perf_counter() > deadline
        if timed_out and complete:
            complete = False
            fallback_reason = "deadline expired after decision"
        return ActionDecision(
            action=best_action,
            label=best_label,
            value=best_value,
            action_values=action_values,
            remaining_depth=depth,
            elapsed_ms=elapsed_ms,
            time_budget_ms=time_budget_ms,
            complete=complete,
            timed_out=timed_out,
            fallback_reason=fallback_reason,
        )

    def _rollout_value(
        self,
        runtime: PyRDDLGymRuntime,
        action: Mapping[str, Any],
        depth: int,
        deadline: float | None,
        cache: dict[tuple[tuple[tuple[str, str], ...], int], float],
    ) -> float:
        """Evaluate one candidate with cloned recursive rollout. / 用克隆递归 rollout 评估一个候选动作。"""
        _raise_if_deadline_expired(deadline)
        _, reward, terminated, truncated, _ = runtime.step(action)
        if depth <= 1 or terminated or truncated:
            return reward
        best_future = self._best_future_value(runtime, depth - 1, deadline, cache)
        return reward + runtime.discount * best_future

    def _best_future_value(
        self,
        runtime: PyRDDLGymRuntime,
        depth: int,
        deadline: float | None,
        cache: dict[tuple[tuple[tuple[str, str], ...], int], float],
    ) -> float:
        """Return best rollout value from the current cloned state. / 返回当前克隆 state 的最佳 rollout value。"""
        key = (_state_cache_key(runtime.state), depth)
        if key in cache:
            return cache[key]
        best_future = float("-inf")
        for next_action in runtime.action_candidates():
            _raise_if_deadline_expired(deadline)
            value = self._rollout_value(runtime.clone(), next_action, depth, deadline, cache)
            best_future = max(best_future, value)
        if best_future == float("-inf"):
            best_future = 0.0
        cache[key] = best_future
        return best_future


def action_label(action: Mapping[str, Any]) -> str:
    """Return a compact label for an action dictionary. / 返回动作字典的紧凑标签。"""
    active = []
    for name, value in action.items():
        python_value = _json_ready(value)
        if python_value is True:
            active.append(name)
        elif python_value not in (False, 0, None):
            active.append(f"{name}={python_value}")
    return "+".join(active) if active else "noop"


class _RuntimeDeadlineExceeded(RuntimeError):
    """Signal that runtime rollout exceeded its deadline. / 表示 runtime rollout 超过 deadline。"""


def _raise_if_deadline_expired(deadline: float | None) -> None:
    """Raise when the current time is beyond the runtime deadline. / 当前时间超过 runtime deadline 时抛出异常。"""
    if deadline is not None and perf_counter() >= deadline:
        raise _RuntimeDeadlineExceeded("hard runtime rollout deadline expired")


def _state_cache_key(state: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    """Return a hashable state key for rollout memoization. / 返回 rollout 缓存使用的可哈希 state key。"""
    return tuple(sorted((str(key), repr(_json_ready(value))) for key, value in state.items()))
