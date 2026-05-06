"""Local PROST-like online execution loop."""

# TODO(phase-4.3): Replace action-averaged initial observation likelihoods with
# an explicit reset-observation model once the compiler supports it.
# TODO(phase-5.2): Let future anytime/search planners return their best partial
# policy when a hard deadline expires.

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Mapping

from darp.core.problem import PlanningProblem
from darp.core.types import Action, Distribution, Observation, State
from darp.sim.local import LocalSimulator


@dataclass(frozen=True)
class OnlineDecision:
    """Store one DARP online action choice. / 保存一次 DARP 在线动作选择。"""

    action: Action
    value: float
    action_values: Mapping[Action, float]
    remaining_depth: int
    elapsed_ms: float
    time_budget_ms: float | None = None
    complete: bool = True
    timed_out: bool = False
    fallback_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly decision record. / 返回适合 JSON 的决策记录。"""
        return {
            "action": self.action,
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


@dataclass(frozen=True)
class OnlineStep:
    """Store one online interaction step. / 保存一次在线交互步骤。"""

    step: int
    observation: Observation
    belief: Distribution
    decision: OnlineDecision
    reward: float
    next_observation: Observation
    next_belief: Distribution
    done: bool
    info: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly step record. / 返回适合 JSON 的步骤记录。"""
        return {
            "step": self.step,
            "observation": self.observation,
            "belief": dict(self.belief),
            "action": self.decision.action,
            "decision": self.decision.to_dict(),
            "reward": self.reward,
            "next_observation": self.next_observation,
            "next_belief": dict(self.next_belief),
            "done": self.done,
            "info": dict(self.info),
        }


@dataclass(frozen=True)
class OnlineSessionResult:
    """Store a complete local online session. / 保存一次完整本地在线会话。"""

    mode: str
    problem: str
    planner: str
    seed: int
    horizon: float
    max_depth: int
    total_reward: float
    steps: tuple[OnlineStep, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly session record. / 返回适合 JSON 的会话记录。"""
        return {
            "mode": self.mode,
            "problem": self.problem,
            "planner": self.planner,
            "seed": self.seed,
            "horizon": self.horizon,
            "max_depth": self.max_depth,
            "total_reward": self.total_reward,
            "steps": [step.to_dict() for step in self.steps],
        }


@dataclass
class FiniteHorizonOnlinePlanner:
    """Replan with finite-horizon dynamic programming. / 使用有限 horizon 动态规划进行重规划。"""

    problem: PlanningProblem
    name: str = "finite-horizon-dp"

    def choose_action(
        self,
        belief: Distribution,
        *,
        remaining_depth: int,
        time_budget_ms: float | None = None,
    ) -> OnlineDecision:
        """Choose one action for the current belief. / 为当前 belief 选择一个动作。"""
        started_at = perf_counter()
        if not self.problem.actions:
            raise ValueError("Cannot choose an action for a problem with no actions.")
        if time_budget_ms is not None and time_budget_ms < 0.0:
            raise ValueError("time_budget_ms must be non-negative.")
        deadline = None if time_budget_ms is None else started_at + (time_budget_ms / 1000.0)
        depth = max(1, remaining_depth)
        action_values: dict[Action, float] = {}
        complete = True
        fallback_reason = None
        try:
            _raise_if_deadline_expired(deadline)
            values = self._state_values(depth, deadline=deadline)
            for action_candidate in self.problem.actions:
                _raise_if_deadline_expired(deadline)
                action_values[action_candidate] = self._belief_action_value(
                    belief,
                    action_candidate,
                    depth,
                    values,
                    deadline=deadline,
                )
            action = max(self.problem.actions, key=lambda candidate: action_values[candidate])
        except _PlanningDeadlineExceeded as exc:
            complete = False
            fallback_reason = str(exc)
            action = _fallback_action(self.problem.actions, action_values)
            action_values.setdefault(action, 0.0)
        elapsed_ms = (perf_counter() - started_at) * 1000.0
        timed_out = deadline is not None and perf_counter() > deadline
        if timed_out and complete:
            complete = False
            fallback_reason = "deadline expired after decision"
        return OnlineDecision(
            action=action,
            value=action_values[action],
            action_values=action_values,
            remaining_depth=depth,
            elapsed_ms=elapsed_ms,
            time_budget_ms=time_budget_ms,
            complete=complete,
            timed_out=timed_out,
            fallback_reason=fallback_reason,
        )

    def _state_values(self, depth: int, deadline: float | None = None) -> list[dict[State, float]]:
        """Compute dynamic-programming values up to a depth. / 计算指定深度内的动态规划值。"""
        values: list[dict[State, float]] = [{state: 0.0 for state in self.problem.states}]
        for horizon in range(1, depth + 1):
            _raise_if_deadline_expired(deadline)
            previous = values[horizon - 1]
            current: dict[State, float] = {}
            for state in self.problem.states:
                _raise_if_deadline_expired(deadline)
                current[state] = max(
                    self.problem.reward(state, action)
                    + self.problem.discount
                    * sum(
                        self.problem.transition_prob(state, action, target)
                        * previous[target]
                        for target in self.problem.states
                    )
                    for action in self.problem.actions
                )
            values.append(current)
        return values

    def _belief_action_value(
        self,
        belief: Distribution,
        action: Action,
        depth: int,
        values: list[dict[State, float]],
        deadline: float | None = None,
    ) -> float:
        """Evaluate one action under the current belief. / 在当前 belief 下评估一个动作。"""
        previous = values[max(0, depth - 1)]
        value = 0.0
        for state, probability in belief.items():
            _raise_if_deadline_expired(deadline)
            value += probability * (
                self.problem.reward(state, action)
                + self.problem.discount
                * sum(
                    self.problem.transition_prob(state, action, target) * previous[target]
                    for target in self.problem.states
                )
            )
        return value


def run_local_online_session(
    problem: PlanningProblem,
    *,
    seed: int = 0,
    time_budget_ms: float | None = None,
) -> OnlineSessionResult:
    """Run DARP against the local simulator step by step. / 逐步运行 DARP 与本地 simulator 的交互。"""
    simulator = LocalSimulator(problem, seed=seed)
    planner = FiniteHorizonOnlinePlanner(problem)
    observation = simulator.reset()
    belief = initial_belief_from_observation(problem, observation)
    trace: list[OnlineStep] = []
    total_reward = 0.0

    for step in range(problem.max_depth):
        remaining_depth = max(1, problem.max_depth - step)
        decision = planner.choose_action(
            belief,
            remaining_depth=remaining_depth,
            time_budget_ms=time_budget_ms,
        )
        next_observation, reward, done, info = simulator.step(decision.action)
        next_belief = update_belief(problem, belief, decision.action, next_observation)
        total_reward += reward
        trace.append(
            OnlineStep(
                step=step,
                observation=observation,
                belief=belief,
                decision=decision,
                reward=reward,
                next_observation=next_observation,
                next_belief=next_belief,
                done=done,
                info=info,
            )
        )
        observation = next_observation
        belief = next_belief
        if done:
            break

    return OnlineSessionResult(
        mode="online",
        problem=problem.name,
        planner=planner.name,
        seed=seed,
        horizon=problem.horizon,
        max_depth=problem.max_depth,
        total_reward=total_reward,
        steps=tuple(trace),
    )


def initial_belief_from_observation(
    problem: PlanningProblem, observation: Observation
) -> Distribution:
    """Build the first online belief from the reset observation. / 根据 reset observation 构建初始在线 belief。"""
    corrected = {
        state: problem.initial_belief.get(state, 0.0)
        * problem.initial_observation_prob(observation, state)
        for state in problem.states
    }
    normalized = _normalize_distribution(corrected, problem.states)
    if sum(normalized.values()) <= 0.0:
        return _normalize_distribution(problem.initial_belief, problem.states)
    return normalized


def update_belief(
    problem: PlanningProblem,
    belief: Distribution,
    action: Action,
    observation: Observation,
) -> Distribution:
    """Apply a Bayesian prediction/update step. / 执行一次贝叶斯预测与更新。"""
    predicted = {
        target: sum(
            belief.get(source, 0.0) * problem.transition_prob(source, action, target)
            for source in problem.states
        )
        for target in problem.states
    }
    corrected = {
        state: predicted[state] * problem.observation_prob(observation, state, action)
        for state in problem.states
    }
    normalized = _normalize_distribution(corrected, problem.states)
    if sum(normalized.values()) <= 0.0:
        return _normalize_distribution(predicted, problem.states)
    return normalized


def _normalize_distribution(values: Mapping[State, float], states: tuple[State, ...]) -> Distribution:
    """Normalize a distribution over known states. / 对已知 states 上的分布归一化。"""
    total = sum(max(0.0, float(values.get(state, 0.0))) for state in states)
    if total <= 0.0:
        return {state: 0.0 for state in states}
    return {state: max(0.0, float(values.get(state, 0.0))) / total for state in states}


class _PlanningDeadlineExceeded(RuntimeError):
    """Signal that a planner exceeded its hard decision deadline. / 表示 planner 超过硬决策 deadline。"""


def _raise_if_deadline_expired(deadline: float | None) -> None:
    """Raise when the current time is beyond the planner deadline. / 当前时间超过规划 deadline 时抛出异常。"""
    if deadline is not None and perf_counter() >= deadline:
        raise _PlanningDeadlineExceeded("hard planning deadline expired")


def _fallback_action(
    actions: tuple[Action, ...], partial_action_values: Mapping[Action, float]
) -> Action:
    """Return the best partial action or the first valid action. / 返回当前最优部分动作或第一个合法动作。"""
    if partial_action_values:
        return max(partial_action_values, key=lambda action: partial_action_values[action])
    return actions[0]
