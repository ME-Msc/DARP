"""Online sessions that connect DARP planners to pyRDDLGym environments."""

# TODO(phase-5.2): Generalize this session loop for rollout, AND-OR, full ILP,
# and HILP planners behind one trace format.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from darp.loaded import LoadedRDDL
from darp.planner import ActionDecision, RolloutPlanner
from darp.runtime import ParticleBelief, PyRDDLGymRuntime, _json_ready


@dataclass(frozen=True)
class OnlineStep:
    """Store one pyRDDLGym environment interaction. / 保存一次 pyRDDLGym 环境交互。"""

    step: int
    observation: Mapping[str, Any]
    state: Mapping[str, Any]
    belief: ParticleBelief
    decision: ActionDecision
    reward: float
    next_observation: Mapping[str, Any]
    next_state: Mapping[str, Any]
    next_belief: ParticleBelief
    terminated: bool
    truncated: bool
    info: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly step record. / 返回适合 JSON 的步骤记录。"""
        return {
            "step": self.step,
            "observation": _json_ready(self.observation),
            "state": _json_ready(self.state),
            "belief": self.belief.to_dict(),
            "action": self.decision.label,
            "action_dict": _json_ready(self.decision.action),
            "decision": self.decision.to_dict(),
            "reward": self.reward,
            "next_observation": _json_ready(self.next_observation),
            "next_state": _json_ready(self.next_state),
            "next_belief": self.next_belief.to_dict(),
            "terminated": self.terminated,
            "truncated": self.truncated,
            "done": self.done,
            "info": _json_ready(dict(self.info)),
        }

    @property
    def done(self) -> bool:
        """Return whether the episode ended after this step. / 返回该步后 episode 是否结束。"""
        return self.terminated or self.truncated


@dataclass(frozen=True)
class OnlineSessionResult:
    """Store a complete pyRDDLGym-backed online session. / 保存一次完整 pyRDDLGym 在线会话。"""

    mode: str
    problem: str
    planner: str
    seed: int
    horizon: int
    max_depth: int
    lookahead_depth: int
    total_reward: float
    is_pomdp: bool
    initial_observation: Mapping[str, Any]
    initial_belief: ParticleBelief
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
            "lookahead_depth": self.lookahead_depth,
            "total_reward": self.total_reward,
            "is_pomdp": self.is_pomdp,
            "initial_observation": _json_ready(self.initial_observation),
            "initial_belief": self.initial_belief.to_dict(),
            "steps": [step.to_dict() for step in self.steps],
        }


def run_online_session(
    loaded: LoadedRDDL,
    *,
    seed: int = 0,
    lookahead_depth: int = 4,
    time_budget_ms: float | None = None,
    particle_count: int = 32,
) -> OnlineSessionResult:
    """Run a PROST-like online loop against pyRDDLGym. / 基于 pyRDDLGym 运行 PROST 风格在线循环。"""
    runtime = PyRDDLGymRuntime.from_loaded(loaded)
    planner = RolloutPlanner(lookahead_depth=lookahead_depth)
    observation = runtime.reset(seed=seed)
    belief = runtime.initial_belief(
        observation,
        seed=seed,
        particle_count=particle_count,
    )
    trace: list[OnlineStep] = []
    total_reward = 0.0
    max_depth = runtime.horizon

    for step in range(max_depth):
        remaining_depth = max(1, max_depth - step)
        state = dict(runtime.state)
        planning_runtime = belief.representative_runtime(runtime)
        decision = planner.choose_action(
            planning_runtime,
            remaining_depth=remaining_depth,
            time_budget_ms=time_budget_ms,
        )
        next_observation, reward, terminated, truncated, info = runtime.step(decision.action)
        next_state = dict(runtime.state)
        next_belief = runtime.update_belief(
            belief,
            decision.action,
            next_observation,
            seed=seed + step + 1,
            particle_count=particle_count,
        )
        total_reward += reward
        trace.append(
            OnlineStep(
                step=step,
                observation=observation,
                state=state,
                belief=belief,
                decision=decision,
                reward=reward,
                next_observation=next_observation,
                next_state=next_state,
                next_belief=next_belief,
                terminated=terminated,
                truncated=truncated,
                info=info,
            )
        )
        observation = next_observation
        belief = next_belief
        if terminated or truncated:
            break

    return OnlineSessionResult(
        mode="online",
        problem=runtime.problem_name,
        planner=planner.name,
        seed=seed,
        horizon=runtime.horizon,
        max_depth=max_depth,
        lookahead_depth=lookahead_depth,
        total_reward=total_reward,
        is_pomdp=runtime.is_pomdp,
        initial_observation=trace[0].observation if trace else observation,
        initial_belief=trace[0].belief if trace else belief,
        steps=tuple(trace),
    )
