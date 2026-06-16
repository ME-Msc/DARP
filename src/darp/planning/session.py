"""Online sessions that connect DARP planners to pyRDDLGym environments."""

# TODO(phase-9.2): Add offline policy replay/evaluation traces and benchmark
# runners on top of this online loop.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping

from darp.adapter.exact import ExactBeliefState
from darp.adapter.problem import PyRDDLGymProblem
from darp.adapter.runtime import ParticleBelief, PyRDDLGymRuntime, _json_ready
from darp.model.duration_sidecar import DurationSidecar, build_duration_sidecar
from darp.planning.full_ilp import FullILPPlanner
from darp.planning.hilp import HILPPlanner
from darp.planning.rollout import ActionDecision, RolloutPlanner

PlannerName = Literal["hilp", "full-ilp", "rollout"]
BeliefRecord = ParticleBelief | ExactBeliefState


@dataclass(frozen=True)
class OnlineStep:
    """Store one pyRDDLGym environment interaction. / 保存一次 pyRDDLGym 环境交互。"""

    step: int
    observation: Mapping[str, Any]
    state: Mapping[str, Any]
    belief: BeliefRecord
    decision: ActionDecision
    reward: float
    next_observation: Mapping[str, Any]
    next_state: Mapping[str, Any]
    next_belief: BeliefRecord
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
    duration: Mapping[str, Any]
    total_reward: float
    is_pomdp: bool
    initial_observation: Mapping[str, Any]
    initial_belief: BeliefRecord
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
            "duration": _json_ready(dict(self.duration)),
            "total_reward": self.total_reward,
            "is_pomdp": self.is_pomdp,
            "initial_observation": _json_ready(self.initial_observation),
            "initial_belief": self.initial_belief.to_dict(),
            "steps": [step.to_dict() for step in self.steps],
        }


def run_online_session(
    problem: PyRDDLGymProblem,
    *,
    seed: int = 0,
    lookahead_depth: int = 4,
    planner_name: PlannerName = "rollout",
    duration_sidecar: DurationSidecar | None = None,
    hilp_iterations: int = 4,
    frontier_width: int = 1,
    risk_budget: float | None = None,
    time_budget_ms: float | None = None,
    particle_count: int = 32,
) -> OnlineSessionResult:
    """Run a PROST-like online loop against pyRDDLGym. / 基于 pyRDDLGym 运行 PROST 风格在线循环。"""
    runtime = PyRDDLGymRuntime.from_problem(problem)
    duration = duration_sidecar or _default_duration_sidecar()
    sidecar_risk = duration.risk_spec()
    planner_risk_budget = risk_budget if risk_budget is not None else sidecar_risk.budget
    planner = _build_planner(
        planner_name,
        lookahead_depth=lookahead_depth,
        hilp_iterations=hilp_iterations,
        frontier_width=frontier_width,
        risk_budget=planner_risk_budget,
    )
    observation = runtime.reset(seed=seed)
    view = None
    interface = None
    if planner_name != "rollout":
        view = problem.build_grounded_view()
        interface = view.build_and_or_interface(runtime, risk=sidecar_risk)
        duration.validate_actions([choice.label for choice in interface.actions])
    if planner_name == "rollout":
        belief: BeliefRecord = runtime.initial_belief(
            observation,
            seed=seed,
            particle_count=particle_count,
        )
    else:
        assert interface is not None
        if interface.exact_kernel is None:
            raise ValueError("Paper-path planners require an exact kernel.")
        belief = ExactBeliefState.from_runtime(
            interface.exact_kernel,
            runtime,
            observation,
            is_pomdp=runtime.is_pomdp,
        )
    trace: list[OnlineStep] = []
    total_reward = 0.0
    max_depth = runtime.horizon

    for step in range(max_depth):
        remaining_depth = max(1, max_depth - step)
        state = dict(runtime.state)
        if planner_name == "rollout":
            assert isinstance(belief, ParticleBelief)
            planning_runtime = belief.representative_runtime(runtime)
            assert isinstance(planner, RolloutPlanner)
            decision = planner.choose_action(
                planning_runtime,
                remaining_depth=remaining_depth,
                time_budget_ms=time_budget_ms,
            )
        else:
            assert interface is not None
            if interface.exact_kernel is None:
                raise ValueError("Paper-path planners require an exact kernel.")
            assert isinstance(belief, ExactBeliefState)
            planning_runtime = runtime.clone()
            root_belief = belief.belief
            assert isinstance(planner, FullILPPlanner | HILPPlanner)
            decision = planner.choose_action(
                planning_runtime,
                interface,
                duration.evaluator(horizon=remaining_depth),
                remaining_depth=remaining_depth,
                root_belief=root_belief,
                time_budget_ms=time_budget_ms,
            )
        next_observation, reward, terminated, truncated, info = runtime.step(decision.action)
        next_state = dict(runtime.state)
        if planner_name == "rollout":
            assert isinstance(belief, ParticleBelief)
            next_belief: BeliefRecord = runtime.update_belief(
                belief,
                decision.action,
                next_observation,
                seed=seed + step + 1,
                particle_count=particle_count,
            )
        else:
            assert interface is not None
            assert interface.exact_kernel is not None
            assert isinstance(belief, ExactBeliefState)
            next_belief = belief.advance(
                interface.exact_kernel,
                decision.action,
                next_observation,
                observed_state=next_state,
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
        duration=_duration_summary(duration, defaulted=duration_sidecar is None),
        total_reward=total_reward,
        is_pomdp=runtime.is_pomdp,
        initial_observation=trace[0].observation if trace else observation,
        initial_belief=trace[0].belief if trace else belief,
        steps=tuple(trace),
    )


def _build_planner(
    planner_name: PlannerName,
    *,
    lookahead_depth: int,
    hilp_iterations: int,
    frontier_width: int,
    risk_budget: float | None,
) -> RolloutPlanner | FullILPPlanner | HILPPlanner:
    """Build the requested online planner. / 构建请求的在线 planner。"""
    if planner_name == "rollout":
        return RolloutPlanner(lookahead_depth=lookahead_depth)
    if planner_name == "full-ilp":
        return FullILPPlanner(
            risk_budget=risk_budget,
        )
    if planner_name == "hilp":
        return HILPPlanner(
            lookahead_depth=lookahead_depth,
            max_iterations=hilp_iterations,
            frontier_width=frontier_width,
            risk_budget=risk_budget,
        )
    raise ValueError(f"Unsupported planner: {planner_name}")


def _default_duration_sidecar() -> DurationSidecar:
    """Return the default unit-duration sidecar. / 返回默认单位 duration sidecar。"""
    return build_duration_sidecar({"kind": "fixed", "default": 1.0})


def _duration_summary(duration: DurationSidecar, *, defaulted: bool) -> dict[str, Any]:
    """Return a compact duration summary for traces. / 返回 trace 使用的 duration 摘要。"""
    return {
        "kind": duration.metadata.get("kind"),
        "path": str(duration.path) if duration.path is not None else None,
        "defaulted": defaulted,
    }
