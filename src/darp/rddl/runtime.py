"""pyRDDLGym-backed online runtime and simple rollout planner."""

# TODO(phase-5.1): Move the rollout planner behind the shared planner registry.
# TODO(phase-5.1): Prefer public pyRDDLGym action-space metadata once a stable
# API is available, instead of relying on private env fields.
# TODO(phase-5.1): Replace rejected-particle fallback with likelihood weighting
# before using particle belief for benchmark-quality POMDP evaluation.

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Mapping

from darp.rddl.artifacts import RDDLArtifacts


ActionDict = dict[str, Any]


@dataclass(frozen=True)
class PyRDDLGymDecision:
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


@dataclass(frozen=True)
class PyRDDLGymStep:
    """Store one pyRDDLGym environment interaction. / 保存一次 pyRDDLGym 环境交互。"""

    step: int
    observation: Mapping[str, Any]
    state: Mapping[str, Any]
    belief: "ParticleBelief"
    decision: PyRDDLGymDecision
    reward: float
    next_observation: Mapping[str, Any]
    next_state: Mapping[str, Any]
    next_belief: "ParticleBelief"
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
class PyRDDLGymOnlineResult:
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
    initial_belief: "ParticleBelief"
    steps: tuple[PyRDDLGymStep, ...]

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


@dataclass
class PyRDDLGymRuntime:
    """Wrap pyRDDLGym reset/step for DARP planners. / 为 DARP planner 封装 pyRDDLGym reset/step。"""

    env: Any

    @classmethod
    def from_loaded(cls, loaded: RDDLArtifacts) -> "PyRDDLGymRuntime":
        """Build a runtime from loaded pyRDDLGym artifacts. / 从已加载的 pyRDDLGym 产物创建 runtime。"""
        if loaded.env is None:
            raise ValueError("PyRDDLGymRuntime requires pyRDDLGym RDDLArtifacts with an env.")
        return cls(loaded.env)

    @property
    def problem_name(self) -> str:
        """Return the pyRDDLGym instance name. / 返回 pyRDDLGym instance 名称。"""
        model = getattr(self.env, "model", None)
        return str(getattr(model, "instance_name", None) or getattr(model, "domain_name", "rddl"))

    @property
    def horizon(self) -> int:
        """Return the environment horizon. / 返回环境 horizon。"""
        return int(getattr(self.env, "horizon", 0) or 0)

    @property
    def discount(self) -> float:
        """Return the model discount factor. / 返回模型 discount factor。"""
        model = getattr(self.env, "model", None)
        return float(getattr(model, "discount", 1.0) or 1.0)

    @property
    def is_pomdp(self) -> bool:
        """Return whether pyRDDLGym treats this model as a POMDP. / 返回 pyRDDLGym 是否将模型视为 POMDP。"""
        sampler = getattr(self.env, "sampler", None)
        if sampler is not None and hasattr(sampler, "is_pomdp"):
            return bool(sampler.is_pomdp)
        model = getattr(self.env, "model", None)
        return bool(getattr(model, "observ_fluents", {}) or {})

    @property
    def state(self) -> Mapping[str, Any]:
        """Return the current grounded state dictionary. / 返回当前 grounded state 字典。"""
        return dict(getattr(self.env, "state", {}) or {})

    def reset(self, seed: int | None = None) -> Mapping[str, Any]:
        """Reset the environment and return the first observation. / 重置环境并返回首个 observation。"""
        observation, _ = self.env.reset(seed=seed)
        return dict(observation)

    def step(self, action: Mapping[str, Any]) -> tuple[Mapping[str, Any], float, bool, bool, dict[str, Any]]:
        """Apply one action through pyRDDLGym. / 通过 pyRDDLGym 执行一个动作。"""
        observation, reward, terminated, truncated, info = self.env.step(dict(action))
        return dict(observation), float(reward), bool(terminated), bool(truncated), dict(info)

    def clone(self) -> "PyRDDLGymRuntime":
        """Return an isolated runtime copy for rollout. / 返回用于 rollout 的隔离 runtime 副本。"""
        return PyRDDLGymRuntime(copy.deepcopy(self.env))

    def noop_action(self) -> ActionDict:
        """Return pyRDDLGym's default action assignment. / 返回 pyRDDLGym 默认动作赋值。"""
        noop = getattr(self.env, "_noop_actions", None)
        if noop is not None:
            return dict(noop)
        model = getattr(self.env, "model", None)
        action_fluents = getattr(model, "action_fluents", {}) if model is not None else {}
        return {str(action): value for action, value in action_fluents.items()}

    def action_candidates(self) -> tuple[ActionDict, ...]:
        """Return noop plus one-active-boolean action candidates. / 返回 noop 和单个 bool action 的候选动作。"""
        base = self.noop_action()
        candidates: list[ActionDict] = [base]
        ranges = getattr(self.env, "_action_ranges", {}) or {}
        for action_name, action_range in ranges.items():
            if str(action_range) != "bool":
                continue
            candidate = dict(base)
            candidate[action_name] = True
            candidates.append(candidate)
        return tuple(candidates)

    def initial_belief(
        self,
        observation: Mapping[str, Any],
        *,
        seed: int | None = None,
        particle_count: int = 32,
        max_attempts: int | None = None,
    ) -> "ParticleBelief":
        """Build an initial state belief from the reset observation. / 根据 reset observation 构建初始 state belief。"""
        if not self.is_pomdp:
            return ParticleBelief.exact(self.clone(), observation=observation, source="mdp-state")
        return self._sample_initial_particles(
            observation,
            seed=seed,
            particle_count=particle_count,
            max_attempts=max_attempts,
        )

    def update_belief(
        self,
        previous: "ParticleBelief",
        action: Mapping[str, Any],
        observation: Mapping[str, Any],
        *,
        seed: int | None = None,
        particle_count: int | None = None,
        max_attempts: int | None = None,
    ) -> "ParticleBelief":
        """Update the state belief after one action and observation. / 根据 action 和 observation 更新 state belief。"""
        if not self.is_pomdp:
            return ParticleBelief.exact(self.clone(), observation=observation, source="mdp-state")
        target_count = particle_count or max(1, previous.particle_count)
        attempts_limit = max_attempts or max(32, target_count * 20)
        source_particles = previous.particles or (self.clone(),)
        particles: list[PyRDDLGymRuntime] = []
        attempts = 0
        while len(particles) < target_count and attempts < attempts_limit:
            source = source_particles[attempts % len(source_particles)].clone()
            sampled_observation, _, _, _, _ = source.step(action)
            if _observations_match(sampled_observation, observation):
                particles.append(source)
            attempts += 1
        if not particles:
            particles.append(self.clone())
        return ParticleBelief(
            particles=tuple(particles),
            observation=dict(observation),
            is_pomdp=True,
            is_exact=False,
            source="particle-filter",
            requested_particles=target_count,
            attempts=attempts,
        )

    def _sample_initial_particles(
        self,
        observation: Mapping[str, Any],
        *,
        seed: int | None,
        particle_count: int,
        max_attempts: int | None,
    ) -> "ParticleBelief":
        """Sample initial particles matching a reset observation. / 采样匹配 reset observation 的初始粒子。"""
        target_count = max(1, particle_count)
        attempts_limit = max_attempts or max(32, target_count * 20)
        particles: list[PyRDDLGymRuntime] = []
        attempts = 0
        while len(particles) < target_count and attempts < attempts_limit:
            candidate = self.clone()
            candidate_seed = None if seed is None else seed + attempts + 1
            sampled_observation = candidate.reset(seed=candidate_seed)
            if _observations_match(sampled_observation, observation):
                particles.append(candidate)
            attempts += 1
        if not particles:
            particles.append(self.clone())
        return ParticleBelief(
            particles=tuple(particles),
            observation=dict(observation),
            is_pomdp=True,
            is_exact=False,
            source="initial-rejection-sampling",
            requested_particles=target_count,
            attempts=attempts,
        )


@dataclass(frozen=True)
class ParticleBelief:
    """Represent an MDP/POMDP belief with runtime particles. / 用 runtime 粒子表示 MDP/POMDP belief。"""

    particles: tuple[PyRDDLGymRuntime, ...]
    observation: Mapping[str, Any]
    is_pomdp: bool
    is_exact: bool
    source: str
    requested_particles: int = 1
    attempts: int = 0

    @classmethod
    def exact(
        cls,
        runtime: PyRDDLGymRuntime,
        *,
        observation: Mapping[str, Any],
        source: str,
    ) -> "ParticleBelief":
        """Build a singleton exact belief. / 构建单粒子精确 belief。"""
        return cls(
            particles=(runtime,),
            observation=dict(observation),
            is_pomdp=False,
            is_exact=True,
            source=source,
            requested_particles=1,
            attempts=1,
        )

    @property
    def particle_count(self) -> int:
        """Return the number of particles. / 返回粒子数量。"""
        return len(self.particles)

    def representative_runtime(self, fallback: PyRDDLGymRuntime) -> PyRDDLGymRuntime:
        """Return one runtime for rollout planning. / 返回一个用于 rollout planning 的 runtime。"""
        if self.particles:
            return self.particles[0].clone()
        return fallback.clone()

    def support(self) -> dict[str, float]:
        """Return empirical state support probabilities. / 返回经验 state support 概率。"""
        if not self.particles:
            return {}
        counts: dict[str, int] = {}
        for particle in self.particles:
            key = state_label(particle.state)
            counts[key] = counts.get(key, 0) + 1
        total = float(len(self.particles))
        return {state: count / total for state, count in sorted(counts.items())}

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly belief summary. / 返回 JSON 友好的 belief 摘要。"""
        return {
            "is_pomdp": self.is_pomdp,
            "is_exact": self.is_exact,
            "source": self.source,
            "particle_count": self.particle_count,
            "requested_particles": self.requested_particles,
            "attempts": self.attempts,
            "observation": _json_ready(self.observation),
            "support": self.support(),
        }


@dataclass
class RolloutOnlinePlanner:
    """Choose actions by cloned pyRDDLGym lookahead rollouts. / 通过克隆 pyRDDLGym rollout 选择动作。"""

    lookahead_depth: int = 4
    name: str = "pyrddlgym-rollout"

    def choose_action(
        self,
        runtime: PyRDDLGymRuntime,
        *,
        remaining_depth: int,
        time_budget_ms: float | None = None,
    ) -> PyRDDLGymDecision:
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
        return PyRDDLGymDecision(
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


def run_pyrddlgym_online_session(
    loaded: RDDLArtifacts,
    *,
    seed: int = 0,
    lookahead_depth: int = 4,
    time_budget_ms: float | None = None,
    particle_count: int = 32,
) -> PyRDDLGymOnlineResult:
    """Run a PROST-like online loop against pyRDDLGym. / 基于 pyRDDLGym 运行 PROST 风格在线循环。"""
    runtime = PyRDDLGymRuntime.from_loaded(loaded)
    planner = RolloutOnlinePlanner(lookahead_depth=lookahead_depth)
    observation = runtime.reset(seed=seed)
    belief = runtime.initial_belief(
        observation,
        seed=seed,
        particle_count=particle_count,
    )
    trace: list[PyRDDLGymStep] = []
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
            PyRDDLGymStep(
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

    return PyRDDLGymOnlineResult(
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


def state_label(state: Mapping[str, Any]) -> str:
    """Return a compact label for a grounded state. / 返回 grounded state 的紧凑标签。"""
    active = [str(name) for name, value in state.items() if _json_ready(value) is True]
    if active:
        return ",".join(active)
    if not state:
        return "(empty)"
    return repr(_json_ready(state))


class _RuntimeDeadlineExceeded(RuntimeError):
    """Signal that runtime rollout exceeded its deadline. / 表示 runtime rollout 超过 deadline。"""


def _raise_if_deadline_expired(deadline: float | None) -> None:
    """Raise when the current time is beyond the runtime deadline. / 当前时间超过 runtime deadline 时抛出异常。"""
    if deadline is not None and perf_counter() >= deadline:
        raise _RuntimeDeadlineExceeded("hard runtime rollout deadline expired")


def _json_ready(value: Any) -> Any:
    """Convert numpy-heavy pyRDDLGym values to JSON-friendly values. / 将 pyRDDLGym 值转为 JSON 友好值。"""
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_json_ready(item) for item in value)
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if hasattr(value, "tolist"):
        return _json_ready(value.tolist())
    if hasattr(value, "item"):
        return value.item()
    return value


def _observations_match(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    """Return whether two observations match after JSON conversion. / 判断两个 observation 规范化后是否匹配。"""
    return _json_ready(dict(left)) == _json_ready(dict(right))


def _state_cache_key(state: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    """Return a hashable state key for rollout memoization. / 返回 rollout 缓存使用的可哈希 state key。"""
    return tuple(sorted((str(key), repr(_json_ready(value))) for key, value in state.items()))
