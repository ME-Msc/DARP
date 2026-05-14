"""pyRDDLGym environment runtime wrapper and belief helpers."""

# TODO(phase-9.1): Prefer public pyRDDLGym action-space metadata once a stable
# API is available, instead of relying on private env fields.
# TODO(phase-9.3): Replace rejected-particle fallback with likelihood weighting
# before using particle belief for benchmark-quality POMDP evaluation.

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Mapping

from darp.adapter.problem import PyRDDLGymProblem


ActionDict = dict[str, Any]


@dataclass
class PyRDDLGymRuntime:
    """Wrap pyRDDLGym reset/step for DARP planners. / 为 DARP planner 封装 pyRDDLGym reset/step。"""

    env: Any

    @classmethod
    def from_problem(cls, problem: PyRDDLGymProblem) -> "PyRDDLGymRuntime":
        """Build a runtime from a pyRDDLGym problem bundle. / 从 pyRDDLGym problem bundle 创建 runtime。"""
        if problem.env is None:
            raise ValueError("PyRDDLGymRuntime requires a PyRDDLGymProblem with an env.")
        return cls(problem.env)

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


def state_label(state: Mapping[str, Any]) -> str:
    """Return a compact label for a grounded state. / 返回 grounded state 的紧凑标签。"""
    active = [str(name) for name, value in state.items() if _json_ready(value) is True]
    if active:
        return ",".join(active)
    if not state:
        return "(empty)"
    return repr(_json_ready(state))


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
