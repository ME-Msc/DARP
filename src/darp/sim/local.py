"""Local sampling simulator for PlanningProblem."""

# TODO(phase-3.3): Carry observation-history keys through episodes for online
# replanning and offline policy evaluation.

from __future__ import annotations

import random
from dataclasses import dataclass

from darp.core.problem import PlanningProblem
from darp.core.types import Action, Observation, State


def _sample_weighted(items: list[tuple[object, float]], rng: random.Random) -> object:
    """Sample one item from non-normalized weights. / 从未归一化权重中采样一个元素。"""
    total = sum(weight for _, weight in items)
    if total <= 0.0:
        raise RuntimeError("Cannot sample from an empty distribution.")
    threshold = rng.random() * total
    running = 0.0
    for item, weight in items:
        running += weight
        if running >= threshold:
            return item
    return items[-1][0]


@dataclass
class LocalSimulator:
    """Run a PlanningProblem as a small local simulator. / 将 PlanningProblem 作为小型本地 simulator 运行。"""

    problem: PlanningProblem
    seed: int | None = None

    def __post_init__(self) -> None:
        """Initialize simulator-local random state. / 初始化 simulator 本地随机状态。"""
        self.rng = random.Random(self.seed)
        self.state: State | None = None
        self.steps = 0

    def reset(self) -> Observation:
        """Reset to an initial state and return the first observation. / 重置到初始 state 并返回首个 observation。"""
        self.steps = 0
        self.state = _sample_weighted(list(self.problem.initial_belief.items()), self.rng)
        if self.state in self.problem.observations:
            return self.state
        return self.problem.observations[0]

    def step(self, action: Action) -> tuple[Observation, float, bool, dict[str, object]]:
        """Apply one action and return observation, reward, done, info. / 执行动作并返回 observation、reward、done、info。"""
        if action not in self.problem.actions:
            raise ValueError(f"Unknown action {action!r}; expected one of {self.problem.actions!r}.")
        if self.state is None:
            self.reset()
        assert self.state is not None
        source = self.state
        target = _sample_weighted(
            [(state, self.problem.transition_prob(source, action, state)) for state in self.problem.states],
            self.rng,
        )
        self.state = target
        self.steps += 1
        observation = self._observe(action)
        reward = self.problem.reward(source, action)
        done = self.steps >= self.problem.max_depth
        return observation, reward, done, {"state": target}

    def _observe(self, action: Action) -> Observation:
        """Sample an observation for the current state and action. / 为当前 state 和 action 采样 observation。"""
        assert self.state is not None
        return _sample_weighted(
            [
                (observation, self.problem.observation_prob(observation, self.state, action))
                for observation in self.problem.observations
            ],
            self.rng,
        )
