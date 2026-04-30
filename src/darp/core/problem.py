"""Finite-horizon POMDP/(C)C-POMDP problem interface."""

# TODO(phase-4.1): Replace permissive mappings with validated typed model builders.
# TODO(phase-4.3): Add multi-constraint and continuous-state extension points.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from darp.core.duration import DurationModel, FixedDurationModel
from darp.core.types import (
    Action,
    Distribution,
    Observation,
    ObservationKey,
    RewardKey,
    State,
    TransitionKey,
)


@dataclass
class PlanningProblem:
    """Represent a finite-horizon planning model. / 表示一个有限 horizon 的规划问题模型。"""

    states: tuple[State, ...]
    actions: tuple[Action, ...]
    observations: tuple[Observation, ...]
    transitions: Mapping[TransitionKey, float]
    observation_model: Mapping[ObservationKey, float]
    rewards: Mapping[RewardKey, float]
    initial_belief: Distribution
    horizon: float
    discount: float = 1.0
    duration_model: DurationModel = field(default_factory=lambda: FixedDurationModel({}))
    zeta: float = 0.0
    costs: Mapping[RewardKey, float] = field(default_factory=dict)
    cost_budget: float | None = None
    risk_states: frozenset[State] = field(default_factory=frozenset)
    risk_budget: float | None = None
    max_depth: int = 12
    name: str = "planning_problem"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def transition_prob(self, source: State, action: Action, target: State) -> float:
        """Return P(target | source, action). / 返回给定 source 和 action 后到达 target 的概率。"""
        return float(self.transitions.get((source, action, target), 0.0))

    def observation_prob(self, observation: Observation, state: State, action: Action) -> float:
        """Return P(observation | state, action). / 返回给定 state 和 action 后观测到 observation 的概率。"""
        return float(self.observation_model.get((observation, state, action), 0.0))

    def reward(self, state: State, action: Action) -> float:
        """Return the immediate reward for one state-action pair. / 返回一个 state-action 对的即时 reward。"""
        return float(self.rewards.get((state, action), 0.0))

    def cost(self, state: State, action: Action) -> float:
        """Return the immediate constraint cost for one pair. / 返回一个 state-action 对的即时约束 cost。"""
        return float(self.costs.get((state, action), 0.0))

    @property
    def constraint_budget(self) -> float | None:
        """Return the active single-constraint budget if present. / 返回当前单约束 budget。"""
        if self.risk_budget is not None:
            initial_risk = sum(
                prob for state, prob in self.initial_belief.items() if state in self.risk_states
            )
            return max(0.0, self.risk_budget - initial_risk)
        return self.cost_budget

    @property
    def has_constraint(self) -> bool:
        """Return whether the problem has an active constraint. / 返回问题是否含有活动约束。"""
        return self.constraint_budget is not None

    def to_summary_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly compact problem summary. / 返回适合 JSON 输出的问题摘要。"""
        nonzero_transitions = sum(1 for value in self.transitions.values() if abs(value) > 1e-12)
        nonzero_observations = sum(
            1 for value in self.observation_model.values() if abs(value) > 1e-12
        )
        return {
            "name": self.name,
            "states": list(self.states),
            "actions": list(self.actions),
            "observations": list(self.observations),
            "horizon": self.horizon,
            "discount": self.discount,
            "initial_belief": dict(self.initial_belief),
            "nonzero_transitions": nonzero_transitions,
            "nonzero_observations": nonzero_observations,
            "metadata": dict(self.metadata),
        }


def make_tiny_grid_problem() -> PlanningProblem:
    """Build a tiny hand-checkable problem used by CLI demos and tests. / 构建用于 demo 和测试的小型问题。"""

    states: tuple[State, ...] = ("start", "safe", "risk", "goal")
    actions: tuple[Action, ...] = ("safe_path", "risky_path")
    observations: tuple[Observation, ...] = ("start", "safe", "risk", "goal")

    transitions: dict[TransitionKey, float] = {}
    for state in states:
        for action in actions:
            for target in states:
                transitions[(state, action, target)] = 0.0

    transitions[("start", "safe_path", "safe")] = 1.0
    transitions[("safe", "safe_path", "goal")] = 1.0
    transitions[("risk", "safe_path", "goal")] = 1.0
    transitions[("goal", "safe_path", "goal")] = 1.0
    transitions[("start", "risky_path", "goal")] = 0.8
    transitions[("start", "risky_path", "risk")] = 0.2
    transitions[("safe", "risky_path", "goal")] = 1.0
    transitions[("risk", "risky_path", "risk")] = 1.0
    transitions[("goal", "risky_path", "goal")] = 1.0

    observation_model: dict[ObservationKey, float] = {}
    for observation in observations:
        for state in states:
            for action in actions:
                observation_model[(observation, state, action)] = 1.0 if observation == state else 0.0

    rewards: dict[RewardKey, float] = {}
    for state in states:
        for action in actions:
            rewards[(state, action)] = 0.0
    rewards[("start", "safe_path")] = 4.0
    rewards[("safe", "safe_path")] = 6.0
    rewards[("start", "risky_path")] = 8.0
    rewards[("goal", "safe_path")] = 0.0
    rewards[("goal", "risky_path")] = 0.0

    duration_model = FixedDurationModel({"safe_path": 1.0, "risky_path": 2.0})

    return PlanningProblem(
        states=states,
        actions=actions,
        observations=observations,
        transitions=transitions,
        observation_model=observation_model,
        rewards=rewards,
        initial_belief={"start": 1.0, "safe": 0.0, "risk": 0.0, "goal": 0.0},
        horizon=2.0,
        discount=1.0,
        duration_model=duration_model,
        zeta=0.0,
        risk_states=frozenset({"risk"}),
        risk_budget=0.25,
        max_depth=4,
        name="tiny_grid_builtin",
        metadata={"source": "builtin"},
    )
