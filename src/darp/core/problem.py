"""Finite-horizon POMDP/(C)C-POMDP problem interface."""

# TODO(phase-7.1): Move larger sparse transition/observation tables behind a
# backend-friendly matrix abstraction before ILP and benchmark scaling.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from darp.core.duration import DurationModel, FixedDurationModel
from darp.core.types import (
    Action,
    Distribution,
    GroundAtom,
    Observation,
    ObservationKey,
    ResetObservationKey,
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
    reset_observation_model: Mapping[ResetObservationKey, float] = field(default_factory=dict)
    action_fluents: Mapping[Action, frozenset[GroundAtom]] = field(default_factory=dict)
    max_nondef_actions: int | None = None
    max_depth: int = 12
    name: str = "planning_problem"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate explicit finite-horizon table semantics. / 校验显式有限 horizon 表语义。"""
        self.validate()

    def validate(self) -> None:
        """Raise a clear error when the model tables are inconsistent. / 模型表不一致时抛出清晰错误。"""
        if not self.states:
            raise ValueError("PlanningProblem must define at least one state.")
        if not self.actions:
            raise ValueError("PlanningProblem must define at least one action.")
        if not self.observations:
            raise ValueError("PlanningProblem must define at least one observation.")
        if self.max_depth < 1:
            raise ValueError("PlanningProblem.max_depth must be at least 1.")
        if self.horizon < 0:
            raise ValueError("PlanningProblem.horizon must be non-negative.")
        if self.discount < 0:
            raise ValueError("PlanningProblem.discount must be non-negative.")
        if self.max_nondef_actions is not None and self.max_nondef_actions < 0:
            raise ValueError("PlanningProblem.max_nondef_actions must be non-negative.")
        self._validate_initial_belief()
        self._validate_transition_model()
        self._validate_observation_model()
        self._validate_reset_observation_model()
        self._validate_action_fluents()

    def _validate_initial_belief(self) -> None:
        """Validate that the initial belief is normalized on known states. / 校验初始 belief 在已知 state 上归一化。"""
        unknown = set(self.initial_belief) - set(self.states)
        if unknown:
            raise ValueError(f"Initial belief contains unknown states: {sorted(map(str, unknown))!r}.")
        total = 0.0
        for state in self.states:
            probability = float(self.initial_belief.get(state, 0.0))
            if probability < -1e-12:
                raise ValueError(f"Initial belief for state {state!r} is negative.")
            total += max(0.0, probability)
        if abs(total - 1.0) > 1e-8:
            raise ValueError(f"Initial belief must sum to 1.0, found {total:.6g}.")

    def _validate_transition_model(self) -> None:
        """Validate transition probability mass for every state-action pair. / 校验每个 state-action 的转移概率质量。"""
        for source in self.states:
            for action in self.actions:
                total = 0.0
                for target in self.states:
                    probability = self.transition_prob(source, action, target)
                    if probability < -1e-12:
                        raise ValueError(
                            f"Transition probability for {(source, action, target)!r} is negative."
                        )
                    total += max(0.0, probability)
                if abs(total - 1.0) > 1e-8:
                    raise ValueError(
                        f"Transition mass for state={source!r}, action={action!r} "
                        f"must sum to 1.0, found {total:.6g}."
                    )

    def _validate_observation_model(self) -> None:
        """Validate observation probability mass for every state-action pair. / 校验每个 state-action 的观测概率质量。"""
        for state in self.states:
            for action in self.actions:
                total = 0.0
                for observation in self.observations:
                    probability = self.observation_prob(observation, state, action)
                    if probability < -1e-12:
                        raise ValueError(
                            f"Observation probability for {(observation, state, action)!r} is negative."
                        )
                    total += max(0.0, probability)
                if abs(total - 1.0) > 1e-8:
                    raise ValueError(
                        f"Observation mass for state={state!r}, action={action!r} "
                        f"must sum to 1.0, found {total:.6g}."
                    )

    def _validate_reset_observation_model(self) -> None:
        """Validate reset-observation mass when provided. / 校验 reset-observation 概率质量。"""
        if not self.reset_observation_model:
            return
        for state in self.states:
            total = 0.0
            for observation in self.observations:
                probability = float(self.reset_observation_model.get((observation, state), 0.0))
                if probability < -1e-12:
                    raise ValueError(
                        f"Reset observation probability for {(observation, state)!r} is negative."
                    )
                total += max(0.0, probability)
            if abs(total - 1.0) > 1e-8:
                raise ValueError(
                    f"Reset observation mass for state={state!r} must sum to 1.0, "
                    f"found {total:.6g}."
                )

    def _validate_action_fluents(self) -> None:
        """Validate action fluent metadata and max-nondef constraints. / 校验 action fluent 元数据和 max-nondef 约束。"""
        unknown = set(self.action_fluents) - set(self.actions)
        if unknown:
            raise ValueError(f"Action fluent metadata contains unknown actions: {sorted(unknown)!r}.")
        if self.max_nondef_actions is None:
            return
        for action in self.actions:
            active = len(self.action_fluents.get(action, frozenset()))
            if active > self.max_nondef_actions:
                raise ValueError(
                    f"Action {action!r} activates {active} fluents, exceeding "
                    f"max_nondef_actions={self.max_nondef_actions}."
                )

    def transition_prob(self, source: State, action: Action, target: State) -> float:
        """Return P(target | source, action). / 返回给定 source 和 action 后到达 target 的概率。"""
        return float(self.transitions.get((source, action, target), 0.0))

    def observation_prob(self, observation: Observation, state: State, action: Action) -> float:
        """Return P(observation | state, action). / 返回给定 state 和 action 后观测到 observation 的概率。"""
        return float(self.observation_model.get((observation, state, action), 0.0))

    def initial_observation_prob(self, observation: Observation, state: State) -> float:
        """Return P(first observation | initial state). / 返回给定初始 state 的首个 observation 概率。"""
        if self.reset_observation_model:
            return float(self.reset_observation_model.get((observation, state), 0.0))
        action_likelihoods = [
            self.observation_prob(observation, state, action) for action in self.actions
        ]
        has_explicit_likelihood = any(
            (observation, state, action) in self.observation_model for action in self.actions
        )
        positive_likelihoods = [value for value in action_likelihoods if value > 0.0]
        if has_explicit_likelihood or positive_likelihoods:
            return sum(action_likelihoods) / len(action_likelihoods)
        return 1.0 if observation == state else 0.0

    def reward(self, state: State, action: Action) -> float:
        """Return the immediate reward for one state-action pair. / 返回一个 state-action 对的即时 reward。"""
        return float(self.rewards.get((state, action), 0.0))

    def cost(self, state: State, action: Action) -> float:
        """Return the immediate constraint cost for one pair. / 返回一个 state-action 对的即时约束 cost。"""
        return float(self.costs.get((state, action), 0.0))

    def is_action_allowed(self, action: Action) -> bool:
        """Return whether an action satisfies static action constraints. / 返回 action 是否满足静态动作约束。"""
        if action not in self.actions:
            return False
        if self.max_nondef_actions is None:
            return True
        return len(self.action_fluents.get(action, frozenset())) <= self.max_nondef_actions

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
            "max_nondef_actions": self.max_nondef_actions,
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
