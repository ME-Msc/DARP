"""Tests for explicit PlanningProblem online helpers."""

from darp.core.duration import FixedDurationModel
from darp.core.problem import PlanningProblem
from darp.core.types import Action, Observation, ObservationKey, RewardKey, State, TransitionKey
from darp.online import (
    FiniteHorizonOnlinePlanner,
    initial_belief_from_observation,
    update_belief,
)


def _explicit_test_problem() -> PlanningProblem:
    """Build a tiny explicit solver test problem. / 构建一个用于 solver 测试的显式小问题。"""
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
                observation_model[(observation, state, action)] = (
                    1.0 if observation == state else 0.0
                )

    rewards: dict[RewardKey, float] = {}
    for state in states:
        for action in actions:
            rewards[(state, action)] = 0.0
    rewards[("start", "safe_path")] = 4.0
    rewards[("safe", "safe_path")] = 6.0
    rewards[("start", "risky_path")] = 8.0

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
        duration_model=FixedDurationModel({"safe_path": 1.0, "risky_path": 2.0}),
        risk_states=frozenset({"risk"}),
        risk_budget=0.25,
        max_depth=4,
        name="explicit_test_problem",
        metadata={"source": "test"},
    )


def _partial_observation_problem() -> PlanningProblem:
    """Build a tiny POMDP with non-identity observations. / 构建一个非 identity observation 的小型 POMDP。"""
    states = ("a", "b")
    actions = ("stay",)
    observations = ("left", "right")
    transitions = {
        ("a", "stay", "a"): 1.0,
        ("a", "stay", "b"): 0.0,
        ("b", "stay", "a"): 0.0,
        ("b", "stay", "b"): 1.0,
    }
    observation_model = {
        ("left", "a", "stay"): 0.75,
        ("right", "a", "stay"): 0.25,
        ("left", "b", "stay"): 0.25,
        ("right", "b", "stay"): 0.75,
    }
    rewards = {("a", "stay"): 0.0, ("b", "stay"): 0.0}
    return PlanningProblem(
        states=states,
        actions=actions,
        observations=observations,
        transitions=transitions,
        observation_model=observation_model,
        rewards=rewards,
        initial_belief={"a": 0.5, "b": 0.5},
        horizon=2.0,
        max_depth=2,
        name="partial_observation_demo",
    )


def test_finite_horizon_online_planner_replans_from_belief():
    """Check finite-horizon replanning selects the safer root action. / 检查有限 horizon 重规划会选择更安全的根动作。"""
    problem = _explicit_test_problem()
    planner = FiniteHorizonOnlinePlanner(problem)
    decision = planner.choose_action(
        problem.initial_belief,
        remaining_depth=problem.max_depth,
    )

    assert decision.action == "safe_path"
    assert decision.action_values["safe_path"] > decision.action_values["risky_path"]


def test_online_belief_update_tracks_identity_observation():
    """Check belief carryover follows identity observations. / 检查 belief 传递会跟随 identity observation。"""
    problem = _explicit_test_problem()
    belief = initial_belief_from_observation(problem, "start")
    next_belief = update_belief(problem, belief, "safe_path", "safe")

    assert belief == {"start": 1.0, "safe": 0.0, "risk": 0.0, "goal": 0.0}
    assert next_belief["safe"] == 1.0
    assert sum(next_belief.values()) == 1.0


def test_online_belief_update_tracks_partial_observation():
    """Check belief carryover uses the observation model. / 检查 belief 传递会使用 observation model。"""
    problem = _partial_observation_problem()
    belief = initial_belief_from_observation(problem, "left")
    next_belief = update_belief(problem, belief, "stay", "left")

    assert belief == {"a": 0.75, "b": 0.25}
    assert round(next_belief["a"], 6) == 0.9
    assert round(next_belief["b"], 6) == 0.1
    assert sum(next_belief.values()) == 1.0


def test_online_planner_hard_deadline_returns_fallback():
    """Check zero budgets force a traceable fallback action. / 检查零预算会强制返回可追踪 fallback action。"""
    problem = _explicit_test_problem()
    planner = FiniteHorizonOnlinePlanner(problem)
    decision = planner.choose_action(
        problem.initial_belief,
        remaining_depth=problem.max_depth,
        time_budget_ms=0.0,
    )
    payload = decision.to_dict()

    assert decision.action == problem.actions[0]
    assert payload["timed_out"] is True
    assert payload["complete"] is False
    assert payload["over_time_budget"] is True
    assert payload["fallback_reason"] == "hard planning deadline expired"
