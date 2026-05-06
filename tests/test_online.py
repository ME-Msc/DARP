"""Tests for Phase 3 local online execution."""

from darp.core.problem import PlanningProblem, make_tiny_grid_problem
from darp.online import (
    FiniteHorizonOnlinePlanner,
    initial_belief_from_observation,
    run_local_online_session,
    update_belief,
)
from darp.sim.local import LocalSimulator


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
    problem = make_tiny_grid_problem()
    planner = FiniteHorizonOnlinePlanner(problem)
    decision = planner.choose_action(
        problem.initial_belief,
        remaining_depth=problem.max_depth,
    )

    assert decision.action == "safe_path"
    assert decision.action_values["safe_path"] > decision.action_values["risky_path"]


def test_online_belief_update_tracks_identity_observation():
    """Check belief carryover follows identity observations. / 检查 belief 传递会跟随 identity observation。"""
    problem = make_tiny_grid_problem()
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


def test_local_simulator_reset_samples_initial_observation_model():
    """Check reset observations come from the observation model. / 检查 reset observation 来自 observation model。"""
    problem = _partial_observation_problem()
    problem.initial_belief = {"a": 1.0, "b": 0.0}
    simulator = LocalSimulator(problem, seed=7)

    assert simulator.reset() == "left"


def test_online_planner_hard_deadline_returns_fallback():
    """Check zero budgets force a traceable fallback action. / 检查零预算会强制返回可追踪 fallback action。"""
    problem = make_tiny_grid_problem()
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


def test_local_online_session_returns_json_ready_trace():
    """Check local online execution records actions and rewards. / 检查本地在线执行会记录动作和 reward。"""
    problem = make_tiny_grid_problem()
    result = run_local_online_session(problem, seed=7, time_budget_ms=10.0)
    payload = result.to_dict()

    assert payload["mode"] == "online"
    assert payload["planner"] == "finite-horizon-dp"
    assert len(payload["steps"]) == problem.max_depth
    assert payload["steps"][0]["action"] == "safe_path"
    assert payload["steps"][0]["decision"]["time_budget_ms"] == 10.0
    assert payload["steps"][0]["decision"]["complete"] is True
    assert payload["steps"][0]["decision"]["timed_out"] is False
