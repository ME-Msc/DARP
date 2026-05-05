"""Tests for Phase 3 local online execution."""

from darp.core.problem import make_tiny_grid_problem
from darp.online import (
    FiniteHorizonOnlinePlanner,
    initial_belief_from_observation,
    run_local_online_session,
    update_belief,
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


def test_local_online_session_returns_json_ready_trace():
    """Check local online execution records actions and rewards. / 检查本地在线执行会记录动作和 reward。"""
    problem = make_tiny_grid_problem()
    result = run_local_online_session(problem, steps=2, seed=7, time_budget_ms=10.0)
    payload = result.to_dict()

    assert payload["mode"] == "online"
    assert payload["planner"] == "finite-horizon-dp"
    assert len(payload["steps"]) == 2
    assert payload["steps"][0]["action"] == "safe_path"
    assert payload["steps"][0]["decision"]["time_budget_ms"] == 10.0
