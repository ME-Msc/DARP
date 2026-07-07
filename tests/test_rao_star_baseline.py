"""Tests for the external RAO* comparison baseline."""

from experiments.baselines.rao_star import RAOStarBaseline
from test_gurobi_ilp import _two_action_inputs


def test_external_rao_star_baseline_selects_safe_best_action():
    """Check the external RAO* baseline can choose a root action. / 检查外部 RAO* baseline 能选择根动作。"""
    runtime, interface, duration = _two_action_inputs()
    planner = RAOStarBaseline(risk_budget=0.0)

    decision = planner.choose_action(runtime, interface, duration, remaining_depth=runtime.horizon)

    assert decision.label == "go"
    assert decision.complete is True
    assert planner.last_stats is not None
    assert planner.last_stats.belief_nodes >= 1
    assert decision.timing["rao_star_belief_nodes"] >= 1.0
