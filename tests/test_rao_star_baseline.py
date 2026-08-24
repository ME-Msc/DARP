"""Tests for the external RAO* comparison baseline."""

import pytest

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
    assert planner.last_stats.terminal_heuristic_used is False
    assert decision.timing["rao_star_belief_nodes"] >= 1.0
    assert decision.timing["rao_star_terminal_heuristic_used"] == 0.0


def test_external_rao_star_uses_optional_terminal_utility_hook():
    """Check a domain tail estimate is backed up and marked heuristic.

    / 检查 domain 尾部估值会参与 backup 并标为 heuristic。
    """
    runtime, interface, duration = _two_action_inputs()
    received_budgets: list[float | None] = []

    def terminal_utility(belief, risk_budget):
        received_budgets.append(risk_budget)
        at_goal_probability = sum(
            probability
            for state, probability in belief.items()
            if bool(dict(state).get("at_goal"))
        )
        return 10.0 * at_goal_probability

    interface.exact_kernel.terminal_utility = terminal_utility
    planner = RAOStarBaseline(risk_budget=0.25)

    decision = planner.choose_action(runtime, interface, duration, remaining_depth=runtime.horizon)

    assert decision.label == "go"
    assert decision.value == pytest.approx(15.0)
    assert decision.complete is False
    assert received_budgets and set(received_budgets) == {0.25}
    assert planner.last_stats is not None
    assert planner.last_stats.terminal_heuristic_used is True
    assert planner.last_stats.terminal_heuristic_evaluations >= 1
    assert decision.timing["rao_star_terminal_heuristic_used"] == 1.0
    assert decision.timing["rao_star_terminal_heuristic_evaluations"] >= 1.0


def test_external_rao_star_rejects_non_finite_terminal_utility():
    """Check an invalid domain heuristic fails explicitly. / 检查非法 domain heuristic 会明确报错。"""
    runtime, interface, duration = _two_action_inputs()
    interface.exact_kernel.terminal_utility = lambda belief, risk_budget: float("nan")

    with pytest.raises(ValueError, match="terminal utility heuristic must be a finite number"):
        RAOStarBaseline().choose_action(
            runtime,
            interface,
            duration,
            remaining_depth=runtime.horizon,
        )
