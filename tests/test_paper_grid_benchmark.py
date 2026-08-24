"""Focused tests for the self-contained paper-grid experiment fixture."""

from __future__ import annotations

import csv
from types import SimpleNamespace

import pytest

from experiments.baselines.rao_star import RAOStarBaseline
from experiments.benchmarks.paper_grid import (
    PAPER_GOAL,
    PAPER_MUDDY_CELLS,
    PAPER_RISKY_CELLS,
    PAPER_START,
    PaperGridKernel,
    build_paper_grid_problem,
)
from experiments.scripts.run_paper_grid import (
    CSV_FIELDS,
    SUMMARY_FIELDS,
    ExperimentConfig,
    run_case,
    summarize_rows,
    write_csv,
    write_summary_csv,
)
import experiments.scripts.run_paper_grid as paper_grid_runner


def test_paper_grid_coordinates_match_published_instance():
    assert PAPER_START == (5, 1)
    assert PAPER_GOAL == (1, 5)
    assert PAPER_RISKY_CELLS == {(1, 1), (2, 4), (2, 5), (4, 1), (4, 2)}
    assert PAPER_MUDDY_CELLS == {(1, 4), (2, 2), (3, 3), (4, 5), (5, 3)}


def test_transition_aggregates_slips_that_hit_a_boundary():
    kernel = PaperGridKernel()

    transition = kernel.transition_distribution({"row": 5, "col": 1}, {"move": "S"})

    by_cell = {(dict(state)["row"], dict(state)["col"]): probability for state, probability in transition.items()}
    assert by_cell == pytest.approx({(5, 1): 0.925, (5, 2): 0.075})
    assert sum(transition.values()) == pytest.approx(1.0)


def test_wall_count_observation_has_paper_noise_model():
    kernel = PaperGridKernel()
    corner = kernel._key_from_cell((5, 1))
    interior = kernel._key_from_cell((3, 3))

    assert kernel.observation_probability((("walls", 2),), corner, {"move": "N"}) == pytest.approx(0.85)
    assert kernel.observation_probability((("walls", 0),), corner, {"move": "N"}) == pytest.approx(0.075)
    assert kernel.observation_probability((("walls", 0),), interior, {"move": "N"}) == pytest.approx(0.85)


def test_safe_expansion_separates_risk_from_surviving_belief():
    kernel = PaperGridKernel()
    belief = kernel.initial_belief_from_model()

    ordinary = kernel.expand_action(belief, {"move": "N"})
    safe = kernel.expand_safe_action(belief, {"move": "N"})

    assert ordinary.risk == pytest.approx(0.85)
    assert safe.risk == pytest.approx(0.85)
    assert safe.survival_probability == pytest.approx(0.15)
    assert all(kernel._cell_from_key(state) not in PAPER_RISKY_CELLS for state in safe.prior_belief)
    assert sum(safe.prior_belief.values()) == pytest.approx(1.0)
    assert sum(outcome.probability for outcome in safe.observations) == pytest.approx(1.0)


def test_terminal_utility_is_negative_expected_manhattan_distance():
    kernel = PaperGridKernel()
    start_belief = kernel.initial_belief_from_model()
    east_prior = kernel.expand_action(start_belief, {"move": "E"}).prior_belief

    assert kernel.terminal_utility(start_belief) == pytest.approx(-8.0)
    assert kernel.terminal_utility(start_belief, 0.1) == pytest.approx(-7.2)
    assert kernel.terminal_utility(east_prior) == pytest.approx(-7.075)


def test_problem_builder_is_rao_star_compatible():
    runtime, interface, duration = build_paper_grid_problem(1)
    planner = RAOStarBaseline(risk_budget=0.1)

    decision = planner.choose_action(runtime, interface, duration, remaining_depth=1)

    assert decision.label in {"E", "S", "W"}
    assert planner.last_stats is not None
    assert planner.last_stats.selected_execution_risk is not None
    assert planner.last_stats.selected_execution_risk <= 0.1


def test_problem_builder_can_disable_terminal_tail_for_exact_prefix_checks():
    runtime, interface, duration = build_paper_grid_problem(
        1,
        terminal_utility_enabled=False,
    )

    decision = RAOStarBaseline(risk_budget=0.1).choose_action(
        runtime,
        interface,
        duration,
        remaining_depth=1,
    )

    assert decision.complete is True
    assert decision.value == pytest.approx(-1.0)


def test_runner_guards_full_ilp_and_writes_stable_tidy_schema(tmp_path):
    config = ExperimentConfig(
        horizons=(4,),
        risk_budgets=(0.1,),
        algorithms=("hilp",),
        include_full_ilp=True,
        full_ilp_max_horizon=3,
    )
    row = run_case(
        algorithm="full-ilp",
        horizon=4,
        risk_budget=0.1,
        repetition=1,
        config=config,
    )
    output = write_csv([row], tmp_path / "results.csv")

    with output.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
    assert tuple(reader.fieldnames or ()) == CSV_FIELDS
    assert rows[0]["status"] == "skipped-horizon-limit"
    assert rows[0]["algorithm"] == "full-ilp"


@pytest.mark.parametrize(
    "overrides, message",
    (
        ({"timeout_seconds": float("inf")}, "timeout_seconds"),
        ({"timeout_seconds": float("nan")}, "timeout_seconds"),
        ({"muddy_step_cost": 0.0}, "muddy_step_cost"),
        ({"muddy_step_cost": float("nan")}, "muddy_step_cost"),
    ),
)
def test_experiment_config_rejects_invalid_numeric_controls(overrides, message):
    """Check malformed CLI-equivalent controls fail before a matrix starts."""
    with pytest.raises(ValueError, match=message):
        ExperimentConfig(**overrides)


def test_runner_marks_capped_hilp_refinement_as_partial(monkeypatch):
    """A terminal tail plus an unexpanded incumbent must not look converged."""
    config = ExperimentConfig(
        horizons=(3,),
        risk_budgets=(0.1,),
        algorithms=("hilp",),
        expansion_rounds=0,
    )

    fake_planner = SimpleNamespace(
        choose_action=lambda *args, **kwargs: SimpleNamespace(
            timing={
                "used_terminal_heuristic": 1.0,
                "frontier_refinement_exhausted": 0.0,
                "planner_elapsed_ms": 1.0,
            },
            complete=False,
            label="E",
            value=-1.0,
        )
    )
    monkeypatch.setattr(paper_grid_runner, "_planner", lambda *args, **kwargs: fake_planner)

    row = run_case(
        algorithm="hilp",
        horizon=3,
        risk_budget=0.1,
        repetition=1,
        config=config,
    )

    assert row["status"] == "heuristic-partial"
    assert row["solution_kind"] == "admissible-tail-partial-refinement"


def test_runner_marks_time_limited_incumbent_as_timeout(monkeypatch):
    """Check a feasible Gurobi incumbent still preserves timeout status."""
    config = ExperimentConfig(
        horizons=(2,),
        risk_budgets=(0.1,),
        algorithms=("hilp",),
    )
    fake_planner = SimpleNamespace(
        choose_action=lambda *args, **kwargs: SimpleNamespace(
            timing={"solver_time_limit_hit": 1.0, "planner_elapsed_ms": 60_000.0},
            complete=False,
            label="E",
            value=-2.0,
        )
    )
    monkeypatch.setattr(paper_grid_runner, "_planner", lambda *args, **kwargs: fake_planner)

    row = run_case(
        algorithm="hilp",
        horizon=2,
        risk_budget=0.1,
        repetition=1,
        config=config,
    )

    assert row["status"] == "timeout"
    assert row["solution_kind"] == "time-limited-incumbent"


def test_summary_reports_median_iqr_and_rao_speedup(tmp_path):
    rows = [
        {
            "benchmark": "paper-grid",
            "algorithm": algorithm,
            "horizon": 3,
            "risk_budget": 0.1,
            "status": "heuristic",
            "solution_kind": "admissible-tail",
            "selected_action": "E",
            "objective_cost": 9.0,
            "objective": -9.0,
            "risk": 0.09,
            "time_ms": time_ms,
            "nodes": nodes,
        }
        for algorithm, time_ms, nodes in (
            ("hilp", 10.0, 12),
            ("hilp", 20.0, 14),
            ("hilp", 30.0, 16),
            ("rao-star-style", 90.0, 100),
            ("rao-star-style", 100.0, 110),
            ("rao-star-style", 110.0, 120),
        )
    ]

    summary = summarize_rows(rows)
    output = write_summary_csv(summary, tmp_path / "summary.csv")
    hilp = next(row for row in summary if row["algorithm"] == "hilp")

    assert hilp["time_ms_median"] == pytest.approx(20.0)
    assert hilp["time_ms_q1"] == pytest.approx(15.0)
    assert hilp["time_ms_q3"] == pytest.approx(25.0)
    assert hilp["speedup_vs_rao_star_style"] == pytest.approx(5.0)
    with output.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        assert tuple(reader.fieldnames or ()) == SUMMARY_FIELDS


def test_summary_does_not_compare_exact_and_heuristic_objectives():
    """Full-ILP exact and terminal-tail values have different contracts."""
    rows = [
        {
            "benchmark": "paper-grid",
            "algorithm": "rao-star-style",
            "horizon": 2,
            "risk_budget": 0.1,
            "status": "heuristic",
            "solution_kind": "admissible-tail",
            "selected_action": "E",
            "objective_cost": 8.0,
            "objective": -8.0,
            "risk": 0.09,
            "time_ms": 10.0,
            "nodes": 100,
        },
        {
            "benchmark": "paper-grid",
            "algorithm": "full-ilp",
            "horizon": 2,
            "risk_budget": 0.1,
            "status": "ok",
            "solution_kind": "exact",
            "selected_action": "S",
            "objective_cost": 2.0,
            "objective": -2.0,
            "risk": 0.0,
            "time_ms": 5.0,
            "nodes": 50,
        },
    ]

    full = next(row for row in summarize_rows(rows) if row["algorithm"] == "full-ilp")

    assert full["speedup_vs_rao_star_style"] == ""
    assert full["objective_cost_gap_vs_rao_star_style"] == ""
