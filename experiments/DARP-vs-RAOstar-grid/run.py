"""Run the paired DARP-HILP and external RAO* experiment."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any

from darp.adapter.loader import load_rddl
from darp.executor import PolicyExecutor
from darp.model.risk_sidecar import load_risk_sidecar
from darp.solve import DARPResult

from .darp_runner import (
    DOMAIN,
    RDDL_DIR,
    RISK,
    run_darp,
)
from .raostar_runner import (
    CONSTRAINED_POMDP_COMMIT,
    RAOSTAR_COMMIT,
    RAOStarRunner,
)

GRID_SIZES = (5, 100)
HORIZONS = (3, 4, 5, 6)
DELTAS = (0.1, 0.2, 0.3)
ALGORITHMS = ("DARP-HILP", "RAO*")
TRIALS = 25

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_CACHE = PROJECT_ROOT / ".cache" / "baselines"
DEFAULT_OUTPUT = EXPERIMENT_DIR / "output" / "table2-raw.csv"

FIELDS = (
    "size",
    "horizon",
    "delta",
    "algorithm",
    "trial",
    "seed",
    "objective",
    "risk",
    "time_s",
    "n",
    "iterations",
    "complete",
    "evaluation_episodes",
    "risk_rate",
    "physical_duration_mean",
    "policy_execution_time_s",
    "result_file",
    "constrained_pomdp_commit",
    "raostar_commit",
)


@dataclass(frozen=True, order=True, slots=True)
class Scenario:
    size: int
    horizon: int
    delta: float


@dataclass(frozen=True, slots=True)
class Case:
    scenario: Scenario
    instance: Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--constrained-pomdp-repo",
        type=Path,
        help="optional local Constrained-POMDP checkout",
    )
    parser.add_argument(
        "--raostar-checkout",
        type=Path,
        help="optional local RAOStar checkout",
    )
    parser.add_argument("--baseline-cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument(
        "--instance",
        type=Path,
        help="single checked-in RDDL instance; size/horizon are read from it",
    )
    parser.add_argument("--sizes", type=int, nargs="+")
    parser.add_argument("--horizons", type=int, nargs="+")
    parser.add_argument("--deltas", type=float, nargs="+")
    parser.add_argument("--trials", type=int, default=TRIALS)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--timeout", type=float, help="same search limit for both solvers")
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _validate_args(args)
    cases = _build_cases(args)
    raostar = RAOStarRunner.create(
        constrained_pomdp_repo=args.constrained_pomdp_repo,
        raostar_repo=args.raostar_checkout,
        cache_root=args.baseline_cache,
    )
    print(f"Constrained-POMDP: {raostar.constrained_pomdp_path}")
    print(f"RAOStar: {raostar.raostar_path}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    existing = _load_existing(args.output, args.seed, args.episodes) if args.resume else {}
    append = args.resume and args.output.is_file() and args.output.stat().st_size > 0
    with args.output.open("a" if append else "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        if not append:
            writer.writeheader()
        for case in cases:
            scenario = case.scenario
            for trial in range(1, args.trials + 1):
                if all(
                    _key(scenario, trial, algorithm) in existing
                    for algorithm in ALGORITHMS
                ):
                    continue
                key = _key(scenario, trial, "DARP-HILP")
                if key not in existing:
                    seed = args.seed + trial - 1
                    with closing(load_rddl(DOMAIN, case.instance).env) as env:
                        result = run_darp(
                            case.instance,
                            delta=scenario.delta,
                            seed=seed,
                            timeout_s=args.timeout,
                        )
                        result_path = _result_path(args.output, scenario, trial)
                        result.save(result_path)
                        result = DARPResult.load(result_path)
                        agent = PolicyExecutor(result.decision.policy)
                        statistics = agent.evaluate(
                            env,
                            episodes=args.episodes,
                            seed=seed,
                            risk_path=RISK,
                        )
                    metrics = _darp_metrics(
                        result,
                        statistics,
                        result_path.relative_to(args.output.parent).as_posix(),
                    )
                    _save(
                        writer,
                        handle,
                        existing,
                        scenario,
                        trial,
                        seed,
                        "DARP-HILP",
                        metrics,
                    )

                key = _key(scenario, trial, "RAO*")
                if key not in existing:
                    seed = args.seed + trial - 1
                    grid = raostar.make_grid(
                        scenario.size, scenario.horizon, scenario.delta
                    )
                    metrics = raostar.run(grid, timeout_s=args.timeout)
                    _save(
                        writer,
                        handle,
                        existing,
                        scenario,
                        trial,
                        seed,
                        "RAO*",
                        metrics,
                    )

    summary = args.summary or args.output.with_suffix(".md")
    _write_summary(
        args.output,
        summary,
        expected_scenarios=tuple(case.scenario for case in cases),
        expected_trials=args.trials,
    )
    print(f"summary: {summary}")
    return 0


def _build_cases(args: argparse.Namespace) -> tuple[Case, ...]:
    if args.instance is not None:
        instance = args.instance.expanduser().resolve()
        size, horizon = _read_instance(instance)
        canonical = _instance_path(size, horizon).resolve()
        if instance != canonical:
            raise ValueError(f"Expected checked-in instance {canonical}")
        return (Case(Scenario(size, horizon, _default_risk_budget()), instance),)

    sizes = args.sizes or GRID_SIZES
    horizons = args.horizons or HORIZONS
    deltas = args.deltas or DELTAS
    cases: list[Case] = []
    for selected_size in sizes:
        for selected_horizon in horizons:
            instance = _instance_path(selected_size, selected_horizon).resolve()
            if _read_instance(instance) != (selected_size, selected_horizon):
                raise ValueError(f"RDDL metadata mismatch: {instance}")
            cases.extend(
                Case(Scenario(selected_size, selected_horizon, delta), instance)
                for delta in deltas
            )
    return tuple(cases)


def _instance_path(size: int, horizon: int) -> Path:
    path = RDDL_DIR / f"instance_{size}_h{horizon}.rddl"
    if not path.is_file():
        raise ValueError(f"Missing Grid instance: {path}")
    return path


def _read_instance(instance: Path) -> tuple[int, int]:
    model = load_rddl(DOMAIN, instance).env.model
    non_fluents = model.non_fluents
    rows = int(non_fluents["max_row"]) + 1
    columns = int(non_fluents["max_col"]) + 1
    if rows != columns:
        raise ValueError("The RAO* comparison requires a square Grid.")
    expected_state = {
        "grid_row": rows - 1,
        "grid_col": 0,
        "row_mod5": (rows - 1) % 5,
        "col_mod5": 0,
    }
    if (
        (int(non_fluents["goal_row"]), int(non_fluents["goal_col"]))
        != (0, columns - 1)
        or float(non_fluents["transition_accuracy"]) != 0.85
        or float(non_fluents["observation_accuracy"]) != 0.85
        or {name: int(model.state_fluents[name]) for name in expected_state}
        != expected_state
        or int(model.max_allowed_actions) != 1
        or float(model.discount) != 1.0
    ):
        raise ValueError(f"RDDL does not match the paper Grid: {instance}")
    return rows, int(model.horizon)


def _default_risk_budget() -> float:
    budget = load_risk_sidecar(RISK).budget
    if budget is None:
        raise ValueError("risk.json must define a default budget.")
    return float(budget)


def _save(
    writer: csv.DictWriter,
    handle: Any,
    existing: dict[tuple[int, int, float, int, str], dict[str, Any]],
    scenario: Scenario,
    trial: int,
    seed: int,
    algorithm: str,
    metrics: dict[str, Any],
) -> None:
    row = {
        "size": scenario.size,
        "horizon": scenario.horizon,
        "delta": scenario.delta,
        "algorithm": algorithm,
        "trial": trial,
        "seed": seed,
        **metrics,
        "constrained_pomdp_commit": CONSTRAINED_POMDP_COMMIT,
        "raostar_commit": RAOSTAR_COMMIT,
    }
    writer.writerow(row)
    handle.flush()
    existing[_key(scenario, trial, algorithm)] = row
    execution_time = metrics.get("policy_execution_time_s")
    execution = (
        f", exec={float(execution_time):.6f}s"
        if execution_time not in (None, "")
        else ""
    )
    print(
        f"{scenario.size}x{scenario.size} h={scenario.horizon} "
        f"delta={scenario.delta:.1f} trial={trial} {algorithm}: "
        f"obj={metrics['objective']:.6f}, risk={metrics['risk']:.6f}, "
        f"time={metrics['time_s']:.3f}s, n={metrics['n']}, "
        f"iter={metrics['iterations']}{execution}"
    )


def _darp_metrics(
    result: DARPResult,
    statistics: dict[str, float],
    result_file: str,
) -> dict[str, Any]:
    decision = result.decision
    utility = decision.policy.achieved_utility
    risk = decision.policy.active_constraint_value
    if utility is None or risk is None:
        raise RuntimeError("DARP policy is missing objective or risk metrics.")
    return {
        "objective": -float(utility),
        "risk": float(risk),
        "time_s": result.elapsed_s,
        "n": int(
            decision.timing["expanded_nodes"]
            + decision.timing["frontier_nodes"]
        ),
        "iterations": int(decision.timing["partial_ilp_solves"]),
        "complete": True,
        "evaluation_episodes": int(statistics["episodes"]),
        "risk_rate": statistics["risk_rate"],
        "physical_duration_mean": statistics["physical_duration_mean"],
        "policy_execution_time_s": statistics["rollout_time_s"] / statistics["episodes"],
        "result_file": result_file,
    }


def _result_path(output: Path, scenario: Scenario, trial: int) -> Path:
    delta = str(scenario.delta).replace(".", "p")
    name = (
        f"darp-{scenario.size}x{scenario.size}-h{scenario.horizon}-"
        f"d{delta}-trial{trial:02d}.json"
    )
    return output.parent / "results" / output.stem / name


def _load_existing(
    path: Path,
    base_seed: int,
    episodes: int,
) -> dict[tuple[int, int, float, int, str], dict[str, Any]]:
    if not path.is_file() or path.stat().st_size == 0:
        return {}
    rows: dict[tuple[int, int, float, int, str], dict[str, Any]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != FIELDS:
            raise ValueError(f"Cannot resume CSV with a different schema: {path}")
        for row in reader:
            trial = int(row["trial"])
            if int(row["seed"]) != base_seed + trial - 1:
                raise ValueError("Resume CSV uses a different trial seed")
            if row["algorithm"] not in ALGORITHMS:
                raise ValueError(f"Unexpected algorithm in {path}: {row['algorithm']}")
            if row["constrained_pomdp_commit"] != CONSTRAINED_POMDP_COMMIT:
                raise ValueError("Resume CSV uses a different Constrained-POMDP commit")
            if row["raostar_commit"] != RAOSTAR_COMMIT:
                raise ValueError("Resume CSV uses a different RAOStar commit")
            if row["complete"].lower() != "true":
                raise ValueError("Resume CSV contains an incomplete search")
            if row["algorithm"] == "DARP-HILP":
                if not row["evaluation_episodes"] or int(row["evaluation_episodes"]) != episodes:
                    raise ValueError("Resume CSV uses a different evaluation episode count")
                if not row["policy_execution_time_s"] or not row["result_file"]:
                    raise ValueError("Resume CSV is missing a DARP policy execution")
                if not (path.parent / row["result_file"]).is_file():
                    raise ValueError(
                        f"Resume CSV references a missing result: {row['result_file']}"
                    )
            scenario = Scenario(int(row["size"]), int(row["horizon"]), float(row["delta"]))
            key = _key(scenario, trial, row["algorithm"])
            if key in rows:
                raise ValueError(f"Duplicate result row: {key}")
            rows[key] = row
    return rows


def _write_summary(
    csv_path: Path,
    output: Path,
    *,
    expected_scenarios: tuple[Scenario, ...],
    expected_trials: int,
) -> None:
    groups: dict[Scenario, dict[str, list[dict[str, str]]]] = defaultdict(dict)
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != FIELDS:
            raise ValueError(f"Cannot summarize CSV with a different schema: {csv_path}")
        seen: set[tuple[Scenario, str, int]] = set()
        for row in reader:
            scenario = Scenario(int(row["size"]), int(row["horizon"]), float(row["delta"]))
            algorithm = row["algorithm"]
            trial = int(row["trial"])
            key = scenario, algorithm, trial
            if key in seen:
                raise ValueError(f"Duplicate summary row: {key}")
            seen.add(key)
            if algorithm not in ALGORITHMS:
                raise ValueError(f"Unexpected algorithm in {csv_path}: {algorithm}")
            if row["complete"].lower() != "true":
                raise ValueError(f"Incomplete result in {csv_path}: {key}")
            if row["constrained_pomdp_commit"] != CONSTRAINED_POMDP_COMMIT:
                raise ValueError("Summary CSV uses a different Constrained-POMDP commit")
            if row["raostar_commit"] != RAOSTAR_COMMIT:
                raise ValueError("Summary CSV uses a different RAOStar commit")
            if algorithm == "DARP-HILP" and (
                not row["policy_execution_time_s"] or not row["result_file"]
            ):
                raise ValueError(f"Missing DARP policy execution in {csv_path}: {key}")
            groups[scenario].setdefault(algorithm, []).append(row)

    expected = set(expected_scenarios)
    if set(groups) != expected:
        raise ValueError(
            "Summary scenario set is incomplete: "
            f"missing={sorted(expected - set(groups))}, "
            f"unexpected={sorted(set(groups) - expected)}"
        )
    required_trials = set(range(1, expected_trials + 1))
    for scenario in expected_scenarios:
        rows_by_algorithm = groups[scenario]
        if set(rows_by_algorithm) != set(ALGORITHMS):
            raise ValueError(f"Missing paired algorithms for {scenario}")
        for algorithm in ALGORITHMS:
            trial_ids = {int(row["trial"]) for row in rows_by_algorithm[algorithm]}
            if trial_ids != required_trials:
                raise ValueError(
                    f"Incomplete trials for {scenario} {algorithm}: "
                    f"missing={sorted(required_trials - trial_ids)}, "
                    f"unexpected={sorted(trial_ids - required_trials)}"
                )

    lines = [
        "# Table 2: Simulation results with heuristics",
        "",
        (
            "Each cell is one completed trial, not a 25-trial mean; time is planner wall-clock seconds."
            if expected_trials == 1 else
            f"Arithmetic means over {expected_trials} completed trials; time is planner wall-clock seconds."
        ),
        "Objective values are solver-native.",
        "",
        "| Problem | h | Δ | DARP-HILP Obj. | DARP-HILP Time (s) | "
        "DARP-HILP n | DARP-HILP Iter. | RAO* Obj. | RAO* Time (s) | "
        "RAO* n | RAO* Iter. |",
        "|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for scenario in sorted(expected):
        rows_by_algorithm = groups[scenario]
        darp = rows_by_algorithm["DARP-HILP"]
        raostar = rows_by_algorithm["RAO*"]
        lines.append(
            f"| {scenario.size}×{scenario.size} | {scenario.horizon} | "
            f"{scenario.delta:.1f} | {_mean(darp, 'objective'):.2f} | "
            f"{_mean(darp, 'time_s'):.2f} | {_mean(darp, 'n'):.0f} | "
            f"{_mean(darp, 'iterations'):.0f} | "
            f"{_mean(raostar, 'objective'):.2f} | "
            f"{_mean(raostar, 'time_s'):.2f} | {_mean(raostar, 'n'):.0f} | "
            f"{_mean(raostar, 'iterations'):.0f} |"
        )
    lines.extend(
        [
            "",
            "## Solver-reported risk",
            "",
            "Risk values come directly from each solver; no model recomputation is performed.",
            "",
            "| Problem | h | Δ | DARP-HILP Risk | RAO* Risk |",
            "|:--|--:|--:|--:|--:|",
        ]
    )
    for scenario in sorted(expected):
        rows_by_algorithm = groups[scenario]
        darp = rows_by_algorithm["DARP-HILP"]
        raostar = rows_by_algorithm["RAO*"]
        lines.append(
            f"| {scenario.size}×{scenario.size} | {scenario.horizon} | "
            f"{scenario.delta:.1f} | {_mean(darp, 'risk'):.6f} | "
            f"{_mean(raostar, 'risk'):.6f} |"
        )
    lines.extend(
        [
            "",
            "## DARP saved-policy execution",
            "",
            "Each saved policy is reloaded and executed in pyRDDLGym. Rows report first-entry risk frequency, mean physical duration and wall-clock time per episode. RAO* reports search metrics only.",
            "",
            "| Case / trial | Episodes | Risk frequency | Mean duration | s/episode |",
            "|:--|--:|--:|--:|--:|",
        ]
    )
    for scenario in sorted(expected):
        for row in sorted(groups[scenario]["DARP-HILP"], key=lambda item: int(item["trial"])):
            lines.append(
                f"| {scenario.size}×{scenario.size} h={scenario.horizon} "
                f"Δ={scenario.delta:.1f} / {row['trial']} | "
                f"{row['evaluation_episodes']} | "
                f"{float(row['risk_rate']):.4f} | "
                f"{float(row['physical_duration_mean']):.3f} | "
                f"{float(row['policy_execution_time_s']):.6f} |"
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _mean(rows: list[dict[str, str]], field: str) -> float:
    return fmean(float(row[field]) for row in rows)


def _validate_args(args: argparse.Namespace) -> None:
    if args.trials < 1:
        raise ValueError("--trials must be positive")
    if args.episodes < 1:
        raise ValueError("--episodes must be positive")
    if args.timeout is not None and args.timeout <= 0:
        raise ValueError("--timeout must be positive")
    if args.summary is not None and args.summary.resolve() == args.output.resolve():
        raise ValueError("--summary and --output must be different files")
    if args.instance is not None and any(
        value is not None for value in (args.sizes, args.horizons, args.deltas)
    ):
        raise ValueError("--instance cannot be combined with matrix filters")
    if args.sizes is not None and set(args.sizes) - set(GRID_SIZES):
        raise ValueError(f"--sizes must be drawn from {GRID_SIZES}")
    if args.horizons is not None and set(args.horizons) - set(HORIZONS):
        raise ValueError(f"--horizons must be drawn from {HORIZONS}")
    if args.deltas is not None and any(
        not any(abs(delta - expected) < 1e-12 for expected in DELTAS)
        for delta in args.deltas
    ):
        raise ValueError(f"--deltas must be drawn from {DELTAS}")
    for name in ("sizes", "horizons", "deltas"):
        values = getattr(args, name)
        if values is not None and len(values) != len(set(values)):
            raise ValueError(f"--{name} must not contain duplicates")


def _key(
    scenario: Scenario, trial: int, algorithm: str
) -> tuple[int, int, float, int, str]:
    return scenario.size, scenario.horizon, scenario.delta, trial, algorithm


if __name__ == "__main__":
    raise SystemExit(main())
