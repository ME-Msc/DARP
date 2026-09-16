"""Run the Table 1 grid matrix with DARP-HILP and DARP full-ILP.

The grid domain and instances are local to this experiment's rddl directory.
The raw CSV is checkpointed after every trial, so ``--resume`` can continue an
interrupted run.  As in the paper, horizon 6 runs HILP only.
"""

from __future__ import annotations

import argparse
import csv
import gc
from itertools import product
from pathlib import Path
from statistics import fmean
from typing import Any

from darp.adapter.loader import load_rddl
from darp.executor import PolicyExecutor
from darp.planning.heuristic import HeuristicInput, UtilityHeuristic
from darp.solve import DARPResult, solve_rddl

EXPERIMENT_DIR = Path(__file__).resolve().parent
RDDL_DIR = EXPERIMENT_DIR / "rddl"
DOMAIN = RDDL_DIR / "domain.rddl"
DEFAULT_OUTPUT = EXPERIMENT_DIR / "output" / "table1-raw.csv"

MODELS = ("F", "E", "S")
HORIZONS = (3, 4, 5, 6)
DELTAS = (0.1, 0.2, 0.3)
PLANNERS = ("full-ilp", "hilp")

FIELDS = (
    "model",
    "horizon",
    "delta",
    "planner",
    "trial",
    "seed",
    "status",
    "objective",
    "time_s",
    "risk",
    "solver_status",
    "expanded_nodes",
    "frontier_nodes",
    "tree_nodes",
    "ilp_variables",
    "ilp_constraints",
    "iterations",
    "evaluation_episodes",
    "risk_rate",
    "physical_duration_mean",
    "policy_execution_time_s",
    "result_file",
    "error",
)


def _manhattan(value: HeuristicInput) -> float:
    """Paper heuristic: optimistic remaining cost, in reward sign convention."""

    row = int(value.state["grid_row"])
    col = int(value.state["grid_col"])
    goal_row = int(value.non_fluents["goal_row"])
    goal_col = int(value.non_fluents["goal_col"])
    return -float(abs(row - goal_row) + abs(col - goal_col))


MANHATTAN = UtilityHeuristic(
    name="table1-grid-manhattan",
    evaluate=_manhattan,
    upper_bound=True,
)


def _instance(model: str, horizon: int) -> Path:
    return RDDL_DIR / f"instance_{model.lower()}_h{horizon}.rddl"


def _key(row: dict[str, str]) -> tuple[str, int, float, str, int]:
    return (
        row["model"],
        int(row["horizon"]),
        float(row["delta"]),
        row["planner"],
        int(row["trial"]),
    )


def _read_rows(path: Path, episodes: int) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != FIELDS:
            raise ValueError(f"Unexpected CSV schema in {path}")
        rows = list(reader)
    for row in rows:
        if not row["evaluation_episodes"] or int(row["evaluation_episodes"]) != episodes:
            raise ValueError("Resume CSV uses a different evaluation episode count.")
        if row["status"] == "ok" and (
            not row["policy_execution_time_s"]
            or not row["result_file"]
            or not (path.parent / row["result_file"]).is_file()
        ):
            raise ValueError("Resume CSV is missing a DARP result or execution.")
    return rows


def _run_trial(
    model: str,
    horizon: int,
    delta: float,
    planner: str,
    trial: int,
    seed: int,
    episodes: int,
    timeout_s: float | None,
    result_path: Path,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "planner": planner,
        "risk_budget": delta,
        "seed": seed,
        "timeout_s": timeout_s,
    }
    kwargs.update(heuristic=MANHATTAN, terminal_heuristic=True)
    if planner == "full-ilp":
        kwargs["full_ilp_max_tree_nodes"] = 800_000

    # Duration is part of the RDDL instance/domain extension, so there is no
    # duration sidecar argument here.
    instance = _instance(model, horizon)
    env = load_rddl(DOMAIN, instance).env
    try:
        result = solve_rddl(DOMAIN, instance, **kwargs)
        result.save(result_path)
        result = DARPResult.load(result_path)
        decision = result.decision
        timing = decision.timing
        policy = decision.policy
        if not decision.complete or policy.feasible is not True:
            raise RuntimeError(
                "DARP did not return a complete feasible policy: "
                f"status={policy.solver_status}"
            )
        utility = policy.achieved_utility
        risk = policy.active_constraint_value
        if utility is None or risk is None:
            raise RuntimeError("DARP policy is missing objective or risk metrics")
        agent = PolicyExecutor(policy)
        statistics = agent.evaluate(
            env,
            episodes=episodes,
            seed=seed,
        )
        objective = -float(utility)

        return {
            "model": model,
            "horizon": horizon,
            "delta": delta,
            "planner": planner,
            "trial": trial,
            "seed": seed,
            "status": "ok",
            "objective": objective,
            "time_s": result.elapsed_s,
            "risk": risk,
            "solver_status": policy.solver_status,
            "expanded_nodes": timing.get("expanded_nodes", ""),
            "frontier_nodes": timing.get("frontier_nodes", ""),
            "tree_nodes": timing.get("tree_nodes", ""),
            "ilp_variables": timing.get("ilp_variables", ""),
            "ilp_constraints": timing.get("ilp_constraints", ""),
            "iterations": timing.get("partial_ilp_solves", 1),
            "evaluation_episodes": episodes,
            "risk_rate": statistics["risk_rate"],
            "physical_duration_mean": statistics["physical_duration_mean"],
            "policy_execution_time_s": statistics["rollout_time_s"] / episodes,
            "result_file": "",
            "error": "",
        }
    finally:
        env.close()


def _result_path(
    output: Path,
    model: str,
    horizon: int,
    delta: float,
    planner: str,
    trial: int,
) -> Path:
    delta_label = str(delta).replace(".", "p")
    name = (
        f"darp-{model.lower()}-h{horizon}-d{delta_label}-"
        f"{planner}-trial{trial:02d}.json"
    )
    return output.parent / "results" / output.stem / name


def _error_row(
    model: str,
    horizon: int,
    delta: float,
    planner: str,
    trial: int,
    seed: int,
    error: Exception,
) -> dict[str, Any]:
    row = {field: "" for field in FIELDS}
    row.update(
        model=model,
        horizon=horizon,
        delta=delta,
        planner=planner,
        trial=trial,
        seed=seed,
        status="error",
        error=f"{type(error).__name__}: {error}",
    )
    return row


def _number(
    grouped: dict[tuple[int, float, str, str], list[dict[str, str]]],
    horizon: int,
    delta: float,
    model: str,
    planner: str,
    field: str,
) -> float | None:
    values = [
        float(row[field])
        for row in grouped.get((horizon, delta, model, planner), ())
        if row["status"] == "ok" and row[field] != ""
    ]
    return fmean(values) if values else None


def _format(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "—"
    return f"{value:.{digits}f}"


def _write_markdown(rows: list[dict[str, str]], output: Path) -> None:
    grouped: dict[tuple[int, float, str, str], list[dict[str, str]]] = {}
    for row in rows:
        key = (
            int(row["horizon"]),
            float(row["delta"]),
            row["model"],
            row["planner"],
        )
        grouped.setdefault(key, []).append(row)

    horizons = sorted({key[0] for key in grouped})
    deltas = sorted({key[1] for key in grouped})
    successful_counts = sorted({
        sum(row["status"] == "ok" for row in trials)
        for trials in grouped.values()
    })
    header = ["h", "Δ"]
    for metric in ("Obj", "Time", "n", "Act.n"):
        header.extend(f"Full-ILP {metric} {model}" for model in MODELS)
    for metric in ("Obj", "Time", "Exp.n", "Exp.%"):
        header.extend(f"DARP-HILP {metric} {model}" for model in MODELS)

    lines = [
        "# DARP Table 1 grid experiment",
        "",
        "Values are means over successful trials; `Exp.% = HILP Exp.n / Full-ILP Act.n`.",
        f"Successful trials per configuration: {', '.join(map(str, successful_counts))}. Single-trial values are individual runs, not 25-trial means.",
        "E/S use source-or-intended mud contact; S uses `Normal(mean, variance=0.1)` and `varsigma=0.3`.",
        "The paper does not publish its E/S artifact, so these are auditable DARP results rather than copied reference output.",
        "`—` means that the configuration has not produced a successful row in the raw CSV.",
        "",
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---:"] * len(header)) + " |",
    ]
    for horizon, delta in product(horizons, deltas):
        cells = [str(horizon), f"{delta:.1f}"]
        for field, digits in (
            ("objective", 2),
            ("time_s", 2),
            ("tree_nodes", 0),
            ("ilp_variables", 0),
        ):
            cells.extend(
                _format(
                    _number(grouped, horizon, delta, model, "full-ilp", field),
                    digits,
                )
                for model in MODELS
            )
        for field, digits in (
            ("objective", 2),
            ("time_s", 2),
            ("ilp_variables", 0),
        ):
            cells.extend(
                _format(
                    _number(grouped, horizon, delta, model, "hilp", field),
                    digits,
                )
                for model in MODELS
            )
        for model in MODELS:
            expanded = _number(
                grouped, horizon, delta, model, "hilp", "ilp_variables"
            )
            action_n = _number(
                grouped, horizon, delta, model, "full-ilp", "ilp_variables"
            )
            ratio = None if expanded is None or not action_n else 100 * expanded / action_n
            cells.append(_format(ratio, 1))
        lines.append("| " + " | ".join(cells) + " |")

    lines.extend(
        [
            "",
            "## Saved-policy execution",
            "",
            "Every policy is saved, reloaded and executed. Rows are individual trials; risk is the fraction of episodes that enter a risky state at least once. Physical duration and execution wall-clock time are reported separately.",
            "",
            "| Case / trial | Planner | Episodes | Risk frequency | Mean physical duration | s/episode |",
            "|:--|:--|--:|--:|--:|--:|",
        ]
    )
    for row in sorted(rows, key=_key):
        if row["status"] != "ok":
            continue
        lines.append(
            f"| {row['model']} h={row['horizon']} Δ={row['delta']} / {row['trial']} | "
            f"{row['planner']} | {row['evaluation_episodes']} | "
            f"{float(row['risk_rate']):.4f} | "
            f"{float(row['physical_duration_mean']):.3f} | "
            f"{float(row['policy_execution_time_s']):.6f} |"
        )

    failures = sum(row["status"] != "ok" for row in rows)
    lines.extend(("", f"Failed trials recorded in raw CSV: {failures}.", ""))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    parser.add_argument("--horizons", nargs="+", type=int, choices=HORIZONS, default=HORIZONS)
    parser.add_argument("--deltas", nargs="+", type=float, choices=DELTAS, default=DELTAS)
    parser.add_argument("--planners", nargs="+", choices=PLANNERS, default=PLANNERS)
    parser.add_argument("--trials", type=int, default=25)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--timeout", type=float, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run F/E/S, h=3, delta=0.1 once with both planners.",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.trials <= 0:
        raise SystemExit("--trials must be positive")
    if args.episodes <= 0:
        raise SystemExit("--episodes must be positive")
    if args.timeout is not None and args.timeout <= 0:
        raise SystemExit("--timeout must be positive")

    models = MODELS if args.smoke else tuple(dict.fromkeys(args.models))
    horizons = (3,) if args.smoke else tuple(dict.fromkeys(args.horizons))
    deltas = (0.1,) if args.smoke else tuple(dict.fromkeys(args.deltas))
    planners = PLANNERS if args.smoke else tuple(dict.fromkeys(args.planners))
    trials = 1 if args.smoke else args.trials
    output = args.output
    if args.smoke and output == DEFAULT_OUTPUT:
        output = DEFAULT_OUTPUT.with_name("smoke-raw.csv")
    summary = args.summary or output.with_suffix(".md")
    if summary.resolve() == output.resolve():
        raise SystemExit("--summary and --output must be different files")

    rows = _read_rows(output, args.episodes) if args.resume else []
    completed = {_key(row) for row in rows}
    output.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.resume and output.exists() else "w"
    failures = sum(row["status"] != "ok" for row in rows)
    with output.open(mode, newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        if mode == "w":
            writer.writeheader()
        for model, horizon, delta, trial, planner in product(
            models, horizons, deltas, range(1, trials + 1), planners
        ):
            if planner == "full-ilp" and horizon == 6:
                continue
            key = (model, horizon, delta, planner, trial)
            if key in completed:
                continue
            seed = args.seed + trial - 1
            result_path = _result_path(
                output, model, horizon, delta, planner, trial
            )
            try:
                row = _run_trial(
                    model,
                    horizon,
                    delta,
                    planner,
                    trial,
                    seed,
                    args.episodes,
                    args.timeout,
                    result_path,
                )
                row["result_file"] = result_path.relative_to(output.parent).as_posix()
            except Exception as error:  # Keep the rest of the long matrix runnable.
                row = _error_row(
                    model, horizon, delta, planner, trial, seed, error
                )
                failures += 1
            row["evaluation_episodes"] = args.episodes
            writer.writerow(row)
            stream.flush()
            rows.append({field: str(row[field]) for field in FIELDS})
            print(
                f"{model} h={horizon} Δ={delta:.1f} {planner} "
                f"trial={trial}: {row['status']}", flush=True,
            )
            # Release large full-tree cycles between independent timed solves.
            gc.collect()

    _write_markdown(rows, summary)
    print(f"raw: {output}")
    print(f"table: {summary}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
