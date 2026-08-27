"""Run the paired DARP-HILP and external RAO* experiment."""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .darp_runner import (
    default_risk_budget,
    instance_path,
    read_instance,
    run_darp,
    warm_up,
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
DEFAULT_CACHE = PROJECT_ROOT / ".cache" / "baselines"
DEFAULT_OUTPUT = PROJECT_ROOT / "output" / "DARP-vs-RAOstar" / "raw.csv"

FIELDS = (
    "size",
    "horizon",
    "delta",
    "algorithm",
    "trial",
    "objective",
    "risk",
    "time_s",
    "n",
    "iterations",
    "complete",
    "certified",
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
    existing = _load_existing(args.output) if args.resume else {}
    if any(
        _key(case.scenario, trial, "DARP-HILP") not in existing
        for case in cases
        for trial in range(1, args.trials + 1)
    ):
        warm_up()
    append = args.resume and args.output.is_file() and args.output.stat().st_size > 0
    parity_checked: set[Path] = set()
    with args.output.open("a" if append else "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
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
                grid = raostar.make_grid(
                    scenario.size, scenario.horizon, scenario.delta
                )
                key = _key(scenario, trial, "DARP-HILP")
                if key not in existing:
                    check = case.instance.resolve() not in parity_checked
                    metrics = run_darp(
                        case.instance,
                        delta=scenario.delta,
                        seed=args.seed + trial - 1,
                        timeout_s=args.timeout,
                        reference_grid=grid if check else None,
                    )
                    if check:
                        parity_checked.add(case.instance.resolve())
                    _save(writer, handle, existing, scenario, trial, "DARP-HILP", metrics)

                key = _key(scenario, trial, "RAO*")
                if key not in existing:
                    metrics = raostar.run(grid, timeout_s=args.timeout)
                    _save(writer, handle, existing, scenario, trial, "RAO*", metrics)

    summary = args.summary or args.output.with_suffix(".md")
    _write_summary(args.output, summary)
    print(f"summary: {summary}")
    return 0


def _build_cases(args: argparse.Namespace) -> tuple[Case, ...]:
    if args.instance is not None:
        instance = args.instance.expanduser().resolve()
        size, horizon = read_instance(instance)
        canonical = instance_path(size, horizon).resolve()
        if instance != canonical:
            raise ValueError(f"Expected checked-in instance {canonical}")
        return (Case(Scenario(size, horizon, default_risk_budget()), instance),)

    sizes = args.sizes or GRID_SIZES
    horizons = args.horizons or HORIZONS
    deltas = args.deltas or DELTAS
    cases: list[Case] = []
    for selected_size in sizes:
        for selected_horizon in horizons:
            instance = instance_path(selected_size, selected_horizon).resolve()
            if read_instance(instance) != (selected_size, selected_horizon):
                raise ValueError(f"RDDL metadata mismatch: {instance}")
            cases.extend(
                Case(Scenario(selected_size, selected_horizon, delta), instance)
                for delta in deltas
            )
    return tuple(cases)


def _save(
    writer: csv.DictWriter,
    handle: Any,
    existing: dict[tuple[int, int, float, int, str], dict[str, Any]],
    scenario: Scenario,
    trial: int,
    algorithm: str,
    metrics: dict[str, Any],
) -> None:
    row = {
        "size": scenario.size,
        "horizon": scenario.horizon,
        "delta": scenario.delta,
        "algorithm": algorithm,
        "trial": trial,
        **metrics,
        "constrained_pomdp_commit": CONSTRAINED_POMDP_COMMIT,
        "raostar_commit": RAOSTAR_COMMIT,
    }
    writer.writerow(row)
    handle.flush()
    existing[_key(scenario, trial, algorithm)] = row
    print(
        f"{scenario.size}x{scenario.size} h={scenario.horizon} "
        f"delta={scenario.delta:.1f} trial={trial} {algorithm}: "
        f"obj={metrics['objective']:.6f}, risk={metrics['risk']:.6f}, "
        f"time={metrics['time_s']:.3f}s, n={metrics['n']}, "
        f"iter={metrics['iterations']}"
    )


def _load_existing(
    path: Path,
) -> dict[tuple[int, int, float, int, str], dict[str, Any]]:
    if not path.is_file() or path.stat().st_size == 0:
        return {}
    rows: dict[tuple[int, int, float, int, str], dict[str, Any]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != FIELDS:
            raise ValueError(f"Cannot resume CSV with a different schema: {path}")
        for row in reader:
            if row["algorithm"] not in ALGORITHMS:
                raise ValueError(f"Unexpected algorithm in {path}: {row['algorithm']}")
            if row["constrained_pomdp_commit"] != CONSTRAINED_POMDP_COMMIT:
                raise ValueError("Resume CSV uses a different Constrained-POMDP commit")
            if row["raostar_commit"] != RAOSTAR_COMMIT:
                raise ValueError("Resume CSV uses a different RAOStar commit")
            if row["complete"].lower() != "true":
                raise ValueError("Resume CSV contains an incomplete search")
            scenario = Scenario(int(row["size"]), int(row["horizon"]), float(row["delta"]))
            key = _key(scenario, int(row["trial"]), row["algorithm"])
            if key in rows:
                raise ValueError(f"Duplicate result row: {key}")
            rows[key] = row
    return rows


def _write_summary(csv_path: Path, output: Path) -> None:
    groups: dict[tuple[Scenario, str], list[dict[str, str]]] = defaultdict(list)
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            scenario = Scenario(int(row["size"]), int(row["horizon"]), float(row["delta"]))
            groups[(scenario, row["algorithm"])].append(row)
    order = {name: index for index, name in enumerate(ALGORITHMS)}
    lines = [
        "| Problem | h | delta | Algorithm | Trials | Native objective | "
        "Risk | Time (s) | n | Iterations |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for (scenario, algorithm), rows in sorted(
        groups.items(), key=lambda item: (item[0][0], order[item[0][1]])
    ):
        def mean(field: str) -> float:
            return statistics.fmean(float(row[field]) for row in rows)

        lines.append(
            f"| {scenario.size}x{scenario.size} | {scenario.horizon} | "
            f"{scenario.delta:.1f} | {algorithm} | {len(rows)} | "
            f"{mean('objective'):.6f} | {mean('risk'):.6f} | "
            f"{mean('time_s'):.3f} | {mean('n'):.1f} | "
            f"{mean('iterations'):.1f} |"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _validate_args(args: argparse.Namespace) -> None:
    if args.trials < 1:
        raise ValueError("--trials must be positive")
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
