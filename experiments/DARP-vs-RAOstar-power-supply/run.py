"""Run the paired reduced Power Supply experiment to natural completion."""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path
from typing import Any

from .darp_runner import instance_path, read_instance, run_darp, warm_up
from .raostar_runner import RAOSTAR_COMMIT, RAOSTAR_URL, RAOStarRunner

SENSORS = (1, 2)
DELTAS = (0.0, 0.5, 1.0)
ALGORITHMS = ("DARP-HILP", "RAO*")
SEMANTIC_SCOPE = "darp_authored_reduced_psr"
SOURCE_NETWORK = "Bonet-Thiebaux-three-line"
FAULT_PRIOR = "uniform exactly-one fault on l1/l2"
THIEBAUX_URL = "https://users.cecs.anu.edu.au/~thiebaux/papers/ecp01.pdf"
BONET_URL = "https://users.cecs.anu.edu.au/~thiebaux/papers/icaps03.pdf"
BENCHMARK_URL = "https://users.cecs.anu.edu.au/~thiebaux/benchmarks/pds/"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE = PROJECT_ROOT / ".cache" / "baselines"
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "output" / "DARP-vs-RAOstar-power-supply" / "raw.csv"
)

FIELDS = (
    "sensors",
    "horizon",
    "delta",
    "algorithm",
    "trial",
    "cost",
    "risk",
    "time_s",
    "expanded_nodes",
    "evaluated_states",
    "iterations",
    "complete",
    "semantic_scope",
    "source_network",
    "fault_prior",
    "raostar_url",
    "raostar_commit",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raostar-checkout", type=Path)
    parser.add_argument("--baseline-cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--sensors", type=int, nargs="+", default=SENSORS)
    parser.add_argument("--deltas", type=float, nargs="+", default=DELTAS)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--timeout", type=float)
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _validate_args(args)
    raostar = RAOStarRunner.create(
        repository=args.raostar_checkout,
        cache_root=args.baseline_cache,
    )
    print(f"RAO*: {raostar.repository}")
    warm_up()

    rows: list[dict[str, Any]] = []
    parity_pending = set(args.sensors)
    for delta in args.deltas:
        for sensors in args.sensors:
            instance = instance_path(sensors)
            _, horizon = read_instance(instance)
            for trial in range(1, args.trials + 1):
                model, belief = raostar.make_problem(sensors)
                darp = run_darp(
                    instance,
                    delta=delta,
                    seed=args.seed + trial - 1,
                    timeout_s=args.timeout,
                    reference_model=model if sensors in parity_pending else None,
                    reference_belief=belief if sensors in parity_pending else None,
                )
                parity_pending.discard(sensors)
                rao = raostar.run(
                    model,
                    belief,
                    delta=delta,
                    timeout_s=args.timeout,
                )
                if abs(float(darp["cost"]) - float(rao["cost"])) > 1e-6:
                    raise RuntimeError(
                        "Paired solvers disagree on optimal cost: "
                        f"DARP={darp['cost']}, RAO*={rao['cost']}"
                    )
                for algorithm, metrics in zip(ALGORITHMS, (darp, rao), strict=True):
                    row = {
                        "sensors": sensors,
                        "horizon": horizon,
                        "delta": delta,
                        "algorithm": algorithm,
                        "trial": trial,
                        **metrics,
                        "semantic_scope": SEMANTIC_SCOPE,
                        "source_network": SOURCE_NETWORK,
                        "fault_prior": FAULT_PRIOR,
                        "raostar_url": RAOSTAR_URL,
                        "raostar_commit": RAOSTAR_COMMIT,
                    }
                    rows.append(row)
                    print(
                        f"s={sensors} delta={delta:g} trial={trial} {algorithm}: "
                        f"cost={metrics['cost']:.6f}, risk={metrics['risk']:.6f}, "
                        f"time={metrics['time_s']:.3f}s"
                    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    summary = args.summary or args.output.with_name("table.md")
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(_markdown(rows, args.trials, tuple(args.sensors)), encoding="utf-8")
    print(f"raw: {args.output}")
    print(f"summary: {summary}")
    return 0


def _validate_args(args: argparse.Namespace) -> None:
    if args.trials < 1:
        raise ValueError("trials must be positive")
    if args.timeout is not None and args.timeout <= 0:
        raise ValueError("timeout must be positive")
    if tuple(args.sensors) != SENSORS:
        raise ValueError("The table comparison requires both sensor counts: 1 2")
    if any(not 0.0 <= delta <= 1.0 for delta in args.deltas):
        raise ValueError("every delta must be in [0, 1]")


def _markdown(
    rows: list[dict[str, Any]],
    trials: int,
    sensors: tuple[int, ...],
) -> str:
    lines = [
        "# Power Supply: DARP-HILP vs RAO*",
        "",
        "| Δ | Algorithm | Time (s), 1/2 sensors | Nodes, 1/2 sensors | "
        "Evaluated particles†, 1/2 sensors | Cost, 1/2 sensors | Risk, 1/2 sensors |",
        "|---:|:---|---:|---:|---:|---:|---:|",
    ]
    for delta in sorted({float(row["delta"]) for row in rows}):
        for algorithm in ALGORITHMS:
            lines.append(
                "| "
                + " | ".join(
                    (
                        f"{delta:g}",
                        algorithm,
                        _slash(rows, sensors, delta, algorithm, "time_s", 3),
                        _slash(rows, sensors, delta, algorithm, "expanded_nodes", 1),
                        _slash(rows, sensors, delta, algorithm, "evaluated_states", 1),
                        _slash(rows, sensors, delta, algorithm, "cost", 3),
                        _slash(rows, sensors, delta, algorithm, "risk", 3),
                    )
                )
                + " |"
            )
    lines.extend(
        (
            "",
            f"Trials per configuration: {trials}; cells show arithmetic means. "
            "The slash follows RAO* Table 2 and means 1/2 breaker sensors.",
            "",
            "DARP nodes are expanded action histories; RAO* nodes are expanded belief "
            "hypergraph nodes. † This is RAO*'s evaluated-belief-particle counter; DARP has "
            "no identical counter, so its cell is shown as —.",
            "",
            "## Scope and provenance",
            "",
            f"- Semantic scope: `{SEMANTIC_SCOPE}`; horizon 4; fixed unit duration.",
            f"- Network: `{SOURCE_NETWORK}`; prior: `{FAULT_PRIOR}`.",
            f"- Domain sources: [Thiébaux & Cordier 2001]({THIEBAUX_URL}) and "
            f"the executable GPT formalization in [Bonet & Thiébaux 2003]({BONET_URL}); "
            f"the [official benchmark page]({BENCHMARK_URL}) indexes network data/tools.",
            "- Costs follow the GPT PSR formalization: one per switch operation and "
            "five per healthy line left unpowered at finish.",
            "- Connecting a generator to a hidden fault sets a nonterminal unsafe state; "
            "the chance constraint bounds the probability of ever reaching it.",
            f"- RAO* source: [{RAOSTAR_URL}]({RAOSTAR_URL}) at `{RAOSTAR_COMMIT}`.",
            "- This is a DARP-authored reduced benchmark whose topology, action effects, "
            "and finish cost follow the cited PSR work. The prior, horizon, sensor subsets, "
            "and chance constraint are explicit experiment choices.",
            "- It is not the original RAO* 2016 semi-rural instance or a numerical "
            "reproduction of Table 2.",
            "",
        )
    )
    return "\n".join(lines)


def _slash(
    rows: list[dict[str, Any]],
    sensors: tuple[int, ...],
    delta: float,
    algorithm: str,
    field: str,
    digits: int,
) -> str:
    values: list[str] = []
    for sensor_count in sensors:
        selected = [
            row[field]
            for row in rows
            if int(row["sensors"]) == sensor_count
            and float(row["delta"]) == delta
            and row["algorithm"] == algorithm
            and row[field] is not None
        ]
        values.append(
            "—"
            if not selected
            else f"{statistics.fmean(float(value) for value in selected):.{digits}f}"
        )
    return "/".join(values)


if __name__ == "__main__":
    raise SystemExit(main())
