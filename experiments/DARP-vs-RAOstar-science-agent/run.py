"""Run paired DARP-HILP and original RAO* Science Agent searches."""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path
from typing import Any

from .darp_runner import run_darp, warm_up
from .raostar_runner import RAOSTAR_COMMIT, RAOSTAR_URL, RAOStarRunner

DELTAS = (0.002, 0.01, 0.05)
ALGORITHMS = ("DARP-HILP", "RAO*")
SEMANTIC_SCOPE = "source_non_scheduling"
SOURCE_MODEL = "tFakePlannerRockSampleModel(perform_scheduling=False)"
PAPER_URL = "https://ojs.aaai.org/index.php/AAAI/article/view/10423"
BENAZERA_URL = "https://aiweb.cs.washington.edu/ai/planning/papers/mausam-ijcai05.pdf"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE = PROJECT_ROOT / ".cache" / "baselines"
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "output" / "DARP-vs-RAOstar-science-agent" / "raw.csv"
)

FIELDS = (
    "delta",
    "algorithm",
    "trial",
    "objective",
    "risk",
    "time_s",
    "expanded_nodes",
    "evaluated_states",
    "iterations",
    "complete",
    "semantic_scope",
    "source_model",
    "raostar_url",
    "raostar_commit",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raostar-checkout", type=Path)
    parser.add_argument("--baseline-cache", type=Path, default=DEFAULT_CACHE)
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
    parity_pending = True
    for delta in args.deltas:
        for trial in range(1, args.trials + 1):
            model, belief = raostar.make_problem()
            darp = run_darp(
                delta=delta,
                seed=args.seed + trial - 1,
                timeout_s=args.timeout,
                reference_model=model if parity_pending else None,
                reference_belief=belief if parity_pending else None,
            )
            parity_pending = False
            rao = raostar.run(
                model,
                belief,
                delta=delta,
                timeout_s=args.timeout,
            )
            if abs(float(darp["risk"]) - float(rao["risk"])) > 1e-12:
                raise RuntimeError(
                    "Paired solvers disagree on Science Agent execution risk: "
                    f"DARP={darp['risk']}, RAO*={rao['risk']}"
                )
            if abs(float(darp["objective"]) - float(rao["objective"])) > 1e-3:
                raise RuntimeError(
                    "Paired solvers disagree beyond the configured MIP tolerance: "
                    f"DARP={darp['objective']}, RAO*={rao['objective']}"
                )
            for algorithm, metrics in zip(ALGORITHMS, (darp, rao), strict=True):
                row = {
                    "delta": delta,
                    "algorithm": algorithm,
                    "trial": trial,
                    **metrics,
                    "semantic_scope": SEMANTIC_SCOPE,
                    "source_model": SOURCE_MODEL,
                    "raostar_url": RAOSTAR_URL,
                    "raostar_commit": RAOSTAR_COMMIT,
                }
                rows.append(row)
                print(
                    f"delta={delta:g} trial={trial} {algorithm}: "
                    f"objective={metrics['objective']:.6f}, "
                    f"risk={metrics['risk']:.6f}, time={metrics['time_s']:.3f}s"
                )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    summary = args.summary or args.output.with_name("table.md")
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(_markdown(rows, args.trials), encoding="utf-8")
    print(f"raw: {args.output}")
    print(f"summary: {summary}")
    return 0


def _validate_args(args: argparse.Namespace) -> None:
    if args.trials < 1:
        raise ValueError("trials must be positive")
    if args.timeout is not None and args.timeout <= 0:
        raise ValueError("timeout must be positive")
    if any(not 0.0 <= delta <= 1.0 for delta in args.deltas):
        raise ValueError("every delta must be in [0, 1]")


def _markdown(rows: list[dict[str, Any]], trials: int) -> str:
    lines = [
        "# Science Agent: DARP-HILP vs RAO*",
        "",
        "| Δ | Algorithm | Objective | Risk | Time (s) | Expanded nodes† | "
        "Evaluated states‡ | Iterations |",
        "|---:|:---|---:|---:|---:|---:|---:|---:|",
    ]
    for delta in sorted({float(row["delta"]) for row in rows}):
        for algorithm in ALGORITHMS:
            selected = [
                row
                for row in rows
                if float(row["delta"]) == delta and row["algorithm"] == algorithm
            ]
            lines.append(
                "| "
                + " | ".join(
                    (
                        f"{delta:g}",
                        algorithm,
                        _mean(selected, "objective", 6),
                        _mean(selected, "risk", 6),
                        _mean_sd(selected, "time_s"),
                        _mean(selected, "expanded_nodes", 1),
                        _mean(selected, "evaluated_states", 1),
                        _mean(selected, "iterations", 1),
                    )
                )
                + " |"
            )
    lines.extend(
        (
            "",
            f"Trials per configuration: {trials}.",
            "",
            "† DARP expands action histories; RAO* expands belief hypergraph nodes. "
            "These are native search-effort counters, not identical units.",
            "",
            "‡ DARP reports distinct grounded states lazily compiled during search. "
            "RAO* reports the original implementation's cumulative evaluated belief particles; "
            "the two counters describe implementation effort but are not identical quantities.",
            "",
            "## Scope and provenance",
            "",
            f"- Semantic scope: `{SEMANTIC_SCOPE}`.",
            f"- Source model: `{SOURCE_MODEL}` with the inert 1000-second "
            "constraint from its checked-in test.",
            "- DARP uses the equivalent RDDL model, fixed unit duration, and horizon 5; "
            "no-revisit actions guarantee relay or crash within that bound.",
            f"- RAO* source: [{RAOSTAR_URL}]({RAOSTAR_URL}) at `{RAOSTAR_COMMIT}`.",
            f"- Paper: [RAO*: An Algorithm for Chance-Constrained POMDPs]({PAPER_URL}).",
            f"- Domain ancestor: [Benazera et al. (2005)]({BENAZERA_URL}) describes "
            "a continuous-resource HAO* rover, not the exact RAO* test parameters.",
            "- This is not a reproduction of the paper's PARIS scheduling/time-window Table 1; "
            "`perform_scheduling=False` is required for exact shared semantics with current DARP.",
            "- Small objective differences are within DARP's configured Gurobi MIPGap; "
            "the complete reachable T/O/reward/risk/heuristic/terminal model is checked "
            "before timing.",
            "",
        )
    )
    return "\n".join(lines)


def _mean(rows: list[dict[str, Any]], field: str, digits: int) -> str:
    return f"{statistics.fmean(float(row[field]) for row in rows):.{digits}f}"


def _mean_sd(rows: list[dict[str, Any]], field: str) -> str:
    values = [float(row[field]) for row in rows]
    mean = statistics.fmean(values)
    return f"{mean:.3f}" if len(values) == 1 else f"{mean:.3f} ± {statistics.stdev(values):.3f}"


if __name__ == "__main__":
    raise SystemExit(main())
