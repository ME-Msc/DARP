"""Run DARP once on an RDDL problem and export the conditional policy."""

from __future__ import annotations

import argparse
import json
from importlib import import_module
from pathlib import Path
from typing import cast

from darp.planning.heuristic import load_utility_heuristic
from darp.solve import (
    PlannerName,
    RootBeliefFactory,
    solve_rddl,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="darp",
        description="Solve one finite RDDL policy with DARP full-ILP or HILP.",
    )
    parser.add_argument("--domain", required=True, help="RDDL domain file")
    parser.add_argument("--instance", required=True, help="RDDL instance file")
    parser.add_argument(
        "--duration",
        required=True,
        help="JSON duration/risk sidecar (duration is intentionally outside RDDL)",
    )
    parser.add_argument(
        "--planner",
        choices=("hilp", "full-ilp"),
        default="hilp",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--risk-budget", type=float)
    parser.add_argument("--expansion-rounds", type=int)
    parser.add_argument(
        "--frontier-width",
        type=int,
        help="optional batch cap; omitted means all incumbent frontier nodes",
    )
    parser.add_argument(
        "--heuristic",
        help="external HILP utility heuristic as module:attribute",
    )
    parser.add_argument(
        "--root-belief",
        help="external initial-belief factory as module:attribute",
    )
    parser.add_argument(
        "--terminal-heuristic",
        action="store_true",
        help="use the external heuristic as the duration-boundary value",
    )
    parser.add_argument("--timeout", type=float, default=60.0, help="solver seconds")
    parser.add_argument("--output", type=Path, help="JSON destination")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    heuristic = load_utility_heuristic(args.heuristic) if args.heuristic else None
    root_belief_factory = (
        _load_root_belief_factory(args.root_belief)
        if args.root_belief
        else None
    )
    result = solve_rddl(
        args.domain,
        args.instance,
        args.duration,
        planner=cast(PlannerName, args.planner),
        seed=args.seed,
        risk_budget=args.risk_budget,
        expansion_rounds=args.expansion_rounds,
        frontier_width=args.frontier_width,
        heuristic=heuristic,
        terminal_heuristic=args.terminal_heuristic,
        timeout_s=args.timeout,
        root_belief_factory=root_belief_factory,
    )
    payload = {
        "planner": args.planner,
        "seed": args.seed,
        "risk_budget": result.risk_budget,
        "heuristic": heuristic.name if heuristic is not None else None,
        "elapsed_s": result.elapsed_s,
        "decision": result.decision.to_dict(),
    }
    document = json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
    if args.output is None:
        print(document, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(document, encoding="utf-8")
    return 0


def _load_root_belief_factory(spec: str) -> RootBeliefFactory:
    module_name, separator, attribute = spec.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("Root belief must use the form 'module:attribute'.")
    value = getattr(import_module(module_name), attribute)
    if not callable(value):
        raise TypeError(f"{spec!r} must resolve to a callable root-belief factory.")
    return cast(RootBeliefFactory, value)


if __name__ == "__main__":
    raise SystemExit(main())
