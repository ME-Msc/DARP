"""Run DARP once on an RDDL problem and export the conditional policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from darp.adapter.loader import load_rddl
from darp.adapter.runtime import PyRDDLGymRuntime
from darp.model.duration_sidecar import (
    build_duration_sidecar,
    load_duration_sidecar,
)
from darp.planning.full_ilp import FullILPPlanner
from darp.planning.hilp import HILPPlanner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="darp",
        description="Solve one finite RDDL policy with DARP full-ILP or HILP.",
    )
    parser.add_argument("--domain", required=True, help="RDDL domain file")
    parser.add_argument("--instance", required=True, help="RDDL instance file")
    parser.add_argument("--duration", help="JSON duration sidecar")
    parser.add_argument(
        "--planner",
        choices=("hilp", "full-ilp"),
        default="hilp",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--risk-budget", type=float)
    parser.add_argument("--heuristic-lookahead-depth", type=int, default=4)
    parser.add_argument("--expansion-rounds", type=int)
    parser.add_argument("--frontier-width", type=int, default=1)
    parser.add_argument(
        "--hilp-heuristic",
        choices=("one-step-greedy", "reachable-bellman"),
        default="reachable-bellman",
    )
    parser.add_argument("--timeout", type=float, default=60.0, help="solver seconds")
    parser.add_argument("--output", type=Path, help="JSON destination")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.timeout <= 0:
        raise ValueError("--timeout must be positive")

    problem = load_rddl(args.domain, args.instance)
    runtime = PyRDDLGymRuntime(problem.env)
    runtime.reset(seed=args.seed)
    duration = (
        load_duration_sidecar(args.duration)
        if args.duration
        else build_duration_sidecar({"kind": "fixed", "default": 1.0})
    )
    risk = duration.risk
    interface = problem.build_grounded_view().build_and_or_interface(
        runtime,
        risk=risk,
    )
    duration.validate_actions([choice.label for choice in interface.actions])
    state_names = getattr(interface.exact_kernel, "state_names", ())
    duration.validate_state_fluents(state_names)
    budget = args.risk_budget if args.risk_budget is not None else risk.budget
    solver_limit_ms = args.timeout * 1000.0
    planner = (
        FullILPPlanner(
            risk_budget=budget,
            solver_time_limit_ms=solver_limit_ms,
        )
        if args.planner == "full-ilp"
        else HILPPlanner(
            heuristic_lookahead_depth=args.heuristic_lookahead_depth,
            expansion_rounds=args.expansion_rounds,
            frontier_width=args.frontier_width,
            heuristic_mode=args.hilp_heuristic,
            risk_budget=budget,
            solver_time_limit_ms=solver_limit_ms,
        )
    )
    decision = planner.choose_action(
        runtime,
        interface,
        duration.evaluator(horizon=runtime.horizon),
    )
    payload = {
        "planner": args.planner,
        "seed": args.seed,
        "risk_budget": budget,
        "decision": decision.to_dict(),
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


if __name__ == "__main__":
    raise SystemExit(main())
