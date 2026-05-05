"""Top-level command-line entrypoint for DARP."""

# TODO(phase-3.2): Add external rddlsim/PROST simulator mode to `darp solve`.

from __future__ import annotations

import argparse
import json
from pathlib import Path

from darp.core.problem import PlanningProblem, make_tiny_grid_problem
from darp.online import run_local_online_session
from darp.rddl.compiler import RDDLCompiler
from darp.rddl.loader import RDDLLoader, available_frontends
from darp.rddl.visualizer import serve_visualizer


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level DARP command parser. / 构建 DARP 顶层命令行 parser。"""
    parser = argparse.ArgumentParser(
        prog="darp",
        description="Durative Action RDDL Planner research prototype.",
    )
    parser.add_argument(
        "--visualizer",
        action="store_true",
        help="start the live HTML RDDL source/AST visualizer",
    )
    parser.add_argument("--domain", help="RDDL domain file path")
    parser.add_argument("--instance", help="RDDL instance file path")
    parser.add_argument(
        "--with-simulator",
        nargs="?",
        const="darp",
        choices=("darp", "rddlgym", "pyrddlgym"),
        help="enable simulator mode; omit the value for DARP internal simulator",
    )
    parser.add_argument(
        "--frontend",
        choices=available_frontends(),
        default="darp",
        help="RDDL frontend used when DARP compiles the problem",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="host for the live visualizer server",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=0,
        help="port for the live visualizer server; 0 chooses a free port",
    )
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="serve the visualizer without opening a browser",
    )
    subcommands = parser.add_subparsers(dest="command")
    solve = subcommands.add_parser("solve", help="run a planner on a RDDL problem")
    solve.add_argument(
        "--mode",
        choices=("online",),
        default="online",
        help="solve mode; Phase 3 currently supports local online execution",
    )
    solve.add_argument("--domain", help="RDDL domain file path")
    solve.add_argument("--instance", help="RDDL instance file path")
    solve.add_argument(
        "--frontend",
        choices=available_frontends(),
        default="darp",
        help="RDDL frontend used when compiling explicit domain/instance inputs",
    )
    solve.add_argument("--steps", type=int, help="maximum number of online decision steps")
    solve.add_argument("--seed", type=int, default=0, help="local simulator random seed")
    solve.add_argument(
        "--time-budget-ms",
        type=float,
        help="soft per-decision time budget recorded in the JSON trace",
    )
    solve.add_argument("--output", help="optional JSON output file")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the DARP command-line entrypoint. / 运行 DARP 命令行入口。"""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "solve":
        return _run_solve(args)
    if not args.visualizer:
        parser.print_help()
        return 0
    if not args.domain or not args.instance:
        parser.error("--visualizer requires both --domain and --instance.")
    return serve_visualizer(
        domain=args.domain,
        instance=args.instance,
        simulator=args.with_simulator,
        frontend=args.frontend,
        host=args.host,
        port=args.port,
        open_browser=not args.no_open,
    )


def _run_solve(args: argparse.Namespace) -> int:
    """Run a command-line solve session. / 运行一次命令行求解会话。"""
    problem = _load_problem(args.domain, args.instance, args.frontend)
    if args.mode != "online":
        raise ValueError(f"Unsupported solve mode: {args.mode!r}.")
    result = run_local_online_session(
        problem,
        steps=args.steps,
        seed=args.seed,
        time_budget_ms=args.time_budget_ms,
    )
    text = json.dumps(result.to_dict(), indent=2, sort_keys=True, default=str)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


def _load_problem(domain: str | None, instance: str | None, frontend: str) -> PlanningProblem:
    """Load RDDL inputs or the built-in demo problem. / 加载 RDDL 输入或内置 demo 问题。"""
    if domain or instance:
        if not (domain and instance):
            raise SystemExit("--domain and --instance must be provided together.")
        loaded = RDDLLoader(frontend).load(domain, instance)
        return RDDLCompiler().compile(loaded)
    return make_tiny_grid_problem()


if __name__ == "__main__":
    raise SystemExit(main())
