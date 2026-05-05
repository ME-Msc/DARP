"""Top-level command-line entrypoint for DARP."""

# TODO(phase-8.1): Add external rddlsim/PROST simulator mode to the online runtime.

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
        help="start the live HTML UI instead of printing a terminal trace",
    )
    parser.add_argument(
        "--mode",
        choices=("online",),
        default="online",
        help="execution mode; Phase 3 currently supports local online execution",
    )
    parser.add_argument("--domain", help="RDDL domain file path")
    parser.add_argument("--instance", help="RDDL instance file path")
    parser.add_argument(
        "--simulator",
        choices=("darp", "rddlgym", "pyrddlgym"),
        default="darp",
        help="runtime simulator used by the live UI; non-visual solving currently uses DARP internal simulator",
    )
    parser.add_argument(
        "--with-simulator",
        nargs="?",
        const="darp",
        choices=("darp", "rddlgym", "pyrddlgym"),
        dest="simulator",
        help=argparse.SUPPRESS,
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
    parser.add_argument("--seed", type=int, default=0, help="local simulator random seed")
    parser.add_argument(
        "--time-budget-ms",
        type=float,
        help="soft per-decision time budget recorded in the trace",
    )
    parser.add_argument("--output", help="optional JSON output file for non-visual execution")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the DARP command-line entrypoint. / 运行 DARP 命令行入口。"""
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.domain and not args.instance and args.visualizer:
        parser.error("--visualizer requires both --domain and --instance.")
    if not args.domain and not args.instance and not args.visualizer:
        return _run_solve(args)
    if not args.domain or not args.instance:
        parser.error("--domain and --instance must be provided together.")
    if not args.visualizer:
        if args.simulator != "darp":
            parser.error("--simulator is currently only supported with --visualizer.")
        return _run_solve(args)
    return serve_visualizer(
        domain=args.domain,
        instance=args.instance,
        simulator=args.simulator,
        frontend=args.frontend,
        host=args.host,
        port=args.port,
        seed=args.seed,
        open_browser=not args.no_open,
    )


def _run_solve(args: argparse.Namespace) -> int:
    """Run a command-line planning session. / 运行一次命令行规划会话。"""
    problem = _load_problem(args.domain, args.instance, args.frontend)
    if args.mode != "online":
        raise ValueError(f"Unsupported execution mode: {args.mode!r}.")
    result = run_local_online_session(
        problem,
        seed=args.seed,
        time_budget_ms=args.time_budget_ms,
    )
    payload = result.to_dict()
    if args.output:
        Path(args.output).write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
    print(_format_terminal_trace(payload))
    return 0


def _format_terminal_trace(payload: dict[str, object]) -> str:
    """Format one online result for human-readable terminal output. / 将在线结果格式化为适合终端阅读的文本。"""
    lines = [
        "DARP online trace",
        f"Problem: {payload['problem']}",
        f"Planner: {payload['planner']}",
        f"Seed: {payload['seed']}",
        f"Horizon: {payload['horizon']} (max depth {payload['max_depth']})",
        "Steps:",
    ]
    steps = payload.get("steps", [])
    assert isinstance(steps, list)
    for step in steps:
        assert isinstance(step, dict)
        decision = step.get("decision", {})
        value = decision.get("value") if isinstance(decision, dict) else None
        value_text = f" value={float(value):.3f}" if isinstance(value, int | float) else ""
        lines.append(
            "  "
            f"t={step['step']} "
            f"obs={step['observation']} "
            f"action={step['action']} "
            f"reward={step['reward']} "
            f"next={step['next_observation']}"
            f"{value_text}"
        )
    lines.append(f"Total reward: {payload['total_reward']}")
    return "\n".join(lines)


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
