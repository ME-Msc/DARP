"""Top-level command-line entrypoint for DARP."""

# TODO(phase-9.2): Add benchmark runner and offline policy replay commands after
# the online HILP/Gurobi path is stable.

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

from darp.adapter.loader import RDDLLoader
from darp.model.duration_sidecar import load_duration_sidecar
from darp.planning.session import run_online_session


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level DARP command parser. / 构建 DARP 顶层命令行 parser。"""
    parser = argparse.ArgumentParser(
        prog="darp",
        description="Durative Action RDDL Planner research prototype.",
    )
    parser.add_argument("--domain", help="RDDL domain file path")
    parser.add_argument("--instance", help="RDDL instance file path")
    parser.add_argument(
        "--duration",
        help="optional YAML/JSON duration sidecar; defaults to fixed unit duration",
    )
    parser.add_argument(
        "--planner",
        choices=("hilp", "full-ilp", "rollout"),
        default="rollout",
        help="online planner to use; rollout is the fast default, hilp/full-ilp enable the paper path",
    )
    parser.add_argument("--seed", type=int, default=0, help="runtime random seed")
    parser.add_argument(
        "--lookahead-depth",
        type=int,
        default=4,
        help=(
            "rollout lookahead depth or HILP partial-tree depth cap; "
            "reachable-bellman also uses this as its local Bellman horizon cap"
        ),
    )
    parser.add_argument(
        "--hilp-iterations",
        type=int,
        default=4,
        help="maximum HILP partial-tree refinement iterations per decision",
    )
    parser.add_argument(
        "--frontier-width",
        type=int,
        default=1,
        help="maximum HILP frontier action histories expanded per iteration",
    )
    parser.add_argument(
        "--hilp-heuristic",
        choices=("one-step-greedy", "reachable-bellman"),
        default="one-step-greedy",
        help="HILP frontier heuristic: one-step-greedy is fastest; reachable-bellman uses local reachable-state Bellman backups",
    )
    parser.add_argument(
        "--risk-budget",
        type=float,
        help="optional risk budget for full-ILP rows",
    )
    parser.add_argument(
        "--particles",
        type=int,
        default=32,
        help="particle count for rollout POMDP belief tracking; full-ilp/hilp use exact belief",
    )
    parser.add_argument("--output", help="optional JSON output file for non-visual execution")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the DARP command-line entrypoint. / 运行 DARP 命令行入口。"""
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.domain or not args.instance:
        parser.error("--domain and --instance must be provided together.")
    return _run_rddl_online(args)


def _run_rddl_online(args: argparse.Namespace) -> int:
    """Run standard RDDL online with pyRDDLGym. / 使用 pyRDDLGym 在线运行标准 RDDL。"""
    if args.lookahead_depth < 1:
        raise ValueError("--lookahead-depth must be at least 1.")
    if args.hilp_iterations < 1:
        raise ValueError("--hilp-iterations must be at least 1.")
    if args.frontier_width < 1:
        raise ValueError("--frontier-width must be at least 1.")
    if args.particles < 1:
        raise ValueError("--particles must be at least 1.")
    load_started_at = perf_counter()
    problem = RDDLLoader().load(args.domain, args.instance)
    rddl_load_ms = (perf_counter() - load_started_at) * 1000.0
    duration_started_at = perf_counter()
    duration = load_duration_sidecar(args.duration) if args.duration else None
    duration_load_ms = (perf_counter() - duration_started_at) * 1000.0
    result = run_online_session(
        problem,
        seed=args.seed,
        planner_name=args.planner,
        duration_sidecar=duration,
        lookahead_depth=args.lookahead_depth,
        hilp_iterations=args.hilp_iterations,
        frontier_width=args.frontier_width,
        hilp_heuristic=args.hilp_heuristic,
        risk_budget=args.risk_budget,
        particle_count=args.particles,
    )
    payload = result.to_dict()
    payload.setdefault("timing", {})
    payload["timing"].update(
        {
            "rddl_load_ms": rddl_load_ms,
            "duration_load_ms": duration_load_ms,
        }
    )
    payload["rddl"] = problem.to_summary_dict()
    if args.output:
        Path(args.output).write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
    print(_format_pyrddlgym_trace(payload))
    return 0


def _format_pyrddlgym_trace(payload: dict[str, object]) -> str:
    """Format one pyRDDLGym online trace for terminal output. / 将 pyRDDLGym 在线轨迹格式化到终端。"""
    lines = [
        "DARP pyRDDLGym online trace",
        f"Problem: {payload['problem']}",
        f"Planner: {payload['planner']}",
        f"Seed: {payload['seed']}",
        f"Horizon: {payload['horizon']} (max depth {payload['max_depth']})",
        _search_depth_label(payload),
        f"Duration: {_duration_label(payload.get('duration', {}))}",
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
            f"state={_active_state_label(step.get('state', {}))} "
            f"action={step['action']} "
            f"reward={step['reward']} "
            f"next={_active_state_label(step.get('next_state', {}))}"
            f"{value_text}"
        )
    lines.append(f"Total reward: {payload['total_reward']}")
    return "\n".join(lines)


def _duration_label(duration: object) -> str:
    """Format duration metadata for terminal output. / 为终端输出格式化 duration metadata。"""
    if not isinstance(duration, dict):
        return str(duration)
    kind = duration.get("kind", "unknown")
    path = duration.get("path")
    suffix = "default" if duration.get("defaulted") else path
    return f"{kind} ({suffix})"


def _search_depth_label(payload: dict[str, object]) -> str:
    """Format the planner search-depth line. / 格式化 planner search depth 行。"""
    if payload.get("planner") == "full-ilp-gurobi":
        return "Search depth: remaining horizon"
    return f"Lookahead depth: {payload['lookahead_depth']}"


def _active_state_label(state: object) -> str:
    """Return true-valued fluent names for compact terminal output. / 返回真值 fluent 名称以压缩终端输出。"""
    if not isinstance(state, dict):
        return str(state)
    active = [str(name) for name, value in state.items() if value is True]
    if active:
        return ",".join(active)
    return "(none)"


if __name__ == "__main__":
    raise SystemExit(main())
