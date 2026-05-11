"""Top-level command-line entrypoint for DARP."""

# TODO(phase-5.2): Route rollout, AND-OR, full ILP, and HILP planners through a
# shared planner registry and trace formatter.

from __future__ import annotations

import argparse
import json
from pathlib import Path

from darp.loader import RDDLLoader
from darp.session import run_online_session


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level DARP command parser. / 构建 DARP 顶层命令行 parser。"""
    parser = argparse.ArgumentParser(
        prog="darp",
        description="Durative Action RDDL Planner research prototype.",
    )
    parser.add_argument("--domain", help="RDDL domain file path")
    parser.add_argument("--instance", help="RDDL instance file path")
    parser.add_argument("--seed", type=int, default=0, help="runtime random seed")
    parser.add_argument(
        "--lookahead-depth",
        type=int,
        default=4,
        help="rollout lookahead depth for pyRDDLGym-backed online execution",
    )
    parser.add_argument(
        "--particles",
        type=int,
        default=32,
        help="particle count for pyRDDLGym POMDP belief tracking",
    )
    parser.add_argument(
        "--time-budget-ms",
        type=float,
        help="hard per-decision time budget in milliseconds",
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
    if args.particles < 1:
        raise ValueError("--particles must be at least 1.")
    loaded = RDDLLoader().load(args.domain, args.instance)
    result = run_online_session(
        loaded,
        seed=args.seed,
        lookahead_depth=args.lookahead_depth,
        time_budget_ms=args.time_budget_ms,
        particle_count=args.particles,
    )
    payload = result.to_dict()
    payload["rddl"] = loaded.to_summary_dict()
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
        f"Lookahead depth: {payload['lookahead_depth']}",
        "Steps:",
    ]
    steps = payload.get("steps", [])
    assert isinstance(steps, list)
    for step in steps:
        assert isinstance(step, dict)
        decision = step.get("decision", {})
        value = decision.get("value") if isinstance(decision, dict) else None
        value_text = f" value={float(value):.3f}" if isinstance(value, int | float) else ""
        status_text = " timeout" if isinstance(decision, dict) and decision.get("timed_out") else ""
        lines.append(
            "  "
            f"t={step['step']} "
            f"state={_active_state_label(step.get('state', {}))} "
            f"action={step['action']} "
            f"reward={step['reward']} "
            f"next={_active_state_label(step.get('next_state', {}))}"
            f"{value_text}"
            f"{status_text}"
        )
    lines.append(f"Total reward: {payload['total_reward']}")
    return "\n".join(lines)


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
