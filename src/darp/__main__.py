"""Top-level command-line entrypoint for DARP."""

# TODO(phase-3.1): Add online solve-loop commands after the planner interface is
# ready to run beyond the visualizer prototype.

from __future__ import annotations

import argparse

from darp.rddl.loader import available_frontends
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
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the DARP command-line entrypoint. / 运行 DARP 命令行入口。"""
    parser = build_parser()
    args = parser.parse_args(argv)
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


if __name__ == "__main__":
    raise SystemExit(main())
