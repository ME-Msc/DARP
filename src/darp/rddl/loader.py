"""RDDL loading through selectable parser frontends."""

# TODO(phase-3.1): Support repository names in addition to explicit domain and
# instance file paths for every frontend that can handle them.

from __future__ import annotations

import argparse
import json
from pathlib import Path

from darp.rddl.extended import DARPExtendedFrontend
from darp.rddl.frontend import ParsedRDDL, RDDLFrontend, RDDLFrontendError
from darp.rddl.pyrddl_frontend import PyRDDLFrontend
from darp.rddl.pyrddlgym_frontend import PyRDDLGymFrontend

LoadedRDDL = ParsedRDDL
FRONTEND_ALIASES = {
    "pyrddlgym": "pyrddlgym",
    "pyrddl_gym": "pyrddlgym",
    "pyrddl": "pyrddl",
    "darp": "darp",
    "extended": "darp",
    "darp_rddl": "darp",
}


class RDDLLoader:
    """Load RDDL through a selectable parser frontend. / 通过可选择的 parser frontend 加载 RDDL。"""

    def __init__(self, frontend: str | RDDLFrontend = "pyrddlgym") -> None:
        """Create a loader with the selected frontend. / 使用指定 frontend 创建加载器。"""
        self.frontend = self._resolve_frontend(frontend)

    def load(
        self,
        domain: str | Path,
        instance: str | Path,
        frontend: str | RDDLFrontend | None = None,
    ) -> LoadedRDDL:
        """Load a domain/instance pair through a frontend. / 通过 frontend 加载 domain/instance 文件对。"""
        parser = self.frontend if frontend is None else self._resolve_frontend(frontend)
        return parser.parse(domain, instance)

    def _resolve_frontend(self, frontend: str | RDDLFrontend) -> RDDLFrontend:
        """Convert a frontend name or object into a frontend instance. / 将 frontend 名称或对象解析为 frontend 实例。"""
        if not isinstance(frontend, str):
            return frontend
        normalized = frontend.lower().replace("-", "_")
        canonical = FRONTEND_ALIASES.get(normalized)
        if canonical == "pyrddlgym":
            return PyRDDLGymFrontend()
        if canonical == "pyrddl":
            return PyRDDLFrontend()
        if canonical == "darp":
            return DARPExtendedFrontend()
        raise RDDLFrontendError(
            f"Unknown RDDL frontend {frontend!r}. Expected pyrddlgym, pyrddl, or darp."
        )


def available_frontends() -> tuple[str, ...]:
    """Return canonical frontend names for CLI choices. / 返回 CLI 可选的标准 frontend 名称。"""
    return ("darp", "pyrddl", "pyrddlgym")


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for frontend inspection. / 构建用于检查 frontend 的命令行 parser。"""
    parser = argparse.ArgumentParser(description="Load RDDL through a selected frontend.")
    parser.add_argument("domain", help="RDDL domain file")
    parser.add_argument("instance", help="RDDL instance file")
    parser.add_argument(
        "--frontend",
        default="darp",
        choices=available_frontends(),
        help="parser frontend to use",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the frontend inspection command. / 运行 frontend 检查命令。"""
    args = build_parser().parse_args(argv)
    loaded = RDDLLoader(args.frontend).load(args.domain, args.instance)
    print(json.dumps(loaded.to_summary_dict(), indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
