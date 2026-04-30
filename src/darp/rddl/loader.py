"""RDDL loading through selectable parser frontends."""

# TODO(phase-2): Support repository names in addition to explicit domain and
# instance file paths for every frontend that can handle them.

from __future__ import annotations

from pathlib import Path

from darp.rddl.extended import DARPExtendedFrontend
from darp.rddl.frontend import ParsedRDDL, RDDLFrontend, RDDLFrontendError
from darp.rddl.pyrddl_frontend import PyRDDLFrontend
from darp.rddl.pyrddlgym_frontend import PyRDDLGymFrontend

LoadedRDDL = ParsedRDDL


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
        if normalized in {"pyrddlgym", "pyRDDLGym".lower()}:
            return PyRDDLGymFrontend()
        if normalized == "pyrddl":
            return PyRDDLFrontend()
        if normalized in {"darp", "extended", "darp_rddl"}:
            return DARPExtendedFrontend()
        raise RDDLFrontendError(
            f"Unknown RDDL frontend {frontend!r}. Expected pyrddlgym, pyrddl, or darp."
        )
