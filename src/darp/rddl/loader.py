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
    def __init__(self, frontend: str | RDDLFrontend = "pyrddlgym") -> None:
        self.frontend = self._resolve_frontend(frontend)

    def load(
        self,
        domain: str | Path,
        instance: str | Path,
        frontend: str | RDDLFrontend | None = None,
    ) -> LoadedRDDL:
        parser = self.frontend if frontend is None else self._resolve_frontend(frontend)
        return parser.parse(domain, instance)

    def _resolve_frontend(self, frontend: str | RDDLFrontend) -> RDDLFrontend:
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
