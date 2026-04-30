"""pyRDDLGym parser/simulator frontend for standard RDDL."""

# TODO(phase-2): Inspect pyRDDLGym 2.x model fields and map them into DARP's
# PlanningProblem compiler IR.
# TODO(parser): If pyRDDLGym exposes stable parser classes, evaluate subclassing
# them for DARP-RDDL extensions instead of treating the package as a black box.

from __future__ import annotations

from pathlib import Path

from darp.rddl.frontend import ParsedRDDL, RDDLFrontendError


class PyRDDLGymFrontend:
    """Adapt pyRDDLGym parsing/simulation to DARP. / 将 pyRDDLGym 解析和仿真适配到 DARP。"""

    name = "pyrddlgym"
    supports_extended_syntax = False

    def parse(self, domain: str | Path, instance: str | Path) -> ParsedRDDL:
        """Build a pyRDDLGym environment and wrap its model. / 创建 pyRDDLGym 环境并封装其模型。"""
        try:
            import pyRDDLGym
        except ImportError as exc:
            raise RDDLFrontendError(
                "pyRDDLGym is required for the pyrddlgym frontend. "
                "Install with `pip install -e .[rddl]`."
            ) from exc

        env = pyRDDLGym.make(str(domain), str(instance))
        return ParsedRDDL(
            frontend=self.name,
            domain=str(domain),
            instance=str(instance),
            env=env,
            model=getattr(env, "model", None),
            metadata={"source": "pyRDDLGym"},
        )
