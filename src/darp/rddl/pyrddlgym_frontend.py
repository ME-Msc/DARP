"""pyRDDLGym parser/simulator frontend for standard RDDL."""

# TODO(phase-8.2): Map stable pyRDDLGym model fields into the external
# simulator adapter when protocol integration starts.
# TODO(phase-8.2): If pyRDDLGym exposes stable parser classes, evaluate subclassing
# them for DARP-RDDL extensions instead of treating the package as a black box.

from __future__ import annotations

import os
from pathlib import Path
import tempfile

from darp.rddl.basic_parser import BasicRDDLParser
from darp.rddl.frontend import ParsedRDDL, RDDLFrontendError, frontend_error, rddl_path


class PyRDDLGymFrontend:
    """Adapt pyRDDLGym parsing/simulation to DARP. / 将 pyRDDLGym 解析和仿真适配到 DARP。"""

    name = "pyrddlgym"
    supports_extended_syntax = False

    def parse(self, domain: str | Path, instance: str | Path) -> ParsedRDDL:
        """Build a pyRDDLGym environment and wrap its model. / 创建 pyRDDLGym 环境并封装其模型。"""
        domain_path = rddl_path(domain)
        instance_path = rddl_path(instance)
        _ensure_matplotlib_cache_dir()
        try:
            import pyRDDLGym
        except ImportError as exc:
            raise RDDLFrontendError(
                "pyRDDLGym is required for the pyrddlgym frontend. "
                "Install with `pip install -e .[rddl]`."
            ) from exc

        try:
            canonical_ast = BasicRDDLParser().parse_files(domain_path, instance_path)
            env = pyRDDLGym.make(str(domain_path), str(instance_path))
        except Exception as exc:
            raise frontend_error(self.name, domain_path, instance_path, exc) from exc
        model = getattr(env, "model", None)
        return ParsedRDDL(
            frontend=self.name,
            domain=str(domain_path),
            instance=str(instance_path),
            ast=canonical_ast,
            env=env,
            model=model,
            metadata={
                "source": "pyRDDLGym",
                "pyRDDLGym_version": getattr(pyRDDLGym, "__version__", None),
                "ast_type": type(canonical_ast).__name__,
                "env_type": type(env).__name__,
                "model_type": type(model).__name__ if model is not None else None,
            },
        )


def _ensure_matplotlib_cache_dir() -> None:
    """Give pyRDDLGym's matplotlib import a writable cache. / 为 pyRDDLGym 的 matplotlib 导入提供可写缓存。"""
    if "MPLCONFIGDIR" in os.environ:
        return
    cache_dir = Path(tempfile.gettempdir()) / "darp-matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(cache_dir)
