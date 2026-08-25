"""RDDL loading through pyRDDLGym."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile

from darp.adapter.problem import PyRDDLGymProblem, RDDLLoadError


def load_rddl(domain: str | Path, instance: str | Path) -> PyRDDLGymProblem:
    """Load one RDDL domain/instance pair through pyRDDLGym."""
    domain_path = Path(domain).expanduser()
    instance_path = Path(instance).expanduser()
    _ensure_matplotlib_cache_dir()
    try:
        import pyRDDLGym
    except ImportError as exc:
        raise RDDLLoadError(
            "pyRDDLGym is required to load RDDL. "
            "Install with `pip install -e .` or run `tools/install.sh`."
        ) from exc

    try:
        env = pyRDDLGym.make(str(domain_path), str(instance_path))
    except Exception as exc:
        raise RDDLLoadError(
            "pyRDDLGym failed to load "
            f"domain={domain_path} instance={instance_path}: {exc}"
        ) from exc
    model = getattr(env, "model", None)
    if model is None:
        raise RDDLLoadError("pyRDDLGym environment did not expose a grounded model source.")
    native_ast = getattr(model, "ast", None)
    if native_ast is None:
        raise RDDLLoadError("pyRDDLGym model did not expose the RDDL AST required for grounding.")
    return PyRDDLGymProblem(native_ast=native_ast, env=env)


def _ensure_matplotlib_cache_dir() -> None:
    """Give pyRDDLGym's matplotlib import a writable cache. / 为 pyRDDLGym 的 matplotlib 导入提供可写缓存。"""
    if "MPLCONFIGDIR" in os.environ:
        return
    cache_dir = Path(tempfile.gettempdir()) / "darp-matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(cache_dir)
