"""RDDL loading through pyRDDLGym."""

# TODO(phase-9.1): Support pyRDDLGym repository names in addition to explicit
# domain and instance file paths for benchmark runs.

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

from darp.rddl.artifacts import RDDLArtifacts, RDDLLoadError, rddl_load_error, rddl_path


class RDDLLoader:
    """Load standard RDDL with pyRDDLGym. / 使用 pyRDDLGym 加载标准 RDDL。"""

    def load(self, domain: str | Path, instance: str | Path) -> RDDLArtifacts:
        """Load a domain/instance pair with pyRDDLGym. / 使用 pyRDDLGym 加载 domain/instance 文件对。"""
        domain_path = rddl_path(domain)
        instance_path = rddl_path(instance)
        _ensure_matplotlib_cache_dir()
        try:
            import pyRDDLGym
        except ImportError as exc:
            raise RDDLLoadError(
                "pyRDDLGym is required to load RDDL. "
                "Install with `pip install -e .` or `pip install -r requirements.txt`."
            ) from exc

        try:
            env = pyRDDLGym.make(str(domain_path), str(instance_path))
        except Exception as exc:
            raise rddl_load_error(domain_path, instance_path, exc) from exc
        model = getattr(env, "model", None)
        native_ast = getattr(model, "ast", None)
        return RDDLArtifacts(
            domain=str(domain_path),
            instance=str(instance_path),
            native_ast=native_ast,
            env=env,
            model=model,
            metadata={
                "source": "pyRDDLGym",
                "pyRDDLGym_version": getattr(pyRDDLGym, "__version__", None),
                "native_ast_type": type(native_ast).__name__ if native_ast is not None else None,
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


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for RDDL inspection. / 构建用于检查 RDDL 加载结果的命令行 parser。"""
    parser = argparse.ArgumentParser(description="Load standard RDDL through pyRDDLGym.")
    parser.add_argument("domain", help="RDDL domain file")
    parser.add_argument("instance", help="RDDL instance file")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the RDDL artifact inspection command. / 运行 RDDL 产物检查命令。"""
    args = build_parser().parse_args(argv)
    loaded = RDDLLoader().load(args.domain, args.instance)
    print(json.dumps(loaded.to_summary_dict(), indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
