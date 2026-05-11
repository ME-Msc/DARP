"""Containers and errors for pyRDDLGym-loaded RDDL artifacts."""

# TODO(phase-4.1): Add typed metadata accessors for finite-discrete enumerability
# checks when the explicit PlanningProblem path starts.

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class RDDLLoadError(RuntimeError):
    """Raised when pyRDDLGym cannot load RDDL. / pyRDDLGym 无法加载 RDDL 时抛出。"""


@dataclass(frozen=True)
class RDDLArtifacts:
    """Carry pyRDDLGym parser/runtime artifacts. / 承载 pyRDDLGym parser/runtime 产物。"""

    domain: str
    instance: str
    native_ast: Any | None = None
    model: Any | None = None
    env: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def artifact_summary(self) -> dict[str, str | None]:
        """Summarize available pyRDDLGym artifacts by type. / 按类型汇总 pyRDDLGym 产物。"""
        return {
            "native_ast": type(self.native_ast).__name__ if self.native_ast is not None else None,
            "model": type(self.model).__name__ if self.model is not None else None,
            "env": type(self.env).__name__ if self.env is not None else None,
        }

    def to_summary_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly summary for CLI inspection. / 返回适合 CLI 检查的 JSON 友好摘要。"""
        return {
            "source": self.metadata.get("source", "pyRDDLGym"),
            "domain": self.domain,
            "instance": self.instance,
            "artifacts": self.artifact_summary(),
            "metadata": self.metadata,
        }


def rddl_path(path: str | Path) -> Path:
    """Normalize one RDDL file path without requiring absolutes. / 规范化 RDDL 文件路径但不强制绝对路径。"""
    return Path(path).expanduser()


def rddl_load_error(domain: Path, instance: Path, exc: Exception) -> RDDLLoadError:
    """Wrap pyRDDLGym load failures with file context. / 用文件上下文包装 pyRDDLGym 加载失败。"""
    return RDDLLoadError(f"pyRDDLGym failed to load domain={domain} instance={instance}: {exc}")
