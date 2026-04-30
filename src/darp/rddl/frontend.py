"""Parser frontend protocol for standard and extended RDDL inputs."""

# TODO(phase-2.4): Add typed AST accessors when compiler grounding expands
# beyond the structural subset.
# TODO(phase-8.2): Add DARP-specific AST nodes once extended RDDL syntax is defined.

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


class RDDLFrontendError(RuntimeError):
    """Raised when a parser frontend cannot parse RDDL. / 在 parser frontend 无法解析 RDDL 时抛出。"""


@dataclass(frozen=True)
class ParsedRDDL:
    """Carry parser output in a frontend-neutral shape. / 以前端无关的形式承载 parser 输出。"""

    frontend: str
    domain: str
    instance: str
    ast: Any | None = None
    native_ast: Any | None = None
    model: Any | None = None
    env: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def artifact_summary(self) -> dict[str, str | None]:
        """Summarize available parser artifacts by type. / 按类型汇总当前可用的解析产物。"""
        return {
            "ast": type(self.ast).__name__ if self.ast is not None else None,
            "native_ast": type(self.native_ast).__name__ if self.native_ast is not None else None,
            "model": type(self.model).__name__ if self.model is not None else None,
            "env": type(self.env).__name__ if self.env is not None else None,
        }

    def to_summary_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly summary for CLI inspection. / 返回适合 CLI 检查的 JSON 友好摘要。"""
        return {
            "frontend": self.frontend,
            "domain": self.domain,
            "instance": self.instance,
            "artifacts": self.artifact_summary(),
            "metadata": self.metadata,
        }


class RDDLFrontend(Protocol):
    """Define the shared parser frontend interface. / 定义通用 parser frontend 接口。"""

    name: str
    supports_extended_syntax: bool

    def parse(self, domain: str | Path, instance: str | Path) -> ParsedRDDL:
        """Parse a domain/instance pair into a shared container. / 将 domain/instance 解析为统一容器。"""
        raise NotImplementedError


def rddl_path(path: str | Path) -> Path:
    """Normalize one RDDL file path without requiring absolutes. / 规范化一个 RDDL 文件路径但不强制转为绝对路径。"""
    return Path(path).expanduser()


def frontend_error(frontend: str, domain: Path, instance: Path, exc: Exception) -> RDDLFrontendError:
    """Wrap parser failures with frontend and file context. / 用 frontend 和文件上下文包装解析失败。"""
    return RDDLFrontendError(
        f"{frontend} failed to parse domain={domain} instance={instance}: {exc}"
    )
