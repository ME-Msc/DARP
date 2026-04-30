"""Parser frontend protocol for standard and extended RDDL inputs."""

# TODO(phase-2): Stabilize the ParsedRDDL fields needed by the compiler after
# evaluating pyRDDLGym and pyrddl AST shapes on real benchmarks.
# TODO(parser): Add DARP-specific AST nodes once extended RDDL syntax is defined.

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
    model: Any | None = None
    env: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class RDDLFrontend(Protocol):
    """Define the shared parser frontend interface. / 定义通用 parser frontend 接口。"""

    name: str
    supports_extended_syntax: bool

    def parse(self, domain: str | Path, instance: str | Path) -> ParsedRDDL:
        """Parse a domain/instance pair into a shared container. / 将 domain/instance 解析为统一容器。"""
        raise NotImplementedError
