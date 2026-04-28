"""Parser frontend protocol for standard and extended RDDL inputs."""

# TODO(phase-2): Stabilize the ParsedRDDL fields needed by the compiler after
# evaluating pyRDDLGym and pyrddl AST shapes on real benchmarks.
# TODO(parser): Add DARP-specific AST nodes once extended RDDL syntax is defined.

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


class RDDLFrontendError(RuntimeError):
    """Raised when a parser frontend cannot parse an RDDL input."""


@dataclass(frozen=True)
class ParsedRDDL:
    """A parser-independent container returned by all RDDL frontends."""

    frontend: str
    domain: str
    instance: str
    ast: Any | None = None
    model: Any | None = None
    env: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class RDDLFrontend(Protocol):
    """Protocol implemented by every DARP RDDL parser frontend.

    A frontend is the adapter between a concrete parser implementation and
    DARP's compiler. The compiler should consume ParsedRDDL instead of importing
    pyRDDLGym, pyrddl, or a future DARP parser directly.
    """

    name: str
    supports_extended_syntax: bool

    def parse(self, domain: str | Path, instance: str | Path) -> ParsedRDDL:
        """Parse a domain/instance pair into a parser-independent container."""
        raise NotImplementedError
