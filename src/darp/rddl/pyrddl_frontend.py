"""pyrddl frontend for direct RDDL AST parsing."""

# TODO(phase-2): Validate whether pyrddl expects a combined domain/instance
# file or separate snippets for the RDDL suites we care about.
# TODO(parser): Use this frontend as the first candidate for a DARP-owned parser
# fork if extended syntax cannot be layered cleanly on pyRDDLGym.

from __future__ import annotations

from pathlib import Path

from darp.rddl.frontend import ParsedRDDL, RDDLFrontendError


class PyRDDLFrontend:
    """Adapt pyrddl parsing to DARP's frontend container. / 将 pyrddl 解析适配到 DARP frontend 容器。"""

    name = "pyrddl"
    supports_extended_syntax = False

    def parse(self, domain: str | Path, instance: str | Path) -> ParsedRDDL:
        """Parse RDDL with pyrddl and return a shared container. / 使用 pyrddl 解析 RDDL 并返回统一容器。"""
        try:
            from pyrddl.parser import RDDLParser
        except ImportError as exc:
            raise RDDLFrontendError(
                "pyrddl is required for the pyrddl frontend. "
                "Install with `pip install -e .[pyrddl]`."
            ) from exc

        domain_text = Path(domain).read_text(encoding="utf-8")
        instance_text = Path(instance).read_text(encoding="utf-8")
        parser = RDDLParser()
        parser.build()
        ast = parser.parse(f"{domain_text}\n\n{instance_text}")
        return ParsedRDDL(
            frontend=self.name,
            domain=str(domain),
            instance=str(instance),
            ast=ast,
            metadata={"source": "pyrddl"},
        )
