"""pyrddl frontend for direct RDDL AST parsing."""

# TODO(parser): Validate pyrddl combined domain/instance parsing against larger
# benchmark suites.
# TODO(parser): Use this frontend as the first candidate for a DARP-owned parser
# fork if extended syntax cannot be layered cleanly on pyRDDLGym.

from __future__ import annotations

from pathlib import Path

from darp.rddl.basic_parser import BasicRDDLParser
from darp.rddl.frontend import ParsedRDDL, RDDLFrontendError, frontend_error, rddl_path


class PyRDDLFrontend:
    """Adapt pyrddl parsing to DARP's frontend container. / 将 pyrddl 解析适配到 DARP frontend 容器。"""

    name = "pyrddl"
    supports_extended_syntax = False

    def parse(self, domain: str | Path, instance: str | Path) -> ParsedRDDL:
        """Parse RDDL with pyrddl and return a shared container. / 使用 pyrddl 解析 RDDL 并返回统一容器。"""
        domain_path = rddl_path(domain)
        instance_path = rddl_path(instance)
        try:
            from pyrddl.parser import RDDLParser
        except ImportError as exc:
            raise RDDLFrontendError(
                "pyrddl is required for the pyrddl frontend. "
                "Install with `pip install -e .[pyrddl]`."
            ) from exc

        try:
            domain_text = domain_path.read_text(encoding="utf-8")
            instance_text = instance_path.read_text(encoding="utf-8")
            canonical_ast = BasicRDDLParser().parse_files(domain_path, instance_path)
            parser = RDDLParser()
            parser.build()
            native_ast = parser.parse(f"{domain_text}\n\n{instance_text}")
        except Exception as exc:
            raise frontend_error(self.name, domain_path, instance_path, exc) from exc
        return ParsedRDDL(
            frontend=self.name,
            domain=str(domain_path),
            instance=str(instance_path),
            ast=canonical_ast,
            native_ast=native_ast,
            metadata={
                "source": "pyrddl",
                "ast_type": type(canonical_ast).__name__,
                "native_ast_type": type(native_ast).__name__,
            },
        )
