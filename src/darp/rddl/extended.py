"""DARP-owned parser frontend for future extended RDDL syntax."""

# TODO(phase-8.1): Define the concrete DARP-RDDL grammar extensions for durative
# actions, risk constraints, and HILP annotations.
# TODO(phase-8.2): Decide whether this frontend subclasses a pyRDDLGym parser,
# wraps pyrddl, or uses a DARP-owned grammar implementation.

from __future__ import annotations

from pathlib import Path

from darp.rddl.basic_parser import BasicRDDLParser
from darp.rddl.frontend import ParsedRDDL, frontend_error, rddl_path


class DARPExtendedFrontend:
    """Reserve DARP-owned parsing for future extended syntax. / 为未来 DARP 扩展语法预留自有解析入口。"""

    name = "darp"
    supports_extended_syntax = True

    def parse(self, domain: str | Path, instance: str | Path) -> ParsedRDDL:
        """Parse files with the DARP-owned structural parser. / 使用 DARP 自有结构 parser 解析文件。"""
        domain_path = rddl_path(domain)
        instance_path = rddl_path(instance)
        try:
            ast = BasicRDDLParser().parse_files(domain_path, instance_path)
        except Exception as exc:
            raise frontend_error(self.name, domain_path, instance_path, exc) from exc
        return ParsedRDDL(
            frontend=self.name,
            domain=str(domain_path),
            instance=str(instance_path),
            ast=ast,
            metadata={
                "source": "darp-basic-parser",
                "extended_syntax": "reserved",
                "ast_type": type(ast).__name__,
            },
        )
