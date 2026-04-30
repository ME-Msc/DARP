"""DARP-owned parser frontend for future extended RDDL syntax."""

# TODO(parser): Define the concrete DARP-RDDL grammar extensions for durative
# actions, risk constraints, and HILP annotations.
# TODO(parser): Decide whether this frontend subclasses a pyRDDLGym parser,
# wraps pyrddl, or uses a DARP-owned grammar implementation.

from __future__ import annotations

from pathlib import Path

from darp.rddl.basic_parser import BasicRDDLParser
from darp.rddl.frontend import ParsedRDDL


class DARPExtendedFrontend:
    """Reserve DARP-owned parsing for future extended syntax. / 为未来 DARP 扩展语法预留自有解析入口。"""

    name = "darp"
    supports_extended_syntax = True

    def parse(self, domain: str | Path, instance: str | Path) -> ParsedRDDL:
        """Parse files with the DARP-owned structural parser. / 使用 DARP 自有结构 parser 解析文件。"""
        ast = BasicRDDLParser().parse_files(domain, instance)
        return ParsedRDDL(
            frontend=self.name,
            domain=str(domain),
            instance=str(instance),
            ast=ast,
            metadata={
                "source": "darp-basic-parser",
                "extended_syntax": "reserved",
            },
        )
