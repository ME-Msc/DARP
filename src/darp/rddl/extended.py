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
    name = "darp"
    supports_extended_syntax = True

    def parse(self, domain: str | Path, instance: str | Path) -> ParsedRDDL:
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
