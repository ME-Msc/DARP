"""Small AST model and DOT export for the built-in RDDL parser."""

# TODO(parser): Replace the generic node attributes with typed RDDL AST nodes
# once the DARP-RDDL extension grammar is stable.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


@dataclass
class RDDLASTNode:
    kind: str
    label: str
    children: list["RDDLASTNode"] = field(default_factory=list)

    def add(self, child: "RDDLASTNode") -> "RDDLASTNode":
        self.children.append(child)
        return child

    def walk(self) -> Iterable["RDDLASTNode"]:
        yield self
        for child in self.children:
            yield from child.walk()

    def summary(self) -> str:
        counts: dict[str, int] = {}
        for node in self.walk():
            counts[node.kind] = counts.get(node.kind, 0) + 1
        parts = ", ".join(f"{kind}={count}" for kind, count in sorted(counts.items()))
        return f"nodes={sum(counts.values())}; {parts}"

    def to_dot(self) -> str:
        lines = [
            "digraph RDDLAST {",
            "  rankdir=TB;",
            '  node [shape=box, style="rounded,filled", fillcolor="#f7fbff"];',
        ]
        node_ids: dict[int, str] = {}

        def visit(node: RDDLASTNode) -> str:
            node_id = node_ids.get(id(node))
            if node_id is not None:
                return node_id
            node_id = f"n{len(node_ids)}"
            node_ids[id(node)] = node_id
            label = _escape_dot(f"{node.kind}\\n{node.label}")
            lines.append(f'  {node_id} [label="{label}"];')
            for child in node.children:
                child_id = visit(child)
                lines.append(f"  {node_id} -> {child_id};")
            return node_id

        visit(self)
        lines.append("}")
        return "\n".join(lines)


def _escape_dot(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')
