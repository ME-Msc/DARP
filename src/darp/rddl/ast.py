"""Small AST model for the built-in RDDL parser."""

# TODO(phase-8.2): Replace the generic node attributes with typed RDDL AST nodes
# once the DARP-RDDL extension grammar is stable.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


@dataclass
class RDDLASTNode:
    """Represent one generic RDDL AST node. / 表示一个通用 RDDL AST 节点。"""

    kind: str
    label: str
    children: list["RDDLASTNode"] = field(default_factory=list)
    line: int | None = None
    column: int | None = None
    end_line: int | None = None
    end_column: int | None = None

    def add(self, child: "RDDLASTNode") -> "RDDLASTNode":
        """Add a child node and return it. / 添加子节点并返回该节点。"""
        self.children.append(child)
        return child

    def walk(self) -> Iterable["RDDLASTNode"]:
        """Yield this node and all descendants depth-first. / 以深度优先顺序遍历当前节点和后代。"""
        yield self
        for child in self.children:
            yield from child.walk()

    def summary(self) -> str:
        """Return compact node counts for this AST. / 返回当前 AST 的节点数量摘要。"""
        counts: dict[str, int] = {}
        for node in self.walk():
            counts[node.kind] = counts.get(node.kind, 0) + 1
        parts = ", ".join(f"{kind}={count}" for kind, count in sorted(counts.items()))
        return f"nodes={sum(counts.values())}; {parts}"
