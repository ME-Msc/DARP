"""Shared RDDL lexical constants for parsing and display."""

# TODO(phase-4.1): Extend these constants when the compiler moves from compact
# grounding to a fuller factored RDDL grammar.

from __future__ import annotations

RDDL_KEYWORDS = frozenset(
    {
        "action-fluent",
        "bool",
        "cpfs",
        "default",
        "derived-fluent",
        "discount",
        "domain",
        "else",
        "exists",
        "false",
        "forall",
        "horizon",
        "if",
        "init-state",
        "int",
        "instance",
        "interm-fluent",
        "max-nondef-actions",
        "non-fluent",
        "non-fluents",
        "object",
        "objects",
        "observ-fluent",
        "pvariables",
        "real",
        "requirements",
        "reward",
        "reward-deterministic",
        "state-fluent",
        "sum",
        "then",
        "true",
        "types",
    }
)
"""Known RDDL keywords used by parser validation and display. / parser 校验和显示共用的 RDDL 关键字。"""

RDDL_TOP_LEVEL_BLOCKS = frozenset({"domain", "instance", "non-fluents"})
"""Top-level RDDL block names accepted by the basic parser. / 基础 parser 接受的顶层 RDDL 块名。"""

RDDL_PUNCTUATION = frozenset("{}()[];,=:^|&!<>+*/")
"""Single-character punctuation tokens recognized by the basic parser. / 基础 parser 识别的单字符标点 token。"""

RDDL_TWO_CHAR_OPERATORS = frozenset({"==", "!=", "<=", ">=", "=>"})
"""Two-character operators recognized before single punctuation. / 优先于单字符标点识别的双字符操作符。"""


def is_rddl_keyword(value: str) -> bool:
    """Return whether text is a known RDDL keyword. / 判断文本是否为已知 RDDL 关键字。"""
    return value in RDDL_KEYWORDS
