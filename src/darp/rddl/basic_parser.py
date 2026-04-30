"""A dependency-free, structural RDDL parser with HTML AST output."""

# TODO(parser): This parser intentionally recognizes structural RDDL blocks
# rather than the full expression grammar. Extend it before using it as the
# production compiler input.
# TODO(parser): Add DARP-RDDL extension keywords after the syntax is designed.

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import webbrowser

from darp.rddl.ast import RDDLASTNode


class RDDLParseError(ValueError):
    """Raised when the built-in parser cannot read RDDL. / 在内置 parser 无法读取 RDDL 时抛出。"""


@dataclass(frozen=True)
class Token:
    """Store one lexical token with source position. / 保存一个带源码位置的词法 token。"""

    value: str
    line: int
    column: int


class BasicRDDLParser:
    """Parse structural RDDL for AST inspection. / 解析结构化 RDDL 以便检查 AST。"""

    def parse_files(self, *paths: str | Path) -> RDDLASTNode:
        """Parse one or more RDDL files into a shared AST root. / 将一个或多个 RDDL 文件解析到同一个 AST 根节点。"""
        root = RDDLASTNode("rddl", "files")
        for path in paths:
            path = Path(path)
            root.add(self.parse_text(path.read_text(encoding="utf-8"), source=str(path)))
        return root

    def parse_text(self, text: str, source: str = "<string>") -> RDDLASTNode:
        """Parse RDDL text from one source into a file node. / 将单个来源的 RDDL 文本解析为 file 节点。"""
        tokens = _tokenize(text)
        stream = _TokenStream(tokens, source)
        root = RDDLASTNode("file", source)
        while not stream.done:
            root.add(self._parse_top_level(stream))
        return root

    def _parse_top_level(self, stream: "_TokenStream") -> RDDLASTNode:
        """Parse a top-level domain or instance block. / 解析顶层 domain 或 instance 块。"""
        kind = stream.expect_identifier().value
        name = stream.expect_identifier().value
        stream.expect("{")
        return self._parse_block(stream, kind=kind, label=name)

    def _parse_block(self, stream: "_TokenStream", kind: str, label: str) -> RDDLASTNode:
        """Parse a braced block and its child statements. / 解析花括号块及其内部语句。"""
        node = RDDLASTNode(kind, label)
        while not stream.peek_is("}"):
            if stream.done:
                raise stream.error("Unexpected end of input while parsing block.")
            node.add(self._parse_statement(stream))
        stream.expect("}")
        stream.consume_if(";")
        return node

    def _parse_statement(self, stream: "_TokenStream") -> RDDLASTNode:
        """Parse one structural statement or nested block. / 解析一条结构语句或嵌套块。"""
        first = stream.expect_any()
        if stream.peek_is("{"):
            stream.expect("{")
            return self._parse_block(stream, kind="block", label=first.value)
        if stream.consume_if("="):
            expr = self._collect_until_statement_end(stream)
            return RDDLASTNode("assignment", f"{first.value} = {expr}")
        expr = self._collect_until_statement_end(stream, initial=[first.value])
        return RDDLASTNode("statement", expr)

    def _collect_until_statement_end(
        self, stream: "_TokenStream", initial: list[str] | None = None
    ) -> str:
        """Collect tokens until a top-level semicolon or closing brace. / 收集 token 直到顶层分号或右花括号。"""
        parts = list(initial or [])
        depth = 0
        while not stream.done:
            token = stream.expect_any()
            if token.value in {"{", "(", "["}:
                depth += 1
            elif token.value in {"}", ")", "]"}:
                if token.value == "}" and depth == 0:
                    stream.push_back(token)
                    break
                depth = max(0, depth - 1)
            if token.value == ";" and depth == 0:
                break
            parts.append(token.value)
        return _join_tokens(parts)


class _TokenStream:
    """Provide cursor-style access to parser tokens. / 为 parser token 提供游标式访问。"""

    def __init__(self, tokens: list[Token], source: str) -> None:
        """Create a cursor over parser tokens. / 创建一个遍历 parser token 的游标。"""
        self.tokens = tokens
        self.source = source
        self.index = 0

    @property
    def done(self) -> bool:
        """Return whether all tokens were consumed. / 返回所有 token 是否已读取完。"""
        return self.index >= len(self.tokens)

    def peek_is(self, value: str) -> bool:
        """Check the next token without consuming it. / 在不消耗 token 的情况下检查下一个值。"""
        return not self.done and self.tokens[self.index].value == value

    def consume_if(self, value: str) -> bool:
        """Consume the next token only when it matches. / 仅在匹配时消耗下一个 token。"""
        if self.peek_is(value):
            self.index += 1
            return True
        return False

    def expect(self, value: str) -> Token:
        """Consume the next token and require an exact value. / 消耗下一个 token 并要求值完全匹配。"""
        token = self.expect_any()
        if token.value != value:
            raise self.error(f"Expected {value!r}, found {token.value!r}.", token)
        return token

    def expect_identifier(self) -> Token:
        """Consume the next token and require an identifier. / 消耗下一个 token 并要求它是标识符。"""
        token = self.expect_any()
        if not _is_identifier(token.value):
            raise self.error(f"Expected identifier, found {token.value!r}.", token)
        return token

    def expect_any(self) -> Token:
        """Consume and return the next token. / 消耗并返回下一个 token。"""
        if self.done:
            raise self.error("Unexpected end of input.")
        token = self.tokens[self.index]
        self.index += 1
        return token

    def push_back(self, token: Token) -> None:
        """Move one consumed token back into the stream. / 将刚消耗的一个 token 放回流中。"""
        if self.index == 0 or self.tokens[self.index - 1] != token:
            raise self.error("Internal parser error while pushing back token.", token)
        self.index -= 1

    def error(self, message: str, token: Token | None = None) -> RDDLParseError:
        """Build a parse error with source location. / 构造包含源码位置的解析错误。"""
        token = token or (self.tokens[-1] if self.tokens else Token("", 1, 1))
        return RDDLParseError(f"{self.source}:{token.line}:{token.column}: {message}")


def _tokenize(text: str) -> list[Token]:
    """Tokenize structural RDDL text with line and column positions. / 将结构化 RDDL 文本切分为带行列位置的 token。"""
    tokens: list[Token] = []
    line = 1
    column = 1
    i = 0
    punctuation = set("{}()[];,=:")
    while i < len(text):
        char = text[i]
        if char == "\n":
            line += 1
            column = 1
            i += 1
            continue
        if char.isspace():
            column += 1
            i += 1
            continue
        if char == "#":
            while i < len(text) and text[i] != "\n":
                i += 1
                column += 1
            continue
        if char == "/" and i + 1 < len(text) and text[i + 1] == "/":
            while i < len(text) and text[i] != "\n":
                i += 1
                column += 1
            continue
        if char in punctuation:
            tokens.append(Token(char, line, column))
            i += 1
            column += 1
            continue
        start = i
        start_column = column
        while i < len(text):
            current = text[i]
            if current.isspace() or current in punctuation:
                break
            if current == "#" or (current == "/" and i + 1 < len(text) and text[i + 1] == "/"):
                break
            i += 1
            column += 1
        tokens.append(Token(text[start:i], line, start_column))
    return tokens


def _is_identifier(value: str) -> bool:
    """Return whether a token can start an identifier. / 判断 token 是否可以作为标识符开头。"""
    return bool(value) and (value[0].isalpha() or value[0] == "_")


def _join_tokens(parts: list[str]) -> str:
    """Join token text while cleaning simple spacing. / 拼接 token 文本并清理简单空格。"""
    text = " ".join(parts)
    for before in [" ,", " ;", " )", " ]", " }"]:
        text = text.replace(before, before.strip())
    for after in ["( ", "[ ", "{ "]:
        text = text.replace(after, after.strip())
    return text


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for AST inspection. / 构建用于 AST 检查的命令行 parser。"""
    parser = argparse.ArgumentParser(description="Parse RDDL files and inspect their AST.")
    parser.add_argument("paths", nargs="+", help="RDDL domain/instance files to parse")
    parser.add_argument("--html-output", help="write a standalone graphical HTML AST visualizer")
    parser.add_argument(
        "--open",
        action="store_true",
        help="open the generated HTML visualizer; implies --html-output rddl_ast.html if omitted",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the command-line parser entrypoint. / 运行命令行解析入口。"""
    args = build_parser().parse_args(argv)
    ast = BasicRDDLParser().parse_files(*args.paths)
    print(f"RDDL parse succeeded: {ast.summary()}")
    html_output = args.html_output
    if args.open and not html_output:
        html_output = "rddl_ast.html"
    if html_output:
        from darp.rddl.visualizer import write_html

        output = write_html(html_output, ast, title="RDDL AST")
        print(f"HTML AST visualizer written to {output}")
        if args.open:
            webbrowser.open(output.resolve().as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
