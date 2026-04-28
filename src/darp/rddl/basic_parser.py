"""A dependency-free, structural RDDL parser with DOT AST output."""

# TODO(parser): This parser intentionally recognizes structural RDDL blocks
# rather than the full expression grammar. Extend it before using it as the
# production compiler input.
# TODO(parser): Add DARP-RDDL extension keywords after the syntax is designed.

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from darp.rddl.ast import RDDLASTNode


class RDDLParseError(ValueError):
    """Raised when the built-in parser cannot read an RDDL file."""


@dataclass(frozen=True)
class Token:
    value: str
    line: int
    column: int


class BasicRDDLParser:
    """Parse enough RDDL structure to inspect domains, instances, and blocks."""

    def parse_files(self, *paths: str | Path) -> RDDLASTNode:
        root = RDDLASTNode("rddl", "files")
        for path in paths:
            path = Path(path)
            root.add(self.parse_text(path.read_text(encoding="utf-8"), source=str(path)))
        return root

    def parse_text(self, text: str, source: str = "<string>") -> RDDLASTNode:
        tokens = _tokenize(text)
        stream = _TokenStream(tokens, source)
        root = RDDLASTNode("file", source)
        while not stream.done:
            root.add(self._parse_top_level(stream))
        return root

    def _parse_top_level(self, stream: "_TokenStream") -> RDDLASTNode:
        kind = stream.expect_identifier().value
        name = stream.expect_identifier().value
        stream.expect("{")
        return self._parse_block(stream, kind=kind, label=name)

    def _parse_block(self, stream: "_TokenStream", kind: str, label: str) -> RDDLASTNode:
        node = RDDLASTNode(kind, label)
        while not stream.peek_is("}"):
            if stream.done:
                raise stream.error("Unexpected end of input while parsing block.")
            node.add(self._parse_statement(stream))
        stream.expect("}")
        stream.consume_if(";")
        return node

    def _parse_statement(self, stream: "_TokenStream") -> RDDLASTNode:
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
    def __init__(self, tokens: list[Token], source: str) -> None:
        self.tokens = tokens
        self.source = source
        self.index = 0

    @property
    def done(self) -> bool:
        return self.index >= len(self.tokens)

    def peek_is(self, value: str) -> bool:
        return not self.done and self.tokens[self.index].value == value

    def consume_if(self, value: str) -> bool:
        if self.peek_is(value):
            self.index += 1
            return True
        return False

    def expect(self, value: str) -> Token:
        token = self.expect_any()
        if token.value != value:
            raise self.error(f"Expected {value!r}, found {token.value!r}.", token)
        return token

    def expect_identifier(self) -> Token:
        token = self.expect_any()
        if not _is_identifier(token.value):
            raise self.error(f"Expected identifier, found {token.value!r}.", token)
        return token

    def expect_any(self) -> Token:
        if self.done:
            raise self.error("Unexpected end of input.")
        token = self.tokens[self.index]
        self.index += 1
        return token

    def push_back(self, token: Token) -> None:
        if self.index == 0 or self.tokens[self.index - 1] != token:
            raise self.error("Internal parser error while pushing back token.", token)
        self.index -= 1

    def error(self, message: str, token: Token | None = None) -> RDDLParseError:
        token = token or (self.tokens[-1] if self.tokens else Token("", 1, 1))
        return RDDLParseError(f"{self.source}:{token.line}:{token.column}: {message}")


def _tokenize(text: str) -> list[Token]:
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
    return bool(value) and (value[0].isalpha() or value[0] == "_")


def _join_tokens(parts: list[str]) -> str:
    text = " ".join(parts)
    for before in [" ,", " ;", " )", " ]", " }"]:
        text = text.replace(before, before.strip())
    for after in ["( ", "[ ", "{ "]:
        text = text.replace(after, after.strip())
    return text


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Parse RDDL files and optionally print DOT AST.")
    parser.add_argument("paths", nargs="+", help="RDDL domain/instance files to parse")
    parser.add_argument("--dot", action="store_true", help="print the AST as Graphviz DOT")
    parser.add_argument("--dot-output", help="write the DOT graph to a file")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    ast = BasicRDDLParser().parse_files(*args.paths)
    print(f"RDDL parse succeeded: {ast.summary()}")
    if args.dot or args.dot_output:
        dot = ast.to_dot()
        if args.dot_output:
            Path(args.dot_output).write_text(dot + "\n", encoding="utf-8")
            print(f"DOT AST written to {args.dot_output}")
        if args.dot:
            print(dot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
