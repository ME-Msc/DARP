"""Small RDDL expression parser and evaluator for grounding."""

# TODO(phase-6.2): Preserve expression structure in preprocessing so AND-OR
# expansion can reuse grounded CPF dependencies instead of reparsing text.

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from dataclasses import replace
import math
import re
from typing import Mapping, Protocol

from darp.core.types import Action, GroundAtom, State
from darp.rddl.ast import RDDLASTNode

ExpressionValue = bool | float | str

TOKEN_RE = re.compile(
    r"\s+|==|!=|<=|>=|=>|\d+(?:\.\d+)?|@[A-Za-z0-9_-]+|\?[A-Za-z0-9_-]+|"
    r"[A-Za-z_][A-Za-z0-9_'\-]*|[{}\[\]():^|&!<>+\-*/~,=]|."
)

AGGREGATE_NAMES = frozenset({"sum", "prod", "avg", "forall", "exists"})
BUILTIN_CALLS = frozenset(
    {
        "Bernoulli",
        "DiracDelta",
        "KronDelta",
        "abs",
        "ceil",
        "floor",
        "max",
        "min",
        "round",
    }
)
PROBABILITY_DISTRIBUTIONS = frozenset({"Bernoulli"})
"""Stochastic distribution calls recognized by requirement checks. / requirement 检查识别的随机分布调用。"""


class RDDLExpressionError(ValueError):
    """Raised when a supported RDDL expression cannot be parsed. / 在支持范围内的 RDDL 表达式无法解析时抛出。"""


class Expression(Protocol):
    """Evaluate one parsed expression against a grounding context. / 在 grounding 上下文中求值一个解析表达式。"""

    def evaluate(self, context: "EvaluationContext") -> ExpressionValue:
        """Return the expression value in one context. / 返回表达式在某个上下文中的值。"""


@dataclass(frozen=True)
class EvaluationContext:
    """Carry values needed to evaluate grounded RDDL expressions. / 承载求值 grounded RDDL 表达式所需的值。"""

    state_fluent: str
    current_state: State
    action: Action
    actions: tuple[Action, ...]
    non_fluents: frozenset[tuple[str, tuple[str, ...]]]
    variables: dict[str, str]
    fluent_values: Mapping[tuple[str, tuple[str, ...]], ExpressionValue] = field(default_factory=dict)
    objects: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    state_atoms: frozenset[GroundAtom] = field(default_factory=frozenset)
    state_pvariables: frozenset[str] = field(default_factory=frozenset)
    action_fluents: frozenset[tuple[str, tuple[str, ...]]] = field(default_factory=frozenset)
    action_pvariables: frozenset[str] = field(default_factory=frozenset)

    def bind(self, variable: str, value: str) -> "EvaluationContext":
        """Return a context with one additional variable binding. / 返回绑定一个额外变量后的上下文。"""
        variables = dict(self.variables)
        variables[variable] = value
        return replace(self, variables=variables)


@dataclass(frozen=True)
class LiteralExpression:
    """Represent a literal bool, number, object, or symbol. / 表示 bool、数字、对象或符号 literal。"""

    value: ExpressionValue

    def evaluate(self, context: EvaluationContext) -> ExpressionValue:
        """Return the literal value. / 返回 literal 值。"""
        if isinstance(self.value, str) and self.value in context.variables:
            return context.variables[self.value]
        if isinstance(self.value, str) and self.value in context.state_pvariables:
            return (self.value, ()) in context.state_atoms
        if isinstance(self.value, str) and self.value in context.action_pvariables:
            return (self.value, ()) in context.action_fluents
        if isinstance(self.value, str) and (self.value, ()) in context.fluent_values:
            return context.fluent_values[(self.value, ())]
        if isinstance(self.value, str) and (self.value, ()) in context.non_fluents:
            return True
        if isinstance(self.value, str) and self.value in context.actions:
            return context.action == self.value
        return self.value


@dataclass(frozen=True)
class CallExpression:
    """Represent a pvariable/function-style call. / 表示 pvariable 或函数形式调用。"""

    name: str
    args: tuple[Expression, ...]

    def evaluate(self, context: EvaluationContext) -> ExpressionValue:
        """Evaluate a grounded fluent call. / 求值一个 grounded fluent 调用。"""
        raw_values = tuple(arg.evaluate(context) for arg in self.args)
        if self.name in BUILTIN_CALLS:
            return _evaluate_builtin(self.name, raw_values)
        values = tuple(_object_name(value) for value in raw_values)
        if self.name == context.state_fluent:
            if len(values) != 1:
                raise RDDLExpressionError(f"State fluent {self.name!r} expects one argument.")
            return values[0] == context.current_state
        if self.name in context.state_pvariables:
            return (self.name, values) in context.state_atoms
        if self.name in context.actions and not values:
            return context.action == self.name
        if self.name in context.action_pvariables:
            return (self.name, values) in context.action_fluents
        if (self.name, values) in context.fluent_values:
            return context.fluent_values[(self.name, values)]
        return (self.name, values) in context.non_fluents


@dataclass(frozen=True)
class UnaryExpression:
    """Represent a unary expression. / 表示一元表达式。"""

    operator: str
    operand: Expression

    def evaluate(self, context: EvaluationContext) -> ExpressionValue:
        """Evaluate the unary expression. / 求值一元表达式。"""
        value = self.operand.evaluate(context)
        if self.operator in {"!", "~", "not"}:
            return not _truthy(value)
        if self.operator == "+":
            return _number(value)
        if self.operator == "-":
            return -_number(value)
        raise RDDLExpressionError(f"Unsupported unary operator {self.operator!r}.")


@dataclass(frozen=True)
class BinaryExpression:
    """Represent a binary expression. / 表示二元表达式。"""

    operator: str
    left: Expression
    right: Expression

    def evaluate(self, context: EvaluationContext) -> ExpressionValue:
        """Evaluate the binary expression. / 求值二元表达式。"""
        if self.operator in {"|", "or"}:
            return _truthy(self.left.evaluate(context)) or _truthy(self.right.evaluate(context))
        if self.operator in {"^", "&", "and"}:
            return _truthy(self.left.evaluate(context)) and _truthy(self.right.evaluate(context))
        if self.operator == "=>":
            return (not _truthy(self.left.evaluate(context))) or _truthy(self.right.evaluate(context))
        left = self.left.evaluate(context)
        right = self.right.evaluate(context)
        if self.operator in {"==", "="}:
            return _object_name(left) == _object_name(right)
        if self.operator == "!=":
            return _object_name(left) != _object_name(right)
        if self.operator == "<":
            return _number(left) < _number(right)
        if self.operator == "<=":
            return _number(left) <= _number(right)
        if self.operator == ">":
            return _number(left) > _number(right)
        if self.operator == ">=":
            return _number(left) >= _number(right)
        if self.operator == "+":
            return _number(left) + _number(right)
        if self.operator == "-":
            return _number(left) - _number(right)
        if self.operator == "*":
            return _number(left) * _number(right)
        if self.operator == "/":
            denominator = _number(right)
            if denominator == 0.0:
                raise RDDLExpressionError("Division by zero in RDDL expression.")
            return _number(left) / denominator
        raise RDDLExpressionError(f"Unsupported binary operator {self.operator!r}.")


@dataclass(frozen=True)
class IfExpression:
    """Represent an if-then-else expression. / 表示 if-then-else 表达式。"""

    condition: Expression
    when_true: Expression
    when_false: Expression

    def evaluate(self, context: EvaluationContext) -> ExpressionValue:
        """Evaluate the selected branch. / 求值被选择的分支。"""
        branch = self.when_true if _truthy(self.condition.evaluate(context)) else self.when_false
        return branch.evaluate(context)


@dataclass(frozen=True)
class AggregateExpression:
    """Represent a finite-object RDDL aggregate. / 表示有限对象上的 RDDL 聚合表达式。"""

    kind: str
    bindings: tuple[tuple[str, str], ...]
    body: Expression

    def evaluate(self, context: EvaluationContext) -> ExpressionValue:
        """Evaluate the aggregate over declared objects. / 在已声明对象上求值聚合。"""
        values = [self.body.evaluate(bound) for bound in _bound_contexts(context, self.bindings)]
        if self.kind == "sum":
            return sum(_number(value) for value in values)
        if self.kind == "prod":
            result = 1.0
            for value in values:
                result *= _number(value)
            return result
        if self.kind == "avg":
            if not values:
                raise RDDLExpressionError("avg aggregate has no grounded objects.")
            return sum(_number(value) for value in values) / len(values)
        if self.kind == "forall":
            return all(_truthy(value) for value in values)
        if self.kind == "exists":
            return any(_truthy(value) for value in values)
        raise RDDLExpressionError(f"Unsupported aggregate {self.kind!r}.")


def parse_expression(text: str) -> Expression:
    """Parse one supported RDDL expression string. / 解析一个当前支持的 RDDL 表达式字符串。"""
    stream = _ExpressionStream(_tokenize_expression(text))
    expression = _parse_if(stream)
    if not stream.done:
        raise RDDLExpressionError(f"Unexpected token {stream.peek()!r} in expression {text!r}.")
    return expression


def expression_uses_distribution(expression: Expression) -> bool:
    """Return whether an expression calls a stochastic distribution. / 判断表达式是否调用随机分布。"""
    if isinstance(expression, CallExpression):
        return expression.name in PROBABILITY_DISTRIBUTIONS or any(
            expression_uses_distribution(argument) for argument in expression.args
        )
    if isinstance(expression, UnaryExpression):
        return expression_uses_distribution(expression.operand)
    if isinstance(expression, BinaryExpression):
        return expression_uses_distribution(expression.left) or expression_uses_distribution(expression.right)
    if isinstance(expression, IfExpression):
        return (
            expression_uses_distribution(expression.condition)
            or expression_uses_distribution(expression.when_true)
            or expression_uses_distribution(expression.when_false)
        )
    if isinstance(expression, AggregateExpression):
        return expression_uses_distribution(expression.body)
    return False


def parse_expression_ast(text: str) -> RDDLASTNode:
    """Parse one supported expression into a visual AST. / 将一个支持的表达式解析为可视化 AST。"""
    return _expression_to_ast(parse_expression(text))


def _expression_to_ast(expression: Expression) -> RDDLASTNode:
    """Convert a parsed expression object into generic AST nodes. / 将解析后的表达式对象转换为通用 AST 节点。"""
    if isinstance(expression, IfExpression):
        node = RDDLASTNode("if", "if")
        condition = node.add(RDDLASTNode("condition", "condition"))
        condition.add(_expression_to_ast(expression.condition))
        when_true = node.add(RDDLASTNode("then", "then"))
        when_true.add(_expression_to_ast(expression.when_true))
        when_false = node.add(RDDLASTNode("else", "else"))
        when_false.add(_expression_to_ast(expression.when_false))
        return node
    if isinstance(expression, BinaryExpression):
        node = RDDLASTNode("operator", expression.operator)
        node.add(_expression_to_ast(expression.left))
        node.add(_expression_to_ast(expression.right))
        return node
    if isinstance(expression, UnaryExpression):
        node = RDDLASTNode("operator", expression.operator)
        node.add(_expression_to_ast(expression.operand))
        return node
    if isinstance(expression, AggregateExpression):
        node = RDDLASTNode("aggregate", expression.kind)
        for variable, type_name in expression.bindings:
            node.add(RDDLASTNode("binding", f"{variable}: {type_name}"))
        body = node.add(RDDLASTNode("body", "body"))
        body.add(_expression_to_ast(expression.body))
        return node
    if isinstance(expression, CallExpression):
        node = RDDLASTNode("call", expression.name)
        for arg in expression.args:
            child = node.add(RDDLASTNode("argument", "argument"))
            child.add(_expression_to_ast(arg))
        return node
    if isinstance(expression, LiteralExpression):
        value = expression.value
        if isinstance(value, bool):
            return RDDLASTNode("literal", str(value).lower())
        if isinstance(value, (int, float)):
            return RDDLASTNode("number", f"{float(value):g}")
        if isinstance(value, str) and value.startswith("?"):
            return RDDLASTNode("variable", value)
        return RDDLASTNode("symbol", str(value))
    return RDDLASTNode("expression", str(expression))


def _parse_if(stream: "_ExpressionStream") -> Expression:
    """Parse if-then-else or lower-precedence expressions. / 解析 if-then-else 或更低层表达式。"""
    if stream.consume_if("if"):
        stream.expect("(")
        condition = _parse_if(stream)
        stream.expect(")")
        stream.expect("then")
        when_true = _parse_if(stream)
        stream.expect("else")
        when_false = _parse_if(stream)
        return IfExpression(condition, when_true, when_false)
    return _parse_implies(stream)


def _parse_implies(stream: "_ExpressionStream") -> Expression:
    """Parse implication expressions. / 解析蕴含表达式。"""
    expression = _parse_or(stream)
    if stream.peek() == "=>":
        expression = BinaryExpression(stream.expect_any(), expression, _parse_implies(stream))
    return expression


def _parse_or(stream: "_ExpressionStream") -> Expression:
    """Parse disjunction expressions. / 解析析取表达式。"""
    expression = _parse_and(stream)
    while stream.peek() in {"|", "or"}:
        operator = stream.expect_any()
        expression = BinaryExpression(operator, expression, _parse_and(stream))
    return expression


def _parse_and(stream: "_ExpressionStream") -> Expression:
    """Parse conjunction expressions. / 解析合取表达式。"""
    expression = _parse_comparison(stream)
    while stream.peek() in {"^", "&", "and"}:
        operator = stream.expect_any()
        expression = BinaryExpression(operator, expression, _parse_comparison(stream))
    return expression


def _parse_comparison(stream: "_ExpressionStream") -> Expression:
    """Parse comparison expressions. / 解析比较表达式。"""
    expression = _parse_additive(stream)
    while stream.peek() in {"==", "!=", "=", "<", "<=", ">", ">="}:
        operator = stream.expect_any()
        expression = BinaryExpression(operator, expression, _parse_additive(stream))
    return expression


def _parse_additive(stream: "_ExpressionStream") -> Expression:
    """Parse addition and subtraction expressions. / 解析加减表达式。"""
    expression = _parse_multiplicative(stream)
    while stream.peek() in {"+", "-"}:
        operator = stream.expect_any()
        expression = BinaryExpression(operator, expression, _parse_multiplicative(stream))
    return expression


def _parse_multiplicative(stream: "_ExpressionStream") -> Expression:
    """Parse multiplication and division expressions. / 解析乘除表达式。"""
    expression = _parse_unary(stream)
    while stream.peek() in {"*", "/"}:
        operator = stream.expect_any()
        expression = BinaryExpression(operator, expression, _parse_unary(stream))
    return expression


def _parse_unary(stream: "_ExpressionStream") -> Expression:
    """Parse unary expressions. / 解析一元表达式。"""
    if stream.peek() in {"!", "~", "not", "+", "-"}:
        return UnaryExpression(stream.expect_any(), _parse_unary(stream))
    return _parse_primary(stream)


def _parse_primary(stream: "_ExpressionStream") -> Expression:
    """Parse literals, grouped expressions, and calls. / 解析 literal、括号表达式和调用。"""
    token = stream.expect_any()
    aggregate_name = token.rstrip("_")
    if aggregate_name in AGGREGATE_NAMES and stream.peek() == "{":
        return _parse_aggregate(stream, aggregate_name)
    if token == "(":
        expression = _parse_if(stream)
        stream.expect(")")
        return expression
    if token in {"true", "false"}:
        return LiteralExpression(token == "true")
    if _is_number(token):
        return LiteralExpression(float(token))
    if stream.consume_if("("):
        args: list[Expression] = []
        if not stream.consume_if(")"):
            while True:
                args.append(_parse_if(stream))
                if stream.consume_if(")"):
                    break
                stream.expect(",")
        return CallExpression(token, tuple(args))
    return LiteralExpression(_object_name(token) if token.startswith("@") else token)


def _parse_aggregate(stream: "_ExpressionStream", kind: str) -> Expression:
    """Parse a finite-object aggregate expression. / 解析有限对象聚合表达式。"""
    stream.expect("{")
    bindings: list[tuple[str, str]] = []
    while True:
        variable = stream.expect_any()
        if not variable.startswith("?"):
            raise RDDLExpressionError(f"Aggregate variable must start with '?', found {variable!r}.")
        stream.expect(":")
        type_name = stream.expect_any()
        bindings.append((variable, type_name))
        if stream.consume_if("}"):
            break
        stream.expect(",")
    if stream.consume_if("["):
        body = _parse_if(stream)
        stream.expect("]")
    elif stream.consume_if("("):
        body = _parse_if(stream)
        stream.expect(")")
    else:
        raise RDDLExpressionError("Aggregate body must be wrapped in [] or ().")
    return AggregateExpression(kind=kind, bindings=tuple(bindings), body=body)


class _ExpressionStream:
    """Provide cursor-style access to expression tokens. / 为表达式 token 提供游标式访问。"""

    def __init__(self, tokens: list[str]) -> None:
        """Create a token stream. / 创建 token 流。"""
        self.tokens = tokens
        self.index = 0

    @property
    def done(self) -> bool:
        """Return whether all tokens were consumed. / 返回 token 是否已全部消耗。"""
        return self.index >= len(self.tokens)

    def peek(self) -> str | None:
        """Return the next token without consuming it. / 返回但不消耗下一个 token。"""
        return None if self.done else self.tokens[self.index]

    def consume_if(self, value: str) -> bool:
        """Consume the next token if it matches. / 如果下一个 token 匹配则消耗它。"""
        if self.peek() == value:
            self.index += 1
            return True
        return False

    def expect(self, value: str) -> str:
        """Consume and require one token. / 消耗并要求一个指定 token。"""
        token = self.expect_any()
        if token != value:
            raise RDDLExpressionError(f"Expected token {value!r}, found {token!r}.")
        return token

    def expect_any(self) -> str:
        """Consume and return one token. / 消耗并返回一个 token。"""
        if self.done:
            raise RDDLExpressionError("Unexpected end of expression.")
        token = self.tokens[self.index]
        self.index += 1
        return token


def _tokenize_expression(text: str) -> list[str]:
    """Tokenize a compact RDDL expression. / 切分紧凑 RDDL 表达式。"""
    tokens = [match.group(0) for match in TOKEN_RE.finditer(text) if not match.group(0).isspace()]
    normalized: list[str] = []
    index = 0
    while index < len(tokens):
        if index + 1 < len(tokens) and tokens[index] == "=" and tokens[index + 1] == "=":
            normalized.append("==")
            index += 2
            continue
        normalized.append(tokens[index])
        index += 1
    return normalized


def _truthy(value: ExpressionValue) -> bool:
    """Convert expression values to boolean. / 将表达式值转换为 boolean。"""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return float(value) != 0.0
    return bool(value)


def _number(value: ExpressionValue) -> float:
    """Convert an expression value to a number. / 将表达式值转换为数字。"""
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except ValueError as exc:
        raise RDDLExpressionError(f"Expected numeric value, found {value!r}.") from exc


def _is_number(token: str) -> bool:
    """Return whether a token is numeric. / 判断 token 是否为数字。"""
    try:
        float(token)
    except ValueError:
        return False
    return True


def _object_name(value: ExpressionValue) -> str:
    """Normalize object literals for compact state names. / 将对象 literal 规范化为紧凑 state 名。"""
    return str(value).strip().lstrip("@")


def _evaluate_builtin(name: str, values: tuple[ExpressionValue, ...]) -> ExpressionValue:
    """Evaluate a supported RDDL built-in call. / 求值一个已支持的 RDDL 内置调用。"""
    if name in {"KronDelta", "DiracDelta"}:
        if len(values) != 1:
            raise RDDLExpressionError(f"{name} expects exactly one argument.")
        return _object_name(values[0])
    if name == "Bernoulli":
        if len(values) != 1:
            raise RDDLExpressionError("Bernoulli expects exactly one probability argument.")
        probability = _number(values[0])
        if probability < 0.0 or probability > 1.0:
            raise RDDLExpressionError(f"Bernoulli probability must be in [0, 1], found {probability}.")
        return probability
    if name == "min":
        return min(_number(value) for value in values)
    if name == "max":
        return max(_number(value) for value in values)
    if name == "abs":
        _expect_arity(name, values, 1)
        return abs(_number(values[0]))
    if name == "ceil":
        _expect_arity(name, values, 1)
        return float(math.ceil(_number(values[0])))
    if name == "floor":
        _expect_arity(name, values, 1)
        return float(math.floor(_number(values[0])))
    if name == "round":
        _expect_arity(name, values, 1)
        return float(round(_number(values[0])))
    raise RDDLExpressionError(f"Unsupported built-in call {name!r}.")


def _expect_arity(name: str, values: tuple[ExpressionValue, ...], arity: int) -> None:
    """Require a fixed number of built-in arguments. / 要求内置函数参数数量固定。"""
    if len(values) != arity:
        raise RDDLExpressionError(f"{name} expects {arity} argument(s), found {len(values)}.")


def _bound_contexts(
    context: EvaluationContext, bindings: tuple[tuple[str, str], ...]
) -> list[EvaluationContext]:
    """Enumerate contexts for aggregate variable bindings. / 枚举聚合变量绑定后的上下文。"""
    contexts = [context]
    for variable, type_name in bindings:
        if type_name not in context.objects:
            raise RDDLExpressionError(f"Aggregate type {type_name!r} is not declared.")
        contexts = [bound.bind(variable, value) for bound in contexts for value in context.objects[type_name]]
    return contexts
