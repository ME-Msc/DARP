"""Tests for the RDDL expression parser and evaluator."""

from darp.rddl.expressions import EvaluationContext, parse_expression


def test_expression_evaluates_square_bracket_aggregate_with_decimal():
    """Check RDDL-style aggregate grouping and leading-decimal literals. / 检查 RDDL 风格聚合分组和前导小数。"""
    expression = parse_expression(
        "[sum_{?x : xpos} NEIGHBOR(?x) ^ alive(?x)] + .45"
    )
    context = _context(
        objects={"xpos": ("x1", "x2")},
        non_fluents=frozenset({("NEIGHBOR", ("x1",)), ("NEIGHBOR", ("x2",))}),
        state_atoms=frozenset({("alive", ("x1",))}),
        state_pvariables=frozenset({"alive"}),
    )

    assert expression.evaluate(context) == 1.45


def test_expression_evaluates_unwrapped_exists_and_equivalence():
    """Check unwrapped quantifier bodies and boolean equivalence. / 检查未包裹量词 body 和布尔等价。"""
    expression = parse_expression(
        "(exists_{?x : xpos} alive(?x)) <=> (set(x1) | set(x2))"
    )
    context = _context(
        objects={"xpos": ("x1", "x2")},
        state_atoms=frozenset({("alive", ("x1",))}),
        state_pvariables=frozenset({"alive"}),
        action_fluents=frozenset({("set", ("x2",))}),
        action_pvariables=frozenset({"set"}),
    )

    assert expression.evaluate(context) is True


def test_expression_evaluates_forall_implication():
    """Check forall aggregates over implication bodies. / 检查 forall 聚合中的蕴含表达式。"""
    expression = parse_expression(
        "forall_{?x : xpos} [TARGET(?x) => alive(?x)]"
    )
    context = _context(
        objects={"xpos": ("x1", "x2")},
        non_fluents=frozenset({("TARGET", ("x1",))}),
        state_atoms=frozenset({("alive", ("x1",))}),
        state_pvariables=frozenset({"alive"}),
    )

    assert expression.evaluate(context) is True


def _context(**overrides):
    """Build a minimal expression context. / 构建最小表达式上下文。"""
    defaults = {
        "state_fluent": "",
        "current_state": "",
        "action": "",
        "actions": (),
        "non_fluents": frozenset(),
        "variables": {},
        "fluent_values": {},
        "objects": {},
        "state_atoms": frozenset(),
        "state_pvariables": frozenset(),
        "action_fluents": frozenset(),
        "action_pvariables": frozenset(),
    }
    defaults.update(overrides)
    return EvaluationContext(**defaults)
