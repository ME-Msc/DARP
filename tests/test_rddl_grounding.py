"""Tests for grounding RDDL CPF and reward expressions."""

from pathlib import Path

from darp.rddl.compiler import RDDLCompiler
from darp.rddl.loader import RDDLLoader

DOMAIN = "examples/rddl/tiny_grid_domain.rddl"
INSTANCE = "examples/rddl/tiny_grid_instance.rddl"
FACTORED_DOMAIN = "examples/rddl/factored_door_domain.rddl"
FACTORED_INSTANCE = "examples/rddl/factored_door_instance.rddl"


def test_tiny_grid_transitions_are_grounded_from_cpfs():
    """Check that tiny-grid transitions come from CPF evaluation. / 检查 tiny-grid 转移来自 CPF 求值。"""
    loaded = RDDLLoader("darp").load(DOMAIN, INSTANCE)
    problem = RDDLCompiler().compile(loaded)

    assert problem.transition_prob("c11", "move-north", "c11") == 1.0
    assert problem.transition_prob("c11", "move-east", "c12") == 1.0
    assert problem.transition_prob("c12", "move-south", "c22") == 1.0
    assert problem.transition_prob("c23", "move-south", "c33") == 1.0
    assert problem.transition_prob("c32", "move-east", "c33") == 1.0


def test_tiny_grid_rewards_are_grounded_from_reward_expression():
    """Check that tiny-grid rewards come from the RDDL reward expression. / 检查 tiny-grid reward 来自 RDDL reward 表达式。"""
    loaded = RDDLLoader("darp").load(DOMAIN, INSTANCE)
    problem = RDDLCompiler().compile(loaded)

    assert problem.reward("c11", "move-east") == -1.0
    assert problem.reward("c12", "move-south") == -10.0
    assert problem.reward("c21", "move-east") == -10.0
    assert problem.reward("c23", "move-south") == 20.0
    assert problem.reward("c32", "move-east") == 20.0


def test_grounding_supports_numeric_nonfluents_arithmetic_and_aggregates(tmp_path):
    """Check non-tiny-grid numeric expression grounding. / 检查非 tiny-grid 的数值表达式 grounding。"""
    domain, instance = _write_rddl_pair(
        tmp_path,
        """
        domain numeric_grid {
          types { location : { @a, @b }; };
          pvariables {
            at(location) : { state-fluent, bool, default = false };
            move : { action-fluent, bool, default = false };
            wait : { action-fluent, bool, default = false };
            score(location) : { non-fluent, real, default = 0 };
          };
          cpfs {
            at'(?l) = if (move) then (score(?l) >= 2) else at(?l);
          };
          reward =
            if (move) then
              sum_{?l : location} [if (score(?l) > 1) then score(?l) else 0] * 2 + 1
            else -1;
        }
        """,
        """
        non-fluents numeric_nf {
          domain = numeric_grid;
          non-fluents {
            score(@a) = 1;
            score(@b) = 3;
          };
        }
        instance numeric_inst {
          domain = numeric_grid;
          non-fluents = numeric_nf;
          init-state { at(@a); };
          max-nondef-actions = 1;
          horizon = 1;
          discount = 1;
        }
        """,
    )

    problem = RDDLCompiler().compile(RDDLLoader("darp").load(domain, instance))

    assert problem.states == ("a", "b")
    assert problem.transition_prob("a", "move", "b") == 1.0
    assert problem.transition_prob("b", "wait", "b") == 1.0
    assert problem.reward("a", "move") == 7.0
    assert problem.reward("a", "wait") == -1.0


def test_grounding_supports_parameterized_actions_and_object_cpfs(tmp_path):
    """Check parameterized action calls and object-valued CPF results. / 检查参数化 action 调用和对象值 CPF。"""
    domain, instance = _write_rddl_pair(
        tmp_path,
        """
        domain parameterized_grid {
          types { location : { @a, @b }; };
          pvariables {
            at(location) : { state-fluent, bool, default = false };
            choose(location) : { action-fluent, bool, default = false };
          };
          cpfs {
            at'(?l) = KronDelta(if (choose(@b)) then @b else @a);
          };
          reward = if (choose(@b)) then 5 else 0;
        }
        """,
        """
        instance parameterized_inst {
          domain = parameterized_grid;
          init-state { at(@a); };
          max-nondef-actions = 1;
          horizon = 1;
          discount = 1;
        }
        """,
    )

    problem = RDDLCompiler().compile(RDDLLoader("darp").load(domain, instance))

    assert problem.actions == ("choose(a)", "choose(b)")
    assert problem.transition_prob("a", "choose(b)", "b") == 1.0
    assert problem.transition_prob("b", "choose(a)", "a") == 1.0
    assert problem.reward("a", "choose(b)") == 5.0
    assert problem.reward("a", "choose(a)") == 0.0


def test_factored_grounding_supports_stochastic_cpfs_and_observations():
    """Check factored stochastic transition and noisy observation grounding. / 检查 factored 随机转移和噪声观测 grounding。"""
    loaded = RDDLLoader("darp").load(FACTORED_DOMAIN, FACTORED_INSTANCE)
    problem = RDDLCompiler().compile(loaded)

    assert problem.transition_prob("{}", "pick-key", "{has-key}") == 1.0
    assert problem.transition_prob("{has-key}", "open-door", "{has-key,door-open}") == 0.8
    assert round(problem.transition_prob("{has-key}", "open-door", "{has-key}"), 6) == 0.2
    assert problem.transition_prob("{}", "open-door", "{}") == 1.0
    assert problem.observation_prob("{heard-open}", "{door-open}", "wait") == 0.9
    assert problem.observation_prob("{heard-open}", "{}", "wait") == 0.2
    assert problem.initial_observation_prob("{heard-open}", "{}") == 0.2
    assert problem.reward("{door-open}", "wait") == 10.0
    assert problem.reward("{}", "pick-key") == -1.0


def _write_rddl_pair(tmp_path: Path, domain_text: str, instance_text: str) -> tuple[Path, Path]:
    """Write temporary domain/instance files. / 写入临时 domain 和 instance 文件。"""
    domain = tmp_path / "domain.rddl"
    instance = tmp_path / "instance.rddl"
    domain.write_text(domain_text, encoding="utf-8")
    instance.write_text(instance_text, encoding="utf-8")
    return domain, instance
