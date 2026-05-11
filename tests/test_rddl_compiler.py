"""Tests for compiling ParsedRDDL into PlanningProblem."""

import pytest

from darp.core.problem import PlanningProblem
from darp.rddl.compiler import RDDLCompileError, RDDLCompiler, main
from darp.rddl.frontend import ParsedRDDL
from darp.rddl.loader import RDDLLoader

DOMAIN = "examples/rddl/tiny_grid_domain.rddl"
INSTANCE = "examples/rddl/tiny_grid_instance.rddl"
FACTORED_DOMAIN = "examples/rddl/factored_door_domain.rddl"
FACTORED_INSTANCE = "examples/rddl/factored_door_instance.rddl"
GRID_STATES = ("c11", "c12", "c13", "c21", "c22", "c23", "c31", "c32", "c33")
GRID_ACTIONS = ("move-east", "move-south", "move-west", "move-north")


def test_compiler_builds_planning_problem_from_darp_ast():
    """Check RDDL compilation contract. / 检查 RDDL 编译契约。"""
    loaded = RDDLLoader("darp").load(DOMAIN, INSTANCE)
    problem = RDDLCompiler().compile(loaded)

    assert isinstance(problem, PlanningProblem)
    assert problem.name == "tiny_grid_inst"
    assert problem.states == GRID_STATES
    assert problem.actions == GRID_ACTIONS
    assert problem.observations == problem.states
    assert problem.initial_belief == {state: (1.0 if state == "c11" else 0.0) for state in GRID_STATES}
    assert problem.horizon == 8.0
    assert problem.max_depth == 8
    assert problem.discount == 1.0
    assert problem.observation_prob("c11", "c11", "move-east") == 1.0
    assert problem.metadata["compiler_mode"] == "grounded-rddl-expressions"
    assert problem.metadata["requirements"] == ["cpf-deterministic", "reward-deterministic"]


@pytest.mark.parametrize("frontend", ["darp", "pyrddl", "pyrddlgym"])
def test_compiler_accepts_all_available_frontends(frontend):
    """Check that every frontend feeds the same compiler contract. / 检查所有 frontend 都满足同一 compiler 契约。"""
    if frontend == "pyrddl":
        pytest.importorskip("pyrddl")
    if frontend == "pyrddlgym":
        pytest.importorskip("pyRDDLGym")

    loaded = RDDLLoader(frontend).load(DOMAIN, INSTANCE)
    problem = RDDLCompiler().compile(loaded)

    assert problem.states == GRID_STATES
    assert problem.actions == GRID_ACTIONS
    assert problem.metadata["frontend"] == frontend


def test_compiler_builds_factored_boolean_problem():
    """Check Phase 4 factored-state compilation. / 检查 Phase 4 的 factored state 编译。"""
    loaded = RDDLLoader("darp").load(FACTORED_DOMAIN, FACTORED_INSTANCE)
    problem = RDDLCompiler().compile(loaded)

    assert problem.states == ("{}", "{door-open}", "{has-key}", "{has-key,door-open}")
    assert problem.observations == ("{}", "{heard-open}")
    assert problem.actions == ("pick-key", "open-door", "wait")
    assert problem.initial_belief["{}"] == 1.0
    assert problem.max_nondef_actions == 1
    assert problem.action_fluents["open-door"] == frozenset({("open-door", ())})
    assert problem.metadata["compiler_mode"] == "factored-grounded-rddl-expressions"
    assert problem.metadata["state_fluents"] == ["has-key", "door-open"]
    assert problem.metadata["observation_fluents"] == ["heard-open"]
    assert problem.metadata["requirements"] == ["partially-observed", "reward-deterministic"]


def test_compiler_rejects_missing_canonical_ast():
    """Check that compilation requires the canonical DARP AST. / 检查编译必须使用 DARP 标准 AST。"""
    loaded = ParsedRDDL(frontend="bad", domain=DOMAIN, instance=INSTANCE, ast=None)

    with pytest.raises(RDDLCompileError, match="RDDLASTNode"):
        RDDLCompiler().compile(loaded)


def test_compiler_rejects_stochastic_reward_when_reward_deterministic(tmp_path):
    """Check the first requirement baseline. / 检查第一个 requirement 基线。"""
    domain, instance = _write_rddl_pair(
        tmp_path,
        """
        domain bad_reward_randomness {
          requirements { reward-deterministic };
          pvariables {
            ok : { state-fluent, bool, default = false };
            wait : { action-fluent, bool, default = false };
          };
          cpfs { ok' = ok; };
          reward = Bernoulli(0.5);
        }
        """,
        """
        instance bad_reward_randomness_inst {
          domain = bad_reward_randomness;
          init-state { };
          horizon = 1;
          discount = 1;
          max-nondef-actions = 1;
        }
        """,
    )

    with pytest.raises(RDDLCompileError, match="reward-deterministic"):
        RDDLCompiler().compile(RDDLLoader("darp").load(domain, instance))


def test_compiler_accepts_cpf_deterministic_for_deterministic_cpfs(tmp_path):
    """Check deterministic state CPFs can enable the requirement. / 检查确定性 state CPF 可启用该 requirement。"""
    domain, instance = _write_rddl_pair(
        tmp_path,
        """
        domain deterministic_cpfs {
          requirements { reward-deterministic, cpf-deterministic };
          pvariables {
            ok : { state-fluent, bool, default = false };
            wait : { action-fluent, bool, default = false };
          };
          cpfs { ok' = ok; };
          reward = 0;
        }
        """,
        """
        instance deterministic_cpfs_inst {
          domain = deterministic_cpfs;
          init-state { };
          horizon = 1;
          discount = 1;
          max-nondef-actions = 1;
        }
        """,
    )

    problem = RDDLCompiler().compile(RDDLLoader("darp").load(domain, instance))

    assert problem.metadata["requirements"] == ["cpf-deterministic", "reward-deterministic"]


def test_compiler_rejects_stochastic_state_cpf_when_cpf_deterministic(tmp_path):
    """Check cpf-deterministic rejects stochastic transition CPFs. / 检查 cpf-deterministic 会拒绝随机转移 CPF。"""
    domain, instance = _write_rddl_pair(
        tmp_path,
        """
        domain bad_state_cpf_randomness {
          requirements { reward-deterministic, cpf-deterministic };
          pvariables {
            ok : { state-fluent, bool, default = false };
            wait : { action-fluent, bool, default = false };
          };
          cpfs { ok' = Bernoulli(0.5); };
          reward = 0;
        }
        """,
        """
        instance bad_state_cpf_randomness_inst {
          domain = bad_state_cpf_randomness;
          init-state { };
          horizon = 1;
          discount = 1;
          max-nondef-actions = 1;
        }
        """,
    )

    with pytest.raises(RDDLCompileError, match="cpf-deterministic"):
        RDDLCompiler().compile(RDDLLoader("darp").load(domain, instance))


def test_compiler_rejects_observ_fluent_without_partially_observed(tmp_path):
    """Check observ-fluent requires the matching requirement. / 检查 observ-fluent 必须声明对应 requirement。"""
    domain, instance = _write_rddl_pair(
        tmp_path,
        """
        domain missing_partial_requirement {
          requirements { reward-deterministic };
          pvariables {
            ok : { state-fluent, bool, default = false };
            seen-ok : { observ-fluent, bool, default = false };
            wait : { action-fluent, bool, default = false };
          };
          cpfs {
            ok' = ok;
            seen-ok' = ok;
          };
          reward = 0;
        }
        """,
        """
        instance missing_partial_requirement_inst {
          domain = missing_partial_requirement;
          init-state { };
          horizon = 1;
          discount = 1;
          max-nondef-actions = 1;
        }
        """,
    )

    with pytest.raises(RDDLCompileError, match="observ-fluent.*partially-observed"):
        RDDLCompiler().compile(RDDLLoader("darp").load(domain, instance))


def test_compiler_rejects_partially_observed_without_observ_fluent(tmp_path):
    """Check partially-observed must expose an observation model. / 检查 partially-observed 必须暴露观测模型。"""
    domain, instance = _write_rddl_pair(
        tmp_path,
        """
        domain missing_observ_fluent {
          requirements { reward-deterministic, partially-observed };
          pvariables {
            ok : { state-fluent, bool, default = false };
            wait : { action-fluent, bool, default = false };
          };
          cpfs { ok' = ok; };
          reward = 0;
        }
        """,
        """
        instance missing_observ_fluent_inst {
          domain = missing_observ_fluent;
          init-state { };
          horizon = 1;
          discount = 1;
          max-nondef-actions = 1;
        }
        """,
    )

    with pytest.raises(RDDLCompileError, match="at least one observ-fluent"):
        RDDLCompiler().compile(RDDLLoader("darp").load(domain, instance))


def test_compiler_rejects_partially_observed_missing_observation_cpf(tmp_path):
    """Check observation fluents need CPFs. / 检查观测 fluent 必须有 CPF。"""
    domain, instance = _write_rddl_pair(
        tmp_path,
        """
        domain missing_observation_cpf {
          requirements { reward-deterministic, partially-observed };
          pvariables {
            ok : { state-fluent, bool, default = false };
            seen-ok : { observ-fluent, bool, default = false };
            wait : { action-fluent, bool, default = false };
          };
          cpfs { ok' = ok; };
          reward = 0;
        }
        """,
        """
        instance missing_observation_cpf_inst {
          domain = missing_observation_cpf;
          init-state { };
          horizon = 1;
          discount = 1;
          max-nondef-actions = 1;
        }
        """,
    )

    with pytest.raises(RDDLCompileError, match="observ-fluent"):
        RDDLCompiler().compile(RDDLLoader("darp").load(domain, instance))


def test_compiler_still_rejects_other_unimplemented_requirements(tmp_path):
    """Check future requirements are still added one by one. / 检查未来 requirements 仍会逐个加入。"""
    domain, instance = _write_rddl_pair(
        tmp_path,
        """
        domain unsupported_requirement {
          requirements { reward-deterministic, concurrent };
          pvariables {
            ok : { state-fluent, bool, default = false };
            wait : { action-fluent, bool, default = false };
          };
          cpfs { ok' = ok; };
          reward = 0;
        }
        """,
        """
        instance unsupported_requirement_inst {
          domain = unsupported_requirement;
          init-state { };
          horizon = 1;
          discount = 1;
          max-nondef-actions = 1;
        }
        """,
    )

    with pytest.raises(
        RDDLCompileError,
        match="Only 'reward-deterministic', 'cpf-deterministic', and 'partially-observed'",
    ):
        RDDLCompiler().compile(RDDLLoader("darp").load(domain, instance))


def test_planning_problem_validation_rejects_bad_transition_mass():
    """Check core model validation catches invalid tables. / 检查核心模型校验能发现非法表。"""
    with pytest.raises(ValueError, match="Transition mass"):
        PlanningProblem(
            states=("s0", "s1"),
            actions=("a",),
            observations=("s0", "s1"),
            transitions={
                ("s0", "a", "s0"): 0.5,
                ("s0", "a", "s1"): 0.0,
                ("s1", "a", "s0"): 0.0,
                ("s1", "a", "s1"): 1.0,
            },
            observation_model={
                ("s0", "s0", "a"): 1.0,
                ("s1", "s0", "a"): 0.0,
                ("s0", "s1", "a"): 0.0,
                ("s1", "s1", "a"): 1.0,
            },
            rewards={("s0", "a"): 0.0, ("s1", "a"): 0.0},
            initial_belief={"s0": 1.0, "s1": 0.0},
            horizon=1.0,
        )


def test_compiler_cli_prints_problem_summary(capsys):
    """Check the compiler inspection CLI. / 检查 compiler 检查命令。"""
    exit_code = main([DOMAIN, INSTANCE, "--frontend", "darp"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert '"name": "tiny_grid_inst"' in captured.out
    assert '"compiler_mode": "grounded-rddl-expressions"' in captured.out


def _write_rddl_pair(tmp_path, domain_text: str, instance_text: str):
    """Write temporary RDDL files. / 写入临时 RDDL 文件。"""
    domain = tmp_path / "domain.rddl"
    instance = tmp_path / "instance.rddl"
    domain.write_text(domain_text, encoding="utf-8")
    instance.write_text(instance_text, encoding="utf-8")
    return domain, instance
