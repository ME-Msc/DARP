"""Tests for compiling ParsedRDDL into PlanningProblem."""

# TODO(phase-2.4): Add semantic CPF/reward grounding tests once the compiler supports them.

import pytest

from darp.core.problem import PlanningProblem
from darp.rddl.compiler import RDDLCompileError, RDDLCompiler, main
from darp.rddl.frontend import ParsedRDDL
from darp.rddl.loader import RDDLLoader

DOMAIN = "examples/rddl/tiny_grid_domain.rddl"
INSTANCE = "examples/rddl/tiny_grid_instance.rddl"


def test_compiler_builds_planning_problem_from_darp_ast():
    """Check structural RDDL compilation. / 检查结构化 RDDL 编译。"""
    loaded = RDDLLoader("darp").load(DOMAIN, INSTANCE)
    problem = RDDLCompiler().compile(loaded)

    assert isinstance(problem, PlanningProblem)
    assert problem.name == "tiny_grid_inst"
    assert problem.states == ("start", "safe", "risk", "goal")
    assert problem.actions == ("move-safe", "move-risky")
    assert problem.observations == problem.states
    assert problem.initial_belief == {"start": 1.0, "safe": 0.0, "risk": 0.0, "goal": 0.0}
    assert problem.horizon == 2.0
    assert problem.discount == 1.0
    assert problem.transition_prob("start", "move-safe", "start") == 1.0
    assert problem.transition_prob("start", "move-safe", "safe") == 0.0
    assert problem.observation_prob("start", "start", "move-safe") == 1.0
    assert problem.reward("risk", "move-risky") == 0.0
    assert problem.metadata["compiler_mode"] == "structural-identity"


@pytest.mark.parametrize("frontend", ["darp", "pyrddl", "pyrddlgym"])
def test_compiler_accepts_all_available_frontends(frontend):
    """Check that every frontend feeds the same compiler contract. / 检查所有 frontend 都满足同一 compiler 契约。"""
    if frontend == "pyrddl":
        pytest.importorskip("pyrddl")
    if frontend == "pyrddlgym":
        pytest.importorskip("pyRDDLGym")

    loaded = RDDLLoader(frontend).load(DOMAIN, INSTANCE)
    problem = RDDLCompiler().compile(loaded)

    assert problem.states == ("start", "safe", "risk", "goal")
    assert problem.actions == ("move-safe", "move-risky")
    assert problem.metadata["frontend"] == frontend


def test_compiler_rejects_missing_canonical_ast():
    """Check that compilation requires the canonical DARP AST. / 检查编译必须使用 DARP 标准 AST。"""
    loaded = ParsedRDDL(frontend="bad", domain=DOMAIN, instance=INSTANCE, ast=None)

    with pytest.raises(RDDLCompileError, match="RDDLASTNode"):
        RDDLCompiler().compile(loaded)


def test_compiler_cli_prints_problem_summary(capsys):
    """Check the compiler inspection CLI. / 检查 compiler 检查命令。"""
    exit_code = main([DOMAIN, INSTANCE, "--frontend", "darp"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert '"name": "tiny_grid_inst"' in captured.out
    assert '"compiler_mode": "structural-identity"' in captured.out
