"""Tests for pyRDDLGym RDDL loading."""

from types import SimpleNamespace

import pytest

from darp.adapter.grounded import GroundedRDDLView, UnsupportedRDDLFeatureError
from darp.adapter.loader import RDDLLoader
from darp.adapter.problem import PyRDDLGymProblem, RDDLLoadError
from darp.adapter.runtime import PyRDDLGymRuntime
from darp.model.and_or_tree import ANDORNodeKind

DOMAIN = "examples/rddl/tiny_grid_domain.rddl"
INSTANCE = "examples/rddl/tiny_grid_instance.rddl"


def test_pyrddlgym_loader_returns_model_and_env_when_installed():
    """Check pyRDDLGym loading returns model/env components. / 检查 pyRDDLGym 加载会返回 model/env 组件。"""
    pytest.importorskip("pyRDDLGym")

    problem = RDDLLoader().load(DOMAIN, INSTANCE)

    assert isinstance(problem, PyRDDLGymProblem)
    assert problem.native_ast is not None
    assert problem.model is not None
    assert problem.env is not None
    assert problem.metadata["source"] == "pyRDDLGym"
    assert problem.component_summary()["native_ast"] == "RDDL"
    assert problem.component_summary()["model"] == "RDDLLiftedModel"
    assert problem.component_summary()["env"] == "RDDLEnv"


def test_pyrddlgym_loader_gives_clear_file_errors():
    """Check missing RDDL files fail with contextual errors. / 检查缺失 RDDL 文件会给出上下文错误。"""
    pytest.importorskip("pyRDDLGym")

    with pytest.raises(RDDLLoadError, match="pyRDDLGym failed to load"):
        RDDLLoader().load("missing-domain.rddl", "missing-instance.rddl")


def test_pyrddlgym_problem_summary_exposes_planner_interfaces():
    """Check PyRDDLGymProblem exposes planner interface boundaries. / 检查 PyRDDLGymProblem 暴露 planner interface 边界。"""
    pytest.importorskip("pyRDDLGym")
    problem = RDDLLoader().load(DOMAIN, INSTANCE)
    summary = problem.to_summary_dict()

    assert summary["source"] == "pyRDDLGym"
    assert "pyRDDLGym grounded model view" in summary["planner_interfaces"]
    assert "Phase 7 full-tree/HILP search over the grounded model and duration sidecars" in summary["planner_interfaces"]
    assert "Phase 8 Gurobi full-ILP/p-ILP solver" in summary["planner_interfaces"]
    assert "at" in summary["model"]["state_fluents"]
    assert "move-east" in summary["model"]["action_fluents"]


def test_pyrddlgym_problem_reuses_pyrddlgym_grounder():
    """Check grounding is delegated to pyRDDLGym. / 检查 grounding 委托给 pyRDDLGym。"""
    pytest.importorskip("pyRDDLGym")
    problem = RDDLLoader().load(DOMAIN, INSTANCE)
    with pytest.warns(UserWarning):
        grounded = problem.build_grounded_model()

    assert type(grounded).__name__ == "RDDLGroundedModel"
    assert "at___c11" in grounded.state_fluents
    assert "move-east" in grounded.action_fluents


def test_pyrddlgym_problem_grounded_view_exposes_solver_boundary():
    """Check DARP wraps pyRDDLGym grounding behind a stable view. / 检查 DARP 用稳定 view 封装 pyRDDLGym grounding。"""
    pytest.importorskip("pyRDDLGym")
    problem = RDDLLoader().load(DOMAIN, INSTANCE)
    with pytest.warns(UserWarning):
        view = problem.build_grounded_view()

    assert view.horizon == 8
    assert "at___c11" in view.state_fluents()
    assert "move-east" in view.action_fluents()
    assert "at___c11'" in view.cpfs()
    assert view.reward_expression() is not None


def test_grounded_view_builds_and_or_search_interface():
    """Check grounded model and runtime produce AND-OR search inputs. / 检查 grounded model 和 runtime 能生成 AND-OR 搜索输入。"""
    pytest.importorskip("pyRDDLGym")
    problem = RDDLLoader().load(DOMAIN, INSTANCE)
    runtime = PyRDDLGymRuntime.from_problem(problem)
    runtime.reset(seed=7)
    with pytest.warns(UserWarning):
        interface = problem.build_grounded_view().build_and_or_interface(runtime)

    assert interface.root.kind == ANDORNodeKind.OR
    assert [action.label for action in interface.actions] == [
        "noop",
        "move-east",
        "move-south",
        "move-west",
        "move-north",
    ]
    assert interface.observation_scope.mode == "mdp-state"
    assert "at___c11" in interface.observation_scope.variables

    action_nodes = interface.action_nodes()
    assert action_nodes[1].kind == ANDORNodeKind.AND
    assert action_nodes[1].history.actions == ("move-east",)
    observation_node = interface.observation_node(action_nodes[1], "at___c12")
    assert observation_node.kind == ANDORNodeKind.OR
    assert observation_node.history.label() == "a0=move-east / o1=at___c12"


def test_grounded_view_reports_unsupported_action_ranges():
    """Check unsupported grounded actions fail clearly. / 检查不支持的 grounded action 会清晰失败。"""
    view = GroundedRDDLView(
        _fake_grounded_model(
            action_fluents={"set-level": 0},
            action_ranges={"set-level": "int"},
        )
    )

    with pytest.raises(UnsupportedRDDLFeatureError, match="non-bool action fluents"):
        view.validate_supported()


def test_grounded_view_reports_unsupported_concurrent_actions():
    """Check unsupported concurrent action sets fail clearly. / 检查暂不支持的并行动作集合会清晰失败。"""
    view = GroundedRDDLView(_fake_grounded_model(max_allowed_actions=2))

    with pytest.raises(UnsupportedRDDLFeatureError, match="concurrent action combinations"):
        view.validate_supported()


def _fake_grounded_model(**overrides):
    """Build a tiny pyRDDLGym-shaped grounded model double. / 构造一个小型 pyRDDLGym grounded model 替身。"""
    values = {
        "domain_name": "fake",
        "instance_name": "fake_inst",
        "horizon": 1,
        "discount": 1.0,
        "state_fluents": {"s": False},
        "state_ranges": {"s": "bool"},
        "action_fluents": {"act": False},
        "action_ranges": {"act": "bool"},
        "observ_fluents": {},
        "observ_ranges": {},
        "non_fluents": {},
        "cpfs": {"s'": ([], object())},
        "reward": object(),
        "variable_ranges": {"s": "bool", "act": "bool"},
        "max_allowed_actions": 1,
    }
    values.update(overrides)
    return SimpleNamespace(**values)
