"""Tests for pyRDDLGym RDDL loading."""

import pytest

from darp.rddl.artifacts import RDDLArtifacts, RDDLLoadError
from darp.rddl.compiler import PyRDDLGymPlanningAdapter, RDDLCompileError, summarize_pyrddlgym_artifacts
from darp.rddl.loader import RDDLLoader

DOMAIN = "examples/rddl/tiny_grid_domain.rddl"
INSTANCE = "examples/rddl/tiny_grid_instance.rddl"


def test_pyrddlgym_loader_returns_model_and_env_when_installed():
    """Check pyRDDLGym loading returns model/env artifacts. / 检查 pyRDDLGym 加载会返回 model/env 产物。"""
    pytest.importorskip("pyRDDLGym")

    loaded = RDDLLoader().load(DOMAIN, INSTANCE)

    assert isinstance(loaded, RDDLArtifacts)
    assert loaded.native_ast is not None
    assert loaded.model is not None
    assert loaded.env is not None
    assert loaded.metadata["source"] == "pyRDDLGym"
    assert loaded.artifact_summary()["native_ast"] == "RDDL"
    assert loaded.artifact_summary()["model"] == "RDDLLiftedModel"
    assert loaded.artifact_summary()["env"] == "RDDLEnv"


def test_pyrddlgym_loader_gives_clear_file_errors():
    """Check missing RDDL files fail with contextual errors. / 检查缺失 RDDL 文件会给出上下文错误。"""
    pytest.importorskip("pyRDDLGym")

    with pytest.raises(RDDLLoadError, match="pyRDDLGym failed to load"):
        RDDLLoader().load("missing-domain.rddl", "missing-instance.rddl")


def test_pyrddlgym_summary_exposes_future_extraction_fields():
    """Check pyRDDLGym artifacts expose the future adapter boundary. / 检查 pyRDDLGym 产物暴露未来适配边界。"""
    pytest.importorskip("pyRDDLGym")
    loaded = RDDLLoader().load(DOMAIN, INSTANCE)
    summary = summarize_pyrddlgym_artifacts(loaded)

    assert summary["source"] == "pyRDDLGym"
    assert summary["planning_problem"] is None
    assert "pyRDDLGym generative runtime interface" in summary["planned_extraction"]
    assert "DurationModel sidecar hook" in summary["planned_extraction"]
    assert "at" in summary["model"]["state_fluents"]
    assert "move-east" in summary["model"]["action_fluents"]


def test_pyrddlgym_planning_adapter_is_explicitly_future_work():
    """Check planning extraction fails as clear future work. / 检查规划模型抽取以清晰 future work 失败。"""
    pytest.importorskip("pyRDDLGym")
    loaded = RDDLLoader().load(DOMAIN, INSTANCE)

    with pytest.raises(RDDLCompileError, match="planned but not implemented"):
        PyRDDLGymPlanningAdapter().compile(loaded)
