"""Tests for pyRDDLGym RDDL loading."""

import pytest

from darp.loaded import LoadedRDDL, RDDLLoadError
from darp.loader import RDDLLoader

DOMAIN = "examples/rddl/tiny_grid_domain.rddl"
INSTANCE = "examples/rddl/tiny_grid_instance.rddl"


def test_pyrddlgym_loader_returns_model_and_env_when_installed():
    """Check pyRDDLGym loading returns model/env components. / 检查 pyRDDLGym 加载会返回 model/env 组件。"""
    pytest.importorskip("pyRDDLGym")

    loaded = RDDLLoader().load(DOMAIN, INSTANCE)

    assert isinstance(loaded, LoadedRDDL)
    assert loaded.native_ast is not None
    assert loaded.model is not None
    assert loaded.env is not None
    assert loaded.metadata["source"] == "pyRDDLGym"
    assert loaded.component_summary()["native_ast"] == "RDDL"
    assert loaded.component_summary()["model"] == "RDDLLiftedModel"
    assert loaded.component_summary()["env"] == "RDDLEnv"


def test_pyrddlgym_loader_gives_clear_file_errors():
    """Check missing RDDL files fail with contextual errors. / 检查缺失 RDDL 文件会给出上下文错误。"""
    pytest.importorskip("pyRDDLGym")

    with pytest.raises(RDDLLoadError, match="pyRDDLGym failed to load"):
        RDDLLoader().load("missing-domain.rddl", "missing-instance.rddl")


def test_loaded_rddl_summary_exposes_future_search_boundaries():
    """Check loaded RDDL exposes the future search boundary. / 检查 loaded RDDL 暴露未来搜索边界。"""
    pytest.importorskip("pyRDDLGym")
    loaded = RDDLLoader().load(DOMAIN, INSTANCE)
    summary = loaded.to_summary_dict()

    assert summary["source"] == "pyRDDLGym"
    assert "pyRDDLGym grounded model view" in summary["future_interfaces"]
    assert "ILP/HILP search over the grounded model and duration sidecars" in summary["future_interfaces"]
    assert "at" in summary["model"]["state_fluents"]
    assert "move-east" in summary["model"]["action_fluents"]


def test_loaded_rddl_reuses_pyrddlgym_grounder():
    """Check grounding is delegated to pyRDDLGym. / 检查 grounding 委托给 pyRDDLGym。"""
    pytest.importorskip("pyRDDLGym")
    loaded = RDDLLoader().load(DOMAIN, INSTANCE)
    with pytest.warns(UserWarning):
        grounded = loaded.build_grounded_model()

    assert type(grounded).__name__ == "RDDLGroundedModel"
    assert "at___c11" in grounded.state_fluents
    assert "move-east" in grounded.action_fluents
