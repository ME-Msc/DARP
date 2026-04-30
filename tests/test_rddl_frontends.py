"""Tests for RDDLFrontend alignment across parser backends."""

import pytest

from darp.rddl.frontend import ParsedRDDL, RDDLFrontendError
from darp.rddl.loader import RDDLLoader, available_frontends

DOMAIN = "examples/rddl/tiny_grid_domain.rddl"
INSTANCE = "examples/rddl/tiny_grid_instance.rddl"


def test_loader_exposes_canonical_frontends():
    """Check the canonical frontend choices. / 检查标准 frontend 选项。"""
    assert available_frontends() == ("darp", "pyrddl", "pyrddlgym")


def test_darp_frontend_returns_parsed_container():
    """Check that the DARP frontend returns shared ParsedRDDL. / 检查 DARP frontend 返回统一 ParsedRDDL。"""
    loaded = RDDLLoader("darp").load(DOMAIN, INSTANCE)

    assert isinstance(loaded, ParsedRDDL)
    assert loaded.frontend == "darp"
    assert loaded.ast is not None
    assert loaded.native_ast is None
    assert loaded.model is None
    assert loaded.env is None
    assert loaded.metadata["source"] == "darp-basic-parser"
    assert loaded.artifact_summary()["ast"] == "RDDLASTNode"
    assert loaded.artifact_summary()["native_ast"] is None


def test_unknown_frontend_gives_clear_error():
    """Check that unknown frontends raise a clear error. / 检查未知 frontend 会给出清晰错误。"""
    with pytest.raises(RDDLFrontendError, match="Unknown RDDL frontend"):
        RDDLLoader("missing")


def test_pyrddl_frontend_returns_parsed_container_when_installed():
    """Check pyrddl integration when the optional package is installed. / 检查可选 pyrddl 安装后的集成。"""
    pytest.importorskip("pyrddl")

    loaded = RDDLLoader("pyrddl").load(DOMAIN, INSTANCE)

    assert isinstance(loaded, ParsedRDDL)
    assert loaded.frontend == "pyrddl"
    assert loaded.ast is not None
    assert loaded.native_ast is not None
    assert loaded.metadata["source"] == "pyrddl"
    assert loaded.artifact_summary()["ast"] == "RDDLASTNode"
    assert loaded.artifact_summary()["native_ast"] == "RDDL"


def test_pyrddlgym_frontend_returns_model_and_env_when_installed():
    """Check pyRDDLGym integration when the optional package is installed. / 检查可选 pyRDDLGym 安装后的集成。"""
    pytest.importorskip("pyRDDLGym")

    loaded = RDDLLoader("pyrddlgym").load(DOMAIN, INSTANCE)

    assert isinstance(loaded, ParsedRDDL)
    assert loaded.frontend == "pyrddlgym"
    assert loaded.ast is not None
    assert loaded.model is not None
    assert loaded.env is not None
    assert loaded.metadata["source"] == "pyRDDLGym"
    assert loaded.artifact_summary()["ast"] == "RDDLASTNode"
    assert loaded.artifact_summary()["model"] == "RDDLLiftedModel"
    assert loaded.artifact_summary()["env"] == "RDDLEnv"
