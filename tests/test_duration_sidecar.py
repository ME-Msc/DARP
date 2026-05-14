"""Tests for DARP duration sidecars and history-duration evaluation."""

import json
import sys

import pytest

from darp.adapter.loader import RDDLLoader
from darp.model.and_or_tree import History
from darp.model.duration import (
    FixedDurationModel,
    GaussianDurationModel,
    StateDependentDurationModel,
)
from darp.model.duration_sidecar import (
    DurationSpecError,
    build_duration_sidecar,
    duration_action_names,
    load_duration_sidecar,
)

DOMAIN = "examples/rddl/tiny_grid_domain.rddl"
INSTANCE = "examples/rddl/tiny_grid_instance.rddl"
DURATIONS = "examples/durations/tiny_grid.yaml"


def test_yaml_fixed_duration_sidecar_matches_grounded_actions():
    """Check fixed YAML sidecars load and validate against grounded actions. / 检查 fixed YAML sidecar 能加载并匹配 grounded action。"""
    pytest.importorskip("pyRDDLGym")
    problem = RDDLLoader().load(DOMAIN, INSTANCE)
    sidecar = load_duration_sidecar(DURATIONS)
    with pytest.warns(UserWarning):
        action_names = problem.build_grounded_view().action_fluents()

    sidecar.validate_actions(action_names)

    assert isinstance(sidecar.model, FixedDurationModel)
    assert duration_action_names(sidecar.model) == (
        "move-east",
        "move-north",
        "move-south",
        "move-west",
    )
    assert sidecar.model.estimate({}, "move-east").mean == 1.0


def test_history_duration_evaluator_matches_phase7_duration_model_shape():
    """Check evaluator exposes duration_model(q)-style history duration. / 检查 evaluator 暴露类似 duration_model(q) 的 history 时长。"""
    sidecar = load_duration_sidecar(DURATIONS)
    evaluator = sidecar.evaluator()
    history = History().append_action("move-east").append_observation("at___c12").append_action("move-south")

    progress = evaluator.progress_for_history(history)

    assert progress.mean == 2.0
    assert progress.variance == 0.0
    assert evaluator.elapsed_for_history(history) == 2.0
    assert evaluator.should_expand(history) is True
    assert sidecar.evaluator(horizon=2).should_expand(history) is False


def test_expected_duration_sidecar_uses_belief_weighting(tmp_path):
    """Check expected durations are weighted by belief. / 检查 expected duration 按 belief 加权。"""
    path = tmp_path / "expected.json"
    path.write_text(
        json.dumps(
            {
                "duration_model": {
                    "kind": "expected",
                    "default": 1,
                    "state_actions": {
                        "s0": {"act": 2},
                        "s1": {"act": 4},
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    sidecar = load_duration_sidecar(path)

    assert isinstance(sidecar.model, StateDependentDurationModel)
    estimate = sidecar.model.estimate({"s0": 0.25, "s1": 0.75}, "act")
    assert estimate.mean == 3.5


def test_gaussian_duration_sidecar_builds_tau_model(tmp_path):
    """Check Gaussian sidecars build tau-capable models. / 检查 Gaussian sidecar 能构建支持 tau 的模型。"""
    sidecar = build_duration_sidecar(
        {
            "duration_model": {
                "kind": "gaussian",
                "horizon": 5,
                "default_mean": 1,
                "default_variance": 0,
                "state_actions": {
                    "s0": {"act": {"mean": 2, "variance": 0.25}},
                    "s1": {"act": {"mean": 4, "variance": 0.25}},
                },
            }
        }
    )
    evaluator = sidecar.evaluator()
    history = History().append_action("act")

    assert isinstance(sidecar.model, GaussianDurationModel)
    progress = evaluator.progress_for_history(history, beliefs=({"s0": 0.5, "s1": 0.5},))
    assert progress.mean == 3.0
    assert 0.0 < evaluator.tau_for_history(history, beliefs=({"s0": 0.5, "s1": 0.5},)) < 1.0


def test_duration_sidecar_rejects_unknown_actions():
    """Check action validation catches names absent from RDDL. / 检查 action validation 能发现 RDDL 中不存在的名称。"""
    sidecar = build_duration_sidecar(
        {
            "duration_model": {
                "kind": "fixed",
                "actions": {"unknown-action": 1},
            }
        }
    )

    with pytest.raises(DurationSpecError, match="unknown-action"):
        sidecar.validate_actions({"known-action"})


def test_plugin_duration_sidecar_reserves_python_extension_point(tmp_path, monkeypatch):
    """Check plugin sidecars can construct custom DurationModel objects. / 检查 plugin sidecar 能构造自定义 DurationModel。"""
    plugin = tmp_path / "duration_plugin.py"
    plugin.write_text(
        "\n".join(
            [
                "from darp.model.duration import FixedDurationModel",
                "",
                "def build_duration_model(config):",
                "    return FixedDurationModel(durations={'act': config['duration']}, default=1)",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop("duration_plugin", None)

    sidecar = build_duration_sidecar(
        {
            "duration_model": {
                "kind": "plugin",
                "module": "duration_plugin",
                "factory": "build_duration_model",
                "config": {"duration": 3},
            }
        }
    )

    assert isinstance(sidecar.model, FixedDurationModel)
    assert sidecar.model.estimate({}, "act").mean == 3
