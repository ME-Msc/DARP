"""Tests for pyRDDLGym-backed online execution."""

from types import SimpleNamespace

import pytest

from darp.loader import RDDLLoader
from darp.planner import action_label
from darp.runtime import PyRDDLGymRuntime
from darp.session import run_online_session

DOMAIN = "examples/rddl/tiny_grid_domain.rddl"
INSTANCE = "examples/rddl/tiny_grid_instance.rddl"


def test_pyrddlgym_runtime_exposes_noop_and_single_action_candidates():
    """Check runtime action candidates come from pyRDDLGym. / 检查 runtime 动作候选来自 pyRDDLGym。"""
    pytest.importorskip("pyRDDLGym")
    loaded = RDDLLoader().load(DOMAIN, INSTANCE)
    runtime = PyRDDLGymRuntime.from_loaded(loaded)
    runtime.reset(seed=7)

    labels = [action_label(action) for action in runtime.action_candidates()]

    assert labels == ["noop", "move-east", "move-south", "move-west", "move-north"]


def test_pyrddlgym_runtime_builds_exact_mdp_belief():
    """Check MDP observations create an exact singleton belief. / 检查 MDP observation 会创建精确单粒子 belief。"""
    pytest.importorskip("pyRDDLGym")
    loaded = RDDLLoader().load(DOMAIN, INSTANCE)
    runtime = PyRDDLGymRuntime.from_loaded(loaded)
    observation = runtime.reset(seed=7)
    belief = runtime.initial_belief(observation, seed=7)

    assert runtime.is_pomdp is False
    assert belief.is_exact is True
    assert belief.is_pomdp is False
    assert belief.particle_count == 1
    assert belief.support() == {"at___c11": 1.0}


def test_pyrddlgym_online_session_reaches_tiny_grid_goal():
    """Check DARP can run a simple RDDL online loop through pyRDDLGym. / 检查 DARP 能通过 pyRDDLGym 运行简单 RDDL 在线循环。"""
    pytest.importorskip("pyRDDLGym")
    loaded = RDDLLoader().load(DOMAIN, INSTANCE)
    result = run_online_session(loaded, seed=7, lookahead_depth=4)
    payload = result.to_dict()

    assert payload["planner"] == "pyrddlgym-rollout"
    assert payload["is_pomdp"] is False
    assert payload["initial_belief"]["is_exact"] is True
    assert payload["horizon"] == 8
    assert payload["steps"][0]["action"] == "move-east"
    assert payload["steps"][0]["belief"]["support"] == {"at___c11": 1.0}
    assert payload["steps"][1]["action"] == "move-east"
    assert payload["steps"][2]["action"] == "move-south"
    assert payload["steps"][3]["action"] == "move-south"
    assert payload["steps"][3]["next_state"]["at___c33"] is True
    assert payload["total_reward"] == 97.0


def test_pyrddlgym_runtime_updates_particle_belief_for_pomdp():
    """Check POMDP observations can drive particle filtering. / 检查 POMDP observation 能驱动粒子滤波。"""
    runtime = PyRDDLGymRuntime(_FakePOMDPEnv())
    observation = runtime.reset(seed=1)
    belief = runtime.initial_belief(observation, seed=1, particle_count=3, max_attempts=20)

    assert runtime.is_pomdp is True
    assert belief.is_exact is False
    assert belief.particle_count == 3
    assert belief.support() == {"hidden": 1.0}

    action = {"flip": True}
    next_observation, _, _, _, _ = runtime.step(action)
    next_belief = runtime.update_belief(
        belief,
        action,
        next_observation,
        particle_count=3,
        max_attempts=20,
    )

    assert next_observation == {"seen": False}
    assert next_belief.is_pomdp is True
    assert next_belief.particle_count == 3
    assert next_belief.support() == {"{'hidden': False}": 1.0}


class _FakePOMDPEnv:
    """Small pyRDDLGym-shaped POMDP test double. / 小型 pyRDDLGym 形状的 POMDP 测试替身。"""

    horizon = 2
    _noop_actions = {"flip": False}
    _action_ranges = {"flip": "bool"}

    def __init__(self) -> None:
        """Initialize fake model/sampler fields. / 初始化假的 model/sampler 字段。"""
        self.model = SimpleNamespace(
            instance_name="fake_pomdp",
            domain_name="fake",
            discount=1.0,
            observ_fluents={"seen": None},
        )
        self.sampler = SimpleNamespace(is_pomdp=True)
        self.state = {"hidden": False}

    def reset(self, seed=None):
        """Reset hidden state from seed parity. / 根据 seed 奇偶性重置 hidden state。"""
        self.state = {"hidden": bool((seed or 0) % 2)}
        return {"seen": self.state["hidden"]}, {}

    def step(self, action):
        """Flip hidden state when requested. / 按需翻转 hidden state。"""
        if action.get("flip"):
            self.state = {"hidden": not self.state["hidden"]}
        return {"seen": self.state["hidden"]}, 1.0, False, False, {}
