"""Tests for pyRDDLGym-backed online execution."""

from types import SimpleNamespace

import pytest

from darp.adapter.loader import RDDLLoader
from darp.adapter.runtime import PyRDDLGymRuntime
from darp.model.and_or_tree import ANDORSearchInterface, ActionChoice, ObservationScope
from darp.planning.rollout import action_label
from darp.planning.session import run_online_session

DOMAIN = "examples/rddl/tiny_grid_domain.rddl"
INSTANCE = "examples/rddl/tiny_grid_instance.rddl"


def test_pyrddlgym_runtime_exposes_noop_and_single_action_candidates():
    """Check runtime action candidates come from pyRDDLGym. / 检查 runtime 动作候选来自 pyRDDLGym。"""
    pytest.importorskip("pyRDDLGym")
    problem = RDDLLoader().load(DOMAIN, INSTANCE)
    runtime = PyRDDLGymRuntime.from_problem(problem)
    runtime.reset(seed=7)

    labels = [action_label(action) for action in runtime.action_candidates()]

    assert labels == ["noop", "move-east", "move-south", "move-west", "move-north"]


def test_pyrddlgym_runtime_builds_exact_mdp_belief():
    """Check MDP observations create an exact singleton belief. / 检查 MDP observation 会创建精确单粒子 belief。"""
    pytest.importorskip("pyRDDLGym")
    problem = RDDLLoader().load(DOMAIN, INSTANCE)
    runtime = PyRDDLGymRuntime.from_problem(problem)
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
    problem = RDDLLoader().load(DOMAIN, INSTANCE)
    result = run_online_session(problem, seed=7, lookahead_depth=4)
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


def test_online_session_can_use_full_ilp_planner_path():
    """Check online session can call the full-ILP planner path. / 检查 online session 能调用 full-ILP planner 路径。"""
    problem = _FakePlannerProblem()

    result = run_online_session(problem, planner_name="full-ilp", lookahead_depth=1)
    payload = result.to_dict()

    assert payload["planner"] == "full-ilp-gurobi"
    assert payload["duration"]["defaulted"] is True
    assert payload["steps"][0]["action"] == "go"
    assert payload["steps"][0]["next_state"]["at_goal"] is True


def test_online_session_can_use_hilp_planner_path():
    """Check online session can call the HILP planner path. / 检查 online session 能调用 HILP planner 路径。"""
    problem = _FakePlannerProblem()

    result = run_online_session(problem, planner_name="hilp", lookahead_depth=1, hilp_iterations=1)
    payload = result.to_dict()

    assert payload["planner"] == "hilp-partial-tree"
    assert payload["steps"][0]["action"] == "go"
    assert payload["steps"][0]["next_state"]["at_goal"] is True


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


class _FakePlannerProblem:
    """Small problem double for planner-path session tests. / planner 路径 session 测试用 problem 替身。"""

    def __init__(self) -> None:
        """Initialize the fake pyRDDLGym-style env. / 初始化 fake pyRDDLGym 风格 env。"""
        self.env = _FakePlannerEnv()

    def build_grounded_view(self):
        """Return a fake grounded view exposing an AND-OR interface. / 返回暴露 AND-OR interface 的 fake grounded view。"""
        return _FakeGroundedView()


class _FakePlannerEnv:
    """Small deterministic env where `go` is the best action. / `go` 是最佳动作的小型确定性 env。"""

    horizon = 1
    _noop_actions = {"go": False}
    _action_ranges = {"go": "bool"}

    def __init__(self) -> None:
        """Initialize fake model fields and state. / 初始化 fake model 字段和 state。"""
        self.model = SimpleNamespace(
            instance_name="fake_planner",
            domain_name="fake",
            discount=1.0,
            observ_fluents={},
        )
        self.sampler = SimpleNamespace(is_pomdp=False)
        self.state = {"at_goal": False}

    def reset(self, seed=None):
        """Reset to the non-goal state. / 重置到非目标状态。"""
        self.state = {"at_goal": False}
        return dict(self.state), {}

    def step(self, action):
        """Apply `go` and return its reward. / 执行 `go` 并返回 reward。"""
        reward = 5.0 if action.get("go") else 0.0
        self.state = {"at_goal": bool(action.get("go"))}
        return dict(self.state), reward, False, False, {}


class _FakeGroundedView:
    """Small grounded view double for AND-OR planner inputs. / AND-OR planner 输入用 grounded view 替身。"""

    def build_and_or_interface(self, runtime):
        """Build action and observation inputs from the runtime. / 从 runtime 构建 action 和 observation 输入。"""
        return ANDORSearchInterface.from_actions_and_observations(
            actions=tuple(
                ActionChoice(label=action_label(action), assignment=action)
                for action in runtime.action_candidates()
            ),
            observation_scope=ObservationScope(mode="mdp-state", variables=("at_goal",)),
        )
