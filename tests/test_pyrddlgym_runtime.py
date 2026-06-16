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


def test_online_session_can_use_full_ilp_planner_path(monkeypatch):
    """Check online session can call the full-ILP planner path. / 检查 online session 能调用 full-ILP planner 路径。"""
    from test_gurobi_ilp import _install_fake_gurobi

    _install_fake_gurobi(monkeypatch)
    problem = _FakePlannerProblem()

    result = run_online_session(problem, planner_name="full-ilp")
    payload = result.to_dict()

    assert payload["planner"] == "full-ilp-gurobi"
    assert payload["duration"]["defaulted"] is True
    assert payload["initial_belief"]["source"] == "exact-initial-state"
    assert payload["steps"][0]["belief"]["source"] == "exact-initial-state"
    assert payload["steps"][0]["next_belief"]["source"] == "exact-bayes"
    assert payload["steps"][0]["action"] == "go"
    assert payload["steps"][0]["next_state"]["at_goal"] is True


def test_online_session_can_use_hilp_planner_path(monkeypatch):
    """Check online session can call the HILP planner path. / 检查 online session 能调用 HILP planner 路径。"""
    from test_gurobi_ilp import _install_fake_gurobi

    _install_fake_gurobi(monkeypatch)
    problem = _FakePlannerProblem()

    result = run_online_session(problem, planner_name="hilp", lookahead_depth=1, hilp_iterations=1)
    payload = result.to_dict()

    assert payload["planner"] == "hilp-partial-tree"
    assert payload["initial_belief"]["source"] == "exact-initial-state"
    assert payload["steps"][0]["belief"]["source"] == "exact-initial-state"
    assert payload["steps"][0]["next_belief"]["source"] == "exact-bayes"
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

    def build_and_or_interface(self, runtime, risk=None):
        """Build action and observation inputs from the runtime. / 从 runtime 构建 action 和 observation 输入。"""
        return ANDORSearchInterface.from_actions_and_observations(
            actions=tuple(
                ActionChoice(label=action_label(action), assignment=action)
                for action in runtime.action_candidates()
            ),
            observation_scope=ObservationScope(mode="mdp-state", variables=("at_goal",)),
            exact_kernel=_FakeExactKernel(),
        )


class _FakeExactKernel:
    """Exact finite kernel for fake planner-path session tests. / fake planner session 测试用 exact finite kernel。"""

    def initial_belief_from_state(self, state):
        """Return a singleton belief. / 返回单点 belief。"""
        return {self._state_key(bool(state.get("at_goal", False))): 1.0}

    def fluent_belief(self, belief):
        """Return state fluent marginals. / 返回 state fluent 边缘概率。"""
        return {"at_goal": sum(prob for state, prob in belief.items() if dict(state).get("at_goal"))}

    def state_label(self, state):
        """Return a compact state label. / 返回紧凑 state 标签。"""
        return "at_goal" if dict(state).get("at_goal") else "not_at_goal"

    def expand_action(self, belief, action):
        """Return exact transition, observation, reward, and risk constants. / 返回 exact 转移、观测、奖励和风险常量。"""
        prior = {}
        utility = 0.0
        for state, probability in belief.items():
            next_at_goal = bool(dict(state).get("at_goal")) or bool(action.get("go"))
            utility += probability * (5.0 if bool(action.get("go")) else 0.0)
            next_key = self._state_key(next_at_goal)
            prior[next_key] = prior.get(next_key, 0.0) + probability
        observations = tuple(
            SimpleNamespace(
                observation=(("__state__", state),),
                label=self.state_label(state),
                probability=probability,
                belief={state: 1.0},
            )
            for state, probability in prior.items()
        )
        return SimpleNamespace(
            utility=utility,
            risk=0.0,
            prior_belief=prior,
            observations=observations,
        )

    def expand_safe_action(self, safe_belief, action):
        """Return a no-risk safe expansion. / 返回无风险 safe expansion。"""
        expansion = self.expand_action(safe_belief, action)
        return SimpleNamespace(
            utility=expansion.utility,
            risk=0.0,
            prior_belief=expansion.prior_belief,
            survival_probability=1.0,
            observations=expansion.observations,
        )

    def state_from_key(self, key):
        """Convert state key to mapping. / 将 state key 转成 mapping。"""
        return dict(key)

    def transition_distribution(self, state, action):
        """Return the deterministic fake transition. / 返回确定性 fake 转移。"""
        next_at_goal = bool(state.get("at_goal")) or bool(action.get("go"))
        return {self._state_key(next_at_goal): 1.0}

    def observation_probability(self, observation, state, action):
        """Return the MDP-state observation likelihood. / 返回 MDP-state 观测似然。"""
        del action
        return 1.0 if observation == (("__state__", state),) else 0.0

    def _state_key(self, at_goal):
        """Return a stable exact-state key. / 返回稳定 exact state key。"""
        return (("at_goal", bool(at_goal)),)
