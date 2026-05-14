"""Tests for Phase 7 paper-aligned search scaffolding."""

from typing import Any, Mapping

from darp.model.and_or_tree import ANDORSearchInterface, ActionChoice, ObservationScope
from darp.model.duration_sidecar import load_duration_sidecar
from darp.planning.expand import expand_frontier_item
from darp.planning.full_ilp import FullILPPlanner
from darp.planning.hilp import HILPPlanner
from darp.planning.preprocess import preprocess_search_tree
from darp.planning.rollout import action_label

DURATIONS = "examples/durations/tiny_grid.yaml"


def _phase7_inputs():
    """Build shared tiny-grid Phase 7 inputs. / 构建 tiny-grid 的 Phase 7 共享输入。"""
    runtime = _TinyGridRuntime()
    interface = ANDORSearchInterface.from_actions_and_observations(
        actions=tuple(
            ActionChoice(label=action_label(action), assignment=action)
            for action in runtime.action_candidates()
        ),
        observation_scope=ObservationScope(
            mode="mdp-state",
            variables=(
                "at___c11",
                "at___c12",
                "at___c13",
                "at___c21",
                "at___c22",
                "at___c23",
                "at___c31",
                "at___c32",
                "at___c33",
            ),
        ),
    )
    duration = load_duration_sidecar(DURATIONS).evaluator(horizon=runtime.horizon)
    return runtime, interface, duration


def test_preprocess_initializes_root_action_frontier():
    """Check preprocessing creates one root frontier item per action. / 检查 preprocessing 为每个 action 创建根 frontier。"""
    runtime, interface, _ = _phase7_inputs()

    tree = preprocess_search_tree(runtime, interface)
    labels = [item.action_label for item in tree.frontier]

    assert tree.root.node_id == "root"
    assert len(tree.root.children) == len(tree.frontier)
    assert labels == ["noop", "move-east", "move-south", "move-west", "move-north"]
    assert all(item.rho == 1.0 for item in tree.frontier)


def test_expand_frontier_item_computes_paper_metrics():
    """Check Expand computes rho/u/r/tau-style metrics. / 检查 Expand 计算 rho/u/r/tau 风格指标。"""
    runtime, interface, duration = _phase7_inputs()
    tree = preprocess_search_tree(runtime, interface)
    move_east = next(item for item in tree.frontier if item.action_label == "move-east")

    expanded = expand_frontier_item(move_east, interface, duration)

    assert expanded.metrics.reward == -1.0
    assert expanded.metrics.utility == -1.0
    assert expanded.metrics.risk == 0.0
    assert expanded.metrics.rho == 1.0
    assert expanded.metrics.tau == 7.0
    assert expanded.metrics.duration.mean == 1.0
    assert expanded.metrics.observation_label == "at___c12"
    assert [child.action_label for child in expanded.child_frontier] == [
        "noop",
        "move-east",
        "move-south",
        "move-west",
        "move-north",
    ]


def test_full_tree_baseline_chooses_tiny_grid_goal_action():
    """Check the full-tree baseline chooses the optimal first tiny-grid move. / 检查 full-tree baseline 选择 tiny-grid 的最优首步。"""
    runtime, interface, duration = _phase7_inputs()
    planner = FullILPPlanner(lookahead_depth=4)

    decision = planner.choose_action(
        runtime,
        interface,
        duration,
        remaining_depth=runtime.horizon,
    )

    assert decision.label == "move-east"
    assert decision.action_values["move-east"] > decision.action_values["noop"]
    assert decision.complete is True


def test_hilp_partial_frontier_search_uses_expand_bookkeeping():
    """Check HILP keeps frontier stats while choosing the same root action. / 检查 HILP 保留 frontier 统计并选择相同根动作。"""
    runtime, interface, duration = _phase7_inputs()
    planner = HILPPlanner(lookahead_depth=4, max_iterations=3, frontier_width=1)

    decision = planner.choose_action(
        runtime,
        interface,
        duration,
        remaining_depth=runtime.horizon,
    )

    assert decision.label == "move-east"
    assert planner.last_stats is not None
    assert planner.last_stats.expanded_count >= 1
    assert planner.last_stats.used_gurobi is False


class _TinyGridRuntime:
    """Small pyRDDLGym-shaped deterministic runtime for Phase 7 tests. / Phase 7 测试用的小型 deterministic runtime。"""

    horizon = 8
    discount = 1.0

    def __init__(self, location: str = "c11") -> None:
        """Initialize the active grid location. / 初始化当前 grid 位置。"""
        self.location = location
        self.state = {f"at___{location}": True}

    def clone(self) -> "_TinyGridRuntime":
        """Return an isolated runtime copy. / 返回隔离 runtime 副本。"""
        return _TinyGridRuntime(self.location)

    def action_candidates(self) -> tuple[dict[str, Any], ...]:
        """Return noop plus one-active movement actions. / 返回 noop 加单动作移动候选。"""
        base = {
            "move-east": False,
            "move-south": False,
            "move-west": False,
            "move-north": False,
        }
        return (
            dict(base),
            {**base, "move-east": True},
            {**base, "move-south": True},
            {**base, "move-west": True},
            {**base, "move-north": True},
        )

    def step(self, action: Mapping[str, Any]):
        """Apply one tiny-grid transition and reward. / 执行一次 tiny-grid 转移和 reward。"""
        label = action_label(action)
        reward = _tiny_grid_reward(self.location, label)
        self.location = _TRANSITIONS.get((self.location, label), self.location)
        self.state = {f"at___{self.location}": True}
        observation = dict(self.state)
        return observation, reward, False, False, {}


_TRANSITIONS = {
    ("c11", "move-east"): "c12",
    ("c12", "move-east"): "c13",
    ("c21", "move-east"): "c22",
    ("c22", "move-east"): "c23",
    ("c31", "move-east"): "c32",
    ("c32", "move-east"): "c33",
    ("c11", "move-south"): "c21",
    ("c12", "move-south"): "c22",
    ("c13", "move-south"): "c23",
    ("c21", "move-south"): "c31",
    ("c22", "move-south"): "c32",
    ("c23", "move-south"): "c33",
    ("c12", "move-west"): "c11",
    ("c13", "move-west"): "c12",
    ("c22", "move-west"): "c21",
    ("c23", "move-west"): "c22",
    ("c32", "move-west"): "c31",
    ("c33", "move-west"): "c32",
    ("c21", "move-north"): "c11",
    ("c22", "move-north"): "c12",
    ("c23", "move-north"): "c13",
    ("c31", "move-north"): "c21",
    ("c32", "move-north"): "c22",
    ("c33", "move-north"): "c23",
}


def _tiny_grid_reward(location: str, label: str) -> float:
    """Return tiny-grid reward before the transition. / 返回 tiny-grid 转移前 reward。"""
    risky = {
        ("c12", "move-south"),
        ("c21", "move-east"),
        ("c23", "move-west"),
        ("c32", "move-north"),
    }
    goal = {
        ("c32", "move-east"),
        ("c23", "move-south"),
    }
    if (location, label) in risky:
        return -10.0
    if (location, label) in goal or location == "c33":
        return 20.0
    return -1.0
