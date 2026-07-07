"""Tests for exact finite kernels built from pyRDDLGym grounding."""

import pytest

from darp.adapter.exact import ExactBeliefState
from darp.adapter.loader import RDDLLoader
from darp.adapter.runtime import PyRDDLGymRuntime
from darp.model.and_or_tree import ANDORSearchInterface, ActionChoice, ObservationScope
from darp.model.duration_sidecar import build_duration_sidecar, load_duration_sidecar
from darp.planning.expand import expand_frontier_item
from darp.planning.ilp_tree import build_full_tree_ilp
from darp.planning.preprocess import initialize_root_frontier

DOMAIN = "experiments/inputs/rddl/tiny_grid_domain.rddl"
INSTANCE = "experiments/inputs/rddl/tiny_grid_instance.rddl"
FIXED_DURATION = "experiments/inputs/durations/tiny_grid.yaml"
GAUSSIAN_DURATION = "experiments/inputs/durations/tiny_grid_gaussian.yaml"


def _exact_inputs(duration_path=FIXED_DURATION):
    """Build exact tiny-grid inputs from pyRDDLGym grounding. / 从 pyRDDLGym grounding 构建 exact tiny-grid 输入。"""
    pytest.importorskip("pyRDDLGym")
    problem = RDDLLoader().load(DOMAIN, INSTANCE)
    runtime = PyRDDLGymRuntime.from_problem(problem)
    runtime.reset(seed=7)
    sidecar = load_duration_sidecar(duration_path)
    interface = problem.build_grounded_view().build_and_or_interface(runtime, risk=sidecar.risk_spec())
    duration = sidecar.evaluator(horizon=runtime.horizon)
    return runtime, interface, duration, sidecar


def test_exact_expand_uses_grounded_cpf_transition():
    """Check exact Expand follows grounded CPF transition instead of env.step sampling. / 检查 exact Expand 使用 grounded CPF 转移。"""
    runtime, interface, duration, _ = _exact_inputs()
    root_frontier = initialize_root_frontier(runtime, interface)
    move_east = next(item for item in root_frontier.frontier if item.action_label == "move-east")

    expanded = expand_frontier_item(move_east, interface, duration)

    assert move_east.belief is not None
    assert expanded.metrics.utility == -1.0
    assert expanded.metrics.risk == 0.0
    assert expanded.metrics.observation_probability == 1.0
    assert [frontier.probability for frontier in expanded.observation_frontiers] == [1.0]
    assert expanded.observation_frontiers[0].observation_node.metadata["observation"] == "at___c12"


def test_exact_kernel_computes_sidecar_next_state_risk():
    """Check sidecar C-POMDP risk can score risky next states. / 检查 sidecar C-POMDP risk 能给 risky next state 计分。"""
    runtime, interface, _, sidecar = _exact_inputs()
    exact = interface.exact_kernel
    assert exact is not None
    belief = exact.initial_belief_from_state({"at___c21": True})
    action = {name: False for name in exact.action_names}
    action["move-east"] = True

    expansion = exact.expand_action(belief, action)

    assert sidecar.risk_spec().budget == 0.25
    assert expansion.risk == 1.0


def test_exact_full_ilp_contains_sidecar_risk_row():
    """Check fixed-duration C-POMDP risk reaches the full-ILP row. / 检查 fixed-duration C-POMDP risk 进入 full-ILP 约束行。"""
    runtime, interface, _, sidecar = _exact_inputs()
    duration = sidecar.evaluator(horizon=2)

    tree_ilp = build_full_tree_ilp(
        runtime,
        interface,
        duration,
        risk_budget=sidecar.risk_spec().budget,
    )

    risk_row = next(constraint for constraint in tree_ilp.spec.constraints if constraint.name == "risk_budget")
    assert risk_row.rhs == 0.25
    assert risk_row.coefficients
    assert max(risk_row.coefficients.values()) == 1.0


def test_full_ilp_subtracts_initial_safe_belief_risk_budget():
    """Check Lemma 3.3 uses R = Delta - r(b0). / 检查 Lemma 3.3 会用 R = Delta - r(b0)。"""
    pytest.importorskip("pyRDDLGym")
    problem = RDDLLoader().load(DOMAIN, INSTANCE)
    runtime = PyRDDLGymRuntime.from_problem(problem)
    runtime.reset(seed=7)
    sidecar = build_duration_sidecar(
        {
            "kind": "fixed",
            "default": 1,
            "risk": {
                "budget": 0.75,
                "state_fluents": {"at___c11": 0.25},
            },
        }
    )
    interface = problem.build_grounded_view().build_and_or_interface(
        runtime,
        risk=sidecar.risk_spec(),
    )

    tree_ilp = build_full_tree_ilp(
        runtime,
        interface,
        sidecar.evaluator(horizon=1),
        risk_budget=sidecar.risk_spec().budget,
    )

    risk_row = next(constraint for constraint in tree_ilp.spec.constraints if constraint.name == "risk_budget")
    assert risk_row.rhs == pytest.approx(0.5)


def test_safe_belief_recursion_filters_risky_next_states():
    """Check CC-POMDP safe-belief recursion removes failed next states. / 检查 CC-POMDP safe belief 会滤掉失败 next state。"""
    _, interface, _, _ = _exact_inputs()
    exact = interface.exact_kernel
    assert exact is not None
    belief = exact.initial_belief_from_state({"at___c21": True})
    action = {name: False for name in exact.action_names}
    action["move-east"] = True

    expansion = exact.expand_safe_action(belief, action)

    assert expansion.risk == 1.0
    assert expansion.survival_probability == 0.0
    assert expansion.prior_belief == {}
    assert expansion.observations == ()


def test_gaussian_duration_uses_exact_belief_marginals():
    """Check Gaussian duration uses exact state-fluent belief marginals. / 检查 Gaussian duration 使用 exact state fluent belief 边缘概率。"""
    runtime, interface, duration, _ = _exact_inputs(GAUSSIAN_DURATION)
    exact = interface.exact_kernel
    assert exact is not None
    root_frontier = initialize_root_frontier(runtime, interface)
    move_south = next(item for item in root_frontier.frontier if item.action_label == "move-south")

    expanded = expand_frontier_item(move_south, interface, duration)

    assert expanded.metrics.zeta == 0.3
    assert expanded.metrics.duration.mean == 1.0
    assert 0.0 <= expanded.metrics.tau <= 1.0


def test_expand_computes_backward_message_and_smoothed_belief():
    """Check Algorithm 2 smoothing uses future observations. / 检查 Algorithm 2 smoothing 会吸收未来观测。"""
    kernel = _TwoStatePOMDPKernel()
    runtime = _TinyRuntime()
    interface = ANDORSearchInterface.from_actions_and_observations(
        actions=(ActionChoice(label="sense", assignment={"sense": True}),),
        observation_scope=ObservationScope(mode="pomdp-observation", variables=("see-a",)),
        exact_kernel=kernel,
    )
    duration = build_duration_sidecar({"kind": "fixed", "default": 1}).evaluator(horizon=2)
    root_frontier = initialize_root_frontier(runtime, interface)

    expanded = expand_frontier_item(root_frontier.frontier[0], interface, duration)
    branch = next(
        observation
        for observation in expanded.observation_frontiers
        if observation.observation_node.metadata["observation"] == "see-a"
    )

    state_a = kernel.state_key(True)
    state_b = kernel.state_key(False)
    assert branch.smoothing.backward_messages[0][state_a] == pytest.approx(0.74)
    assert branch.smoothing.backward_messages[0][state_b] == pytest.approx(0.18)
    assert branch.smoothing.smoothed_beliefs[0][state_a] == pytest.approx(0.8043478261)
    assert branch.smoothing.smoothed_beliefs[0][state_b] == pytest.approx(0.1956521739)


def test_exact_belief_state_advances_with_bayes_update():
    """Check paper-path online belief uses exact Bayes updates. / 检查论文路径 online belief 使用 exact Bayes 更新。"""
    kernel = _TwoStatePOMDPKernel()
    state_a = kernel.state_key(True)
    state_b = kernel.state_key(False)
    belief = ExactBeliefState.from_belief(
        kernel,
        {state_a: 0.5, state_b: 0.5},
        {"see-a": False},
        is_pomdp=True,
        source="test-root",
    )

    next_belief = belief.advance(kernel, {"sense": True}, {"see-a": True})

    assert next_belief.source == "exact-bayes"
    assert next_belief.belief[state_a] == pytest.approx(0.8804347826)
    assert next_belief.belief[state_b] == pytest.approx(0.1195652174)
    assert next_belief.support == {"a": pytest.approx(0.8804347826), "b": pytest.approx(0.1195652174)}


class _TinyRuntime:
    """Minimal runtime for exact Expand tests. / exact Expand 测试用极简 runtime。"""

    state = {}

    def clone(self):
        """Return an isolated runtime copy. / 返回隔离 runtime 副本。"""
        return _TinyRuntime()


class _TwoStatePOMDPKernel:
    """Tiny POMDP kernel whose future observation changes smoothed belief. / 未来观测会改变 smoothed belief 的小 POMDP。"""

    observation_names = ("see-a",)

    def state_key(self, is_a: bool):
        """Return a stable state key. / 返回稳定 state key。"""
        return (("is-a", is_a),)

    def initial_belief_from_state(self, state):
        """Return a uniform initial belief. / 返回均匀初始 belief。"""
        del state
        return {self.state_key(True): 0.5, self.state_key(False): 0.5}

    def state_from_key(self, key):
        """Convert state key to mapping. / 将 state key 转为 mapping。"""
        return dict(key)

    def state_label(self, key):
        """Return a compact state label. / 返回紧凑 state 标签。"""
        return "a" if dict(key).get("is-a") else "b"

    def fluent_belief(self, belief):
        """Return one fluent marginal for duration sidecars. / 返回 duration sidecar 使用的 fluent 边缘概率。"""
        return {"is-a": sum(prob for key, prob in belief.items() if dict(key).get("is-a"))}

    def expand_action(self, belief, action):
        """Return prior, observations, utility, and risk. / 返回 prior、观测、utility 和 risk。"""
        del action
        prior = {}
        for state, state_prob in belief.items():
            for next_state, trans_prob in self.transition_distribution(dict(state), {"sense": True}).items():
                prior[next_state] = prior.get(next_state, 0.0) + state_prob * trans_prob
        see_a_prob = sum(
            prob * self.observation_probability((("see-a", True),), state, {"sense": True})
            for state, prob in prior.items()
        )
        see_b_prob = 1.0 - see_a_prob
        return _Expansion(
            utility=0.0,
            risk=0.0,
            prior_belief=prior,
            observations=(
                _Outcome(
                    observation=(("see-a", True),),
                    label="see-a",
                    probability=see_a_prob,
                    belief=self._posterior(prior, (("see-a", True),), see_a_prob),
                ),
                _Outcome(
                    observation=(("see-a", False),),
                    label="see-b",
                    probability=see_b_prob,
                    belief=self._posterior(prior, (("see-a", False),), see_b_prob),
                ),
            ),
        )

    def expand_safe_action(self, safe_belief, action):
        """Return a no-risk safe expansion. / 返回无风险 safe expansion。"""
        expansion = self.expand_action(safe_belief, action)
        return _Expansion(
            utility=expansion.utility,
            risk=0.0,
            prior_belief=expansion.prior_belief,
            observations=expansion.observations,
            survival_probability=1.0,
        )

    def transition_distribution(self, state, action):
        """Return a two-state stochastic transition. / 返回双状态随机转移。"""
        del action
        if bool(state.get("is-a")):
            return {self.state_key(True): 0.8, self.state_key(False): 0.2}
        return {self.state_key(True): 0.1, self.state_key(False): 0.9}

    def observation_probability(self, observation, state, action):
        """Return observation likelihood O(o,s,a). / 返回观测似然 O(o,s,a)。"""
        del action
        see_a = bool(observation[0][1])
        is_a = bool(dict(state).get("is-a"))
        if see_a:
            return 0.9 if is_a else 0.1
        return 0.1 if is_a else 0.9

    def _posterior(self, prior, observation, probability):
        """Return normalized posterior belief. / 返回归一化 posterior belief。"""
        return {
            state: prior_prob * self.observation_probability(observation, state, {"sense": True}) / probability
            for state, prior_prob in prior.items()
        }


class _Outcome:
    """Small exact observation outcome. / 小型 exact observation outcome。"""

    def __init__(self, observation, label, probability, belief):
        """Store outcome fields. / 保存 outcome 字段。"""
        self.observation = observation
        self.label = label
        self.probability = probability
        self.belief = belief


class _Expansion:
    """Small exact action expansion. / 小型 exact action expansion。"""

    def __init__(self, utility, risk, prior_belief, observations, survival_probability=1.0):
        """Store expansion fields. / 保存 expansion 字段。"""
        self.utility = utility
        self.risk = risk
        self.prior_belief = prior_belief
        self.observations = observations
        self.survival_probability = survival_probability
