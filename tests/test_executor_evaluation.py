"""Regression checks for RDDL semantics, incremental ILP and policy execution.

Run without Gurobi: ``python -m unittest discover -s tests -p test_executor_evaluation.py``.

/ 无需 Gurobi，验证 RDDL 语义、增量 ILP、策略回放及风险与时长统计。
"""

import csv
import unittest
from dataclasses import asdict, replace
from importlib import import_module
from math import sqrt
from pathlib import Path
from random import Random
from statistics import fmean
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

from darp.adapter.duration import build_duration_evaluator
from darp.adapter.grounded import GroundedRDDLView
from darp.adapter.kernel import RDDLKernel
from darp.adapter.loader import load_rddl
from darp.adapter.problem import PyRDDLGymProblem, RDDLLoadError
from darp.adapter.runtime import PyRDDLGymRuntime
from darp.executor import PolicyExecutor
from darp.ilp.gurobi import (
    GurobiILPSession,
    _optional_attr,
    _optional_float,
    _set_objective_coefficient,
    _set_param,
    _set_start,
)
from darp.ilp.model import (
    ILPLinearConstraint,
    ILPModelDelta,
    ILPModelSpec,
    ILPSolveResult,
    ILPVariable,
)
from darp.model.and_or_tree import ANDORSearchInterface
from darp.planning.expand import apply_terminal_heuristic, expand_frontier_item
from darp.planning.heuristic import UtilityHeuristic
from darp.planning.hilp import (
    HILPPlanner,
    _frontier_leaf_record,
)
from darp.planning.ilp_tree import (
    IncrementalPartialTreeILP,
    PolicyTreeILP,
    build_full_tree_ilp,
    build_partial_tree_ilp,
)
from darp.planning.policy import (
    ConditionalPolicy,
    PolicyNode,
    extract_conditional_policy,
)
from darp.planning.preprocess import initialize_root_frontier, resolve_root_belief
from darp.solve import DARPResult

ROOT = Path(__file__).resolve().parents[1]
TABLE1 = ROOT / "experiments" / "DARP-table1-grid"


def _policy(slow_actions: tuple[bool, ...]) -> ConditionalPolicy:
    nodes = tuple(
        PolicyNode(
            node_id=str(index),
            stage=index,
            action_label="slow" if slow else "noop",
            assignment={"slow": slow},
            transitions={
                (("seen", False),): (
                    str(index + 1) if index + 1 < len(slow_actions) else None
                )
            },
        )
        for index, slow in enumerate(slow_actions)
    )
    return ConditionalPolicy(
        root="0",
        nodes=nodes,
        input_kind="observation",
        solver_status="test-policy",
        duration_complete=True,
        achieved_utility=-float(len(nodes)),
        active_constraint_value=1.0,
        feasible=True,
    )


def _environment(directory: Path, duration: str | None, horizon: int, *, initial_risk=False, risk="unsafe", budget="1.0", termination="false"):
    domain = directory / "domain.rddl"
    instance = directory / "instance.rddl"
    domain.write_text(
        """domain executor_test {
    requirements = { partially-observed };
    pvariables {
        hidden : { state-fluent, bool, default = false };
        unsafe : { state-fluent, bool, default = false };
        slow : { action-fluent, bool, default = false };
        seen : { observ-fluent, bool };
        enabled : { non-fluent, bool, default = true };
        alarm : { interm-fluent, bool, level = 1 };
    };
    cpfs {
        hidden' = Bernoulli(0.5);
        unsafe' = true;
        seen = false;
        alarm = unsafe & enabled;
    };
    reward = -1;
"""
        + (f"    duration = {duration};\n" if duration is not None else "")
        + (f"    risk = {risk};\n" if risk is not None else "")
        + "    termination { " + termination + "; };\n}\n",
        encoding="utf-8",
    )
    instance.write_text(
        """non-fluents test_nf { domain = executor_test; }
instance test_instance {
    domain = executor_test;
    non-fluents = test_nf;
    init-state { unsafe = """ + ("true" if initial_risk else "false") + "; };\n"
        + "    max-nondef-actions = 1;\n"
        + (f"    risk-budget = {budget};\n" if budget is not None else "")
        + f"    horizon = {horizon};\n    discount = 1.0;\n}}\n",
        encoding="utf-8",
    )
    return load_rddl(domain, instance).env


def _terminal_problem(directory, *, transition="if (pos == 3) then 2 else if (Bernoulli(0.5)) then 3 else 0", duration="1", risk="pos == 2", horizon=2, threshold="", observation="false", reward="1"):
    domain = directory / "domain.rddl"
    instance = directory / "instance.rddl"
    domain.write_text(f"""domain terminal_test {{
    requirements = {{ partially-observed }};
    pvariables {{
        pos : {{ state-fluent, int, default = 0 }};
        slow : {{ action-fluent, bool, default = false }};
        seen : {{ observ-fluent, bool }};
    }};
    cpfs {{ pos' = {transition}; seen = {observation}; }};
    reward = {reward};
    duration = {duration};
    risk = {risk};
    termination {{ pos == 3; }};
}}
""", encoding="utf-8")
    instance.write_text(f"""non-fluents terminal_nf {{ domain = terminal_test; }}
instance terminal_instance {{
    domain = terminal_test;
    non-fluents = terminal_nf;
    max-nondef-actions = 1;
    horizon = {horizon};
    discount = 1.0;
    risk-budget = 1.0;
    {threshold}
}}
""", encoding="utf-8")
    return load_rddl(domain, instance)


def _terminal_search(problem):
    runtime = PyRDDLGymRuntime(problem.env)
    runtime.reset(seed=0)
    view = problem.build_grounded_view()
    interface = view.build_and_or_interface(runtime)
    # One action makes the entire tree the unique policy; no optimizer needed.
    # 单动作使整棵树成为唯一策略，不需要优化器。
    interface = ANDORSearchInterface.from_actions_and_observations(
        actions=tuple(action for action in interface.actions if action.label == "noop"),
        observation_mode=interface.observation_mode,
        kernel=interface.kernel,
    )
    evaluator = build_duration_evaluator(interface.kernel, interface.actions, horizon=runtime.horizon)
    return runtime, interface, evaluator


def _terminal_tree(problem, **options):
    runtime, interface, evaluator = _terminal_search(problem)
    view = problem.build_grounded_view()
    return build_full_tree_ilp(runtime, interface, evaluator, risk_budget=view.grounded_model.risk_budget, **options)


def _only_policy(tree):
    variables = tree.spec.variable_ids()
    return extract_conditional_policy(tree, ILPSolveResult(
        status="test-single-action", objective_value=sum(tree.spec.objective.values()),
        variable_values=dict.fromkeys(variables, 1.0), selected_variables=variables, runtime_ms=0,
    ))


class ExecutorEvaluationTests(unittest.TestCase):
    def test_observation_mode_preserves_root_belief_source(self):
        declared = {(("pos", 0),): 1.0}
        observed = {(("pos", 7),): 1.0}
        for fluents, mode, expected in (
            ({"seen": False}, "pomdp-observation", declared),
            ({}, "mdp-state", observed),
        ):
            with self.subTest(mode=mode):
                view = GroundedRDDLView(Mock(observ_fluents=fluents))
                self.assertEqual(view.observation_mode(), mode)
                kernel = Mock()
                kernel.initial_belief_from_model.return_value = declared
                kernel.initial_belief_from_state.return_value = observed
                interface = ANDORSearchInterface.from_actions_and_observations(
                    actions=(), observation_mode=view.observation_mode(), kernel=kernel,
                )
                runtime = Mock(state={"pos": 7})
                self.assertEqual(resolve_root_belief(runtime, interface, None), expected)
                if mode == "pomdp-observation":
                    kernel.initial_belief_from_model.assert_called_once_with()
                    kernel.initial_belief_from_state.assert_not_called()
                else:
                    kernel.initial_belief_from_state.assert_called_once_with(runtime.state)
                    kernel.initial_belief_from_model.assert_not_called()
                self.assertEqual(resolve_root_belief(runtime, interface, declared), declared)

    def test_policy_graph_depth_and_validation(self):
        policy = _policy((False, False, False, False))
        root, left, right, leaf = policy.nodes
        unseen, seen = (("seen", False),), (("seen", True),)
        policy = replace(policy, nodes=(
            replace(root, transitions={unseen: left.node_id, seen: right.node_id}),
            replace(left, transitions={unseen: leaf.node_id, seen: None}),
            replace(right, stage=1, transitions={unseen: leaf.node_id}),
            replace(leaf, stage=2),
        ))
        self.assertEqual(PolicyExecutor(policy).max_steps, 3)
        root, left, right, leaf = policy.nodes
        for invalid, message in (
            (replace(policy, root="missing"), "root does not name"),
            (replace(policy, nodes=policy.nodes + (leaf,)), "duplicate node ids"),
            (replace(policy, nodes=(replace(root, stage=1), left, right, leaf)), "stage zero"),
            (replace(policy, nodes=(replace(root, transitions={}), left, right, leaf)), "no outcomes"),
            (replace(policy, nodes=(root, replace(left, transitions={unseen: "missing"}), right, leaf)), "unknown node"),
            (replace(policy, nodes=(root, left, replace(right, stage=2), leaf)), "advance exactly one stage"),
            (replace(policy, nodes=(root, left, right, replace(leaf, transitions={unseen: root.node_id}))), "advance exactly one stage"),
            (replace(policy, nodes=policy.nodes + (replace(leaf, node_id="unreachable"),)), "unreachable"),
        ):
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                PolicyExecutor(invalid)

    def test_saved_f_e_s_policies_execute(self):
        for model in ("f", "e", "s"):
            with self.subTest(model=model):
                result = DARPResult.load(
                    TABLE1 / "output" / "results" / "smoke-raw"
                    / f"darp-{model}-h3-d0p1-full-ilp-trial01.json"
                )
                env = load_rddl(TABLE1 / "rddl" / "domain.rddl", TABLE1 / "rddl" / f"instance_{model}_h3.rddl").env
                try:
                    agent = PolicyExecutor(result.decision.policy)
                    statistics = agent.evaluate(
                        env, episodes=10, seed=7,
                    )
                    self.assertTrue(agent.at_leaf)
                    self.assertEqual(env.horizon, 3)
                    self.assertEqual(statistics["episodes"], 10)
                    self.assertGreater(statistics["physical_duration_mean"], 0)
                    self.assertTrue(0 <= statistics["risk_rate"] <= 1)
                finally:
                    env.close()

    def test_duration_sampling_and_first_failure_count(self):
        rng = Random(19)
        sampled_mean = fmean(rng.gauss(1, sqrt(0.1)) + rng.gauss(1, sqrt(0.1)) for _ in range(10))
        for duration, expected, initial_risk in (
            ("1", 2.0, False),
            ("if (slow) then 2 else 1", 3.0, True),
            ("if (hidden) then 3 else 1", None, False),
            ("Normal(1, 0.1)", sampled_mean, False),
        ):
            with self.subTest(duration=duration), TemporaryDirectory() as temporary:
                env = _environment(Path(temporary), duration, 2, initial_risk=initial_risk)
                try:
                    policy = _policy((False, True))
                    # Display labels need not uniquely identify RDDL assignments.
                    # 显示标签不必唯一标识 RDDL 动作赋值。
                    policy = replace(policy, nodes=tuple(replace(node, action_label="move") for node in policy.nodes))
                    agent = PolicyExecutor(policy)
                    statistics = agent.evaluate(env, episodes=10, seed=19)
                    self.assertEqual(statistics["risk_rate"], 1.0)
                    self.assertEqual(statistics["mean"], -2.0)
                    self.assertTrue(agent.at_leaf)
                    if expected is not None:
                        self.assertAlmostEqual(statistics["physical_duration_mean"], expected)
                    else:
                        self.assertTrue(2 <= statistics["physical_duration_mean"] <= 4)
                finally:
                    env.close()

    def test_rddl_risk_predicate(self):
        for expression, frequency in (("alarm", 1.0), ("if (enabled) then unsafe else false", 1.0), ("false", 0.0)):
            with self.subTest(risk=expression), TemporaryDirectory() as temporary:
                env = _environment(Path(temporary), "1", 2, risk=expression, budget="0.2")
                try:
                    self.assertEqual(env.model.ast.instance.risk_budget, 0.2)
                    statistics = PolicyExecutor(_policy((False, False))).evaluate(env, episodes=2, seed=19)
                    self.assertEqual(statistics["risk_rate"], frequency)
                finally:
                    env.close()

    def test_optional_rddl_extensions(self):
        for duration, risk, budget, expected_duration, expected_risk in (
            (None, None, None, 2.0, 0.0),
            ("if (slow) then 3 else 2", None, None, 5.0, 0.0),
            (None, None, "0.3", 2.0, 0.0),
            (None, "unsafe", "0.2", 2.0, 1.0),
            (None, "false", "0.0", 2.0, 0.0),
        ):
            with self.subTest(duration=duration, risk=risk, budget=budget), TemporaryDirectory() as temporary:
                env = _environment(Path(temporary), duration, 2, initial_risk=True, risk=risk, budget=budget)
                try:
                    ast = env.model.ast
                    problem = PyRDDLGymProblem(ast, env)
                    grounded = problem.build_grounded_model()
                    self.assertIs(problem.build_grounded_model(), grounded)
                    self.assertIs(problem.build_grounded_view().grounded_model, grounded)
                    self.assertEqual(grounded.risk_budget, 0.0 if budget is None else float(budget))
                    if duration is None:
                        self.assertEqual(grounded.duration.etype[0], "constant")
                        self.assertIsInstance(grounded.duration.value, float)
                        self.assertEqual(grounded.duration.value, 1.0)
                    if risk is None:
                        self.assertIs(grounded.risk.value, False)
                    statistics = PolicyExecutor(_policy((False, True))).evaluate(env, episodes=2, seed=19)
                    self.assertEqual(statistics["physical_duration_mean"], expected_duration)
                    self.assertEqual(statistics["risk_rate"], expected_risk)
                    self.assertTrue(env.state["unsafe"])
                    for owner, name, supplied in (
                        (ast.domain, "duration", duration),
                        (ast.domain, "risk", risk),
                        (ast.instance, "risk_budget", budget),
                    ):
                        if supplied is None:
                            self.assertIsNone(getattr(owner, name, None))
                finally:
                    env.close()

    def test_explicit_risk_requires_budget_even_when_false(self):
        for risk in ("unsafe", "false"):
            with self.subTest(risk=risk), TemporaryDirectory() as temporary:
                with self.assertRaisesRegex(RDDLLoadError, "risk-budget"):
                    _environment(Path(temporary), None, 2, risk=risk, budget=None)

    def test_native_environment_uses_standard_rddl_defaults(self):
        import pyRDDLGym

        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            _environment(directory, None, 2, initial_risk=True, risk=None, budget=None).close()
            env = pyRDDLGym.make(str(directory / "domain.rddl"), str(directory / "instance.rddl"))
            try:
                executor = PolicyExecutor(_policy((False, True)))
                self.assertEqual(executor.history, ())
                self.assertEqual(executor.sample_action(None), {"slow": False})
                self.assertEqual(executor.sample_action({"seen": False}), {"slow": True})
                history = executor.history
                self.assertEqual(history, ((("seen", False),),))
                self.assertIsNone(executor.sample_action({"seen": False}))
                self.assertEqual(executor.history, history + history)
                self.assertEqual(history, ((("seen", False),),))
                executor.reset()
                self.assertEqual(executor.history, ())
                self.assertEqual(history, ((("seen", False),),))
                steps = executor.run_episode(env, seed=19).steps
                self.assertEqual(steps, 2)
                for _ in range(2):
                    statistics = executor.evaluate(env, episodes=2, seed=19)
                    self.assertEqual(statistics["risk_rate"], 0.0)
                    self.assertEqual(statistics["physical_duration_mean"], float(steps))
                    self.assertTrue(env.state["unsafe"])
                self.assertIsNone(getattr(env.model.ast.domain, "duration", None))
                self.assertIsNone(getattr(env.model.ast.domain, "risk", None))
                self.assertIsNone(getattr(env.model.ast.instance, "risk_budget", None))
            finally:
                env.close()

    def test_default_and_explicit_extensions_build_the_same_tree(self):
        trees = []
        policies = []
        for duration, risk, budget in ((None, None, None), ("1.0", "false", "0.0")):
            with TemporaryDirectory() as temporary:
                env = _environment(Path(temporary), duration, 2, initial_risk=True, risk=risk, budget=budget)
                try:
                    tree = _terminal_tree(PyRDDLGymProblem(env.model.ast, env))
                    trees.append(tree)
                    policies.append(_only_policy(tree))
                finally:
                    env.close()
        self.assertEqual(trees[0].spec, trees[1].spec)
        self.assertEqual(policies[0], policies[1])
        self.assertEqual(policies[0].achieved_utility, -2.0)
        self.assertEqual(policies[0].active_constraint_value, 0.0)
        self.assertTrue(policies[0].duration_complete)
        self.assertTrue(policies[0].feasible)
        self.assertEqual(next(row.rhs for row in trees[0].spec.constraints if row.name == "risk_budget"), 0.0)

    def test_invalid_rddl_risk_is_rejected(self):
        for expression in ("slow", "unsafe'", "seen", "Bernoulli(0.5)", "0", "0.1"):
            with self.subTest(risk=expression), TemporaryDirectory() as temporary:
                env = _environment(Path(temporary), "1", 2, risk=expression)
                try:
                    with self.assertRaises(ValueError):
                        PolicyExecutor(_policy((False, False))).evaluate(env)
                finally:
                    env.close()
        for budget in ("true", "1.1", "-0.1", "0.1 + 0.1"):
            with self.subTest(budget=budget), TemporaryDirectory() as temporary:
                with self.assertRaises(RDDLLoadError):
                    _environment(Path(temporary), "1", 2, budget=budget)

    def test_initial_terminal_executes_no_actions(self):
        for initial_risk in (False, True):
            with self.subTest(initial_risk=initial_risk), TemporaryDirectory() as temporary:
                env = _environment(Path(temporary), "1", 2, initial_risk=initial_risk, termination="true")
                try:
                    agent = PolicyExecutor(_policy((False, False)))
                    with patch.object(env, "step", side_effect=AssertionError("terminal episode must not step")):
                        result = agent.run_episode(env)
                        self.assertEqual(result.steps, 0)
                        self.assertEqual(result.total_reward, 0)
                        self.assertTrue(result.terminated)
                        self.assertEqual(result.stop_reason, "model_terminal")
                        statistics = agent.evaluate(env, episodes=2)
                        self.assertEqual(statistics["risk_rate"], float(initial_risk))
                        self.assertEqual(statistics["mean"], 0)
                        self.assertEqual(statistics["physical_duration_mean"], 0)
                        self.assertEqual(env.horizon, 2)
                finally:
                    env.close()

    def test_terminal_mass_preserves_last_reward_and_risk(self):
        for observation in ("false", "pos' == 3"):
            for risk, expected in (("pos == 2", 0), ("pos == 3", .75)):
                with self.subTest(observation=observation, risk=risk), TemporaryDirectory() as temporary:
                    problem = _terminal_problem(Path(temporary), risk=risk, observation=observation)
                    try:
                        tree = _terminal_tree(problem)
                        policy = _only_policy(tree)
                        self.assertTrue(policy.duration_complete)
                        self.assertTrue(policy.feasible)
                        self.assertAlmostEqual(policy.achieved_utility, 1.5)
                        self.assertAlmostEqual(policy.active_constraint_value, expected)
                        for item in tree.variable_items.values():
                            self.assertTrue(all(dict(s)["pos"] == 0 for s in item.ordinary_mass))
                        statistics = PolicyExecutor(policy).evaluate(problem.env, episodes=500, seed=42)
                        self.assertAlmostEqual(statistics["mean"], 1.5, delta=.08)
                        self.assertAlmostEqual(statistics["risk_rate"], expected, delta=.08)
                    finally:
                        problem.env.close()

    def test_terminal_checks_are_cached_per_state(self):
        with TemporaryDirectory() as temporary:
            problem = _terminal_problem(Path(temporary))
            try:
                kernel = RDDLKernel.from_grounded_model(problem.build_grounded_model())
                live, terminal = (("pos", 0),), (("pos", 3),)
                expression = kernel.grounded_model.terminations[0]
                evaluate = RDDLKernel.expression_distribution
                calls = []

                def counted(instance, expr, context):
                    if instance is kernel and expr is expression:
                        calls.append(context["pos"])
                    return evaluate(instance, expr, context)

                with patch.object(RDDLKernel, "expression_distribution", new=counted):
                    for _ in range(3):
                        self.assertEqual(kernel.continuing_mass({live: .4, terminal: .6}), {live: .4})
                        self.assertFalse(kernel.belief_is_terminal({live: 1.0}))
                        self.assertTrue(kernel.belief_is_terminal({terminal: 1.0, live: 0.0}))
                        self.assertFalse(kernel.belief_is_terminal({terminal: 0.0}))
                    self.assertCountEqual(calls, [0, 3])
            finally:
                problem.env.close()

    def test_cached_observation_utility_preserves_action_and_next_state_reward(self):
        transition = "if (Bernoulli(if (slow) then 0.25 else 0.5)) then 3 else if (Bernoulli(0.5)) then 1 else 2"
        observation = "if (pos' == 1) then Bernoulli(if (slow) then 0.6 else 0.2) else if (pos' == 2) then Bernoulli(if (slow) then 0.1 else 0.8) else true"
        with TemporaryDirectory() as temporary:
            problem = _terminal_problem(Path(temporary), transition=transition, observation=observation, reward="10 * pos + pos' + (if (slow) then 100 else 0)")
            try:
                kernel = RDDLKernel.from_grounded_model(problem.build_grounded_model())
                source0, source1 = (("pos", 0),), (("pos", 1),)
                evaluate = RDDLKernel.expected_reward
                calls = []

                def counted(instance, context):
                    if instance is kernel:
                        calls.append((context["pos"], context["slow"], context["pos'"]))
                    return evaluate(instance, context)

                # The two live successor rewards differ; terminal arrivals keep
                # their reward in the total, but never in continuation branches.
                # 存活后继的奖励不同；终止到达的奖励计入总量，但不进入后续分支。
                for slow, seen, probability, expected in (
                    (False, True, .25, .725),
                    (False, False, .25, .65),
                    (True, True, .2625, 13.8),
                    (True, False, .4875, 25.7625),
                ):
                    with self.subTest(slow=slow, seen=seen), patch.object(RDDLKernel, "expected_reward", new=counted):
                        action, observation_key = {"slow": slow}, (("seen", seen),)
                        mass, utility = kernel.action_start_mass_and_utility_for_observation({source0: .3, source1: .2}, action, observation_key)
                        self.assertAlmostEqual(mass[source0], .3 * probability)
                        self.assertAlmostEqual(mass[source1], .2 * probability)
                        self.assertAlmostEqual(utility, expected)
                        warmed_calls = len(calls)
                        scaled, scaled_utility = kernel.action_start_mass_and_utility_for_observation({source0: .6, source1: .4}, action, observation_key)
                        self.assertAlmostEqual(scaled[source0], 2 * mass[source0])
                        self.assertAlmostEqual(scaled[source1], 2 * mass[source1])
                        self.assertAlmostEqual(scaled_utility, 2 * expected)
                        self.assertEqual(len(calls), warmed_calls)
                self.assertAlmostEqual(kernel.utility_coefficient_for_mass({source0: .3, source1: .2}, {"slow": False}), 3.125)
                self.assertAlmostEqual(kernel.utility_coefficient_for_mass({source0: .3, source1: .2}, {"slow": True}), 52.9375)
                self.assertEqual(kernel.action_start_mass_and_utility_for_observation({source0: 1.0}, {"slow": False}, (("seen", 2),)), ({}, 0.0))
            finally:
                problem.env.close()

    def test_terminal_initial_mass_is_not_renormalized(self):
        with TemporaryDirectory() as temporary:
            problem = _terminal_problem(Path(temporary), risk="pos == 3")
            try:
                tree = _terminal_tree(problem, root_belief={(("pos", 0),): .75, (("pos", 3),): .25})
                policy = _only_policy(tree)
                self.assertEqual(tree.risk_budget, 1.0)
                self.assertEqual(tree.initial_risk, .25)
                self.assertAlmostEqual(policy.achieved_utility, 1.125)
                self.assertAlmostEqual(policy.active_constraint_value, .8125)
                row = next(row for row in tree.spec.constraints if row.name == "risk_budget")
                self.assertEqual(row.rhs, .75)
                with self.assertRaisesRegex(ValueError, "already terminal"):
                    _terminal_tree(problem, root_belief={(("pos", 3),): 1.0})
            finally:
                problem.env.close()

    def test_terminal_arrival_is_not_replaced_by_heuristic(self):
        for transition, expected in (("if (Bernoulli(0.5)) then 3 else 0", 50.5), ("3", 1.0)):
            with self.subTest(transition=transition), TemporaryDirectory() as temporary:
                problem = _terminal_problem(Path(temporary), transition=transition, horizon=1, risk="pos == 3")
                try:
                    tree = _terminal_tree(problem, terminal_heuristic=UtilityHeuristic("test", lambda _: 100.0))
                    self.assertEqual(_only_policy(tree).achieved_utility, expected)
                finally:
                    problem.env.close()

    def test_duration_conditions_on_not_terminated(self):
        transition = "if (pos == 0) then (if (Bernoulli(0.5)) then 1 else 2) else if (pos == 1) then 3 else pos"
        expected_duration = "if (pos == 1) then 10 else 1"
        threshold = "max-duration-shortfall-probability = 0.3;"
        for duration, limit, actions in (("1", "", 3), (expected_duration, "", 3), (f"Normal({expected_duration}, 0.1)", threshold, 4), (expected_duration, threshold, 3)):
            with self.subTest(duration=duration, threshold=limit), TemporaryDirectory() as temporary:
                problem = _terminal_problem(Path(temporary), transition=transition, duration=duration, horizon=3, threshold=limit, risk="pos == 3")
                try:
                    tree = _terminal_tree(problem)
                    policy = _only_policy(tree)
                    self.assertEqual(len(policy.nodes), actions)
                    self.assertEqual(policy.achieved_utility, 2 + .5 * (actions - 2))
                    self.assertEqual(policy.active_constraint_value, .5)
                    third = next(item for item in tree.variable_items.values() if len(item.observation_keys) == 2)
                    self.assertEqual(third.ordinary_mass, {(("pos", 2),): .5})
                    self.assertEqual(third.duration_progress.mean, 2)
                    if third.duration_progress.augmented_belief is not None:
                        self.assertEqual(third.duration_progress.augmented_belief, {((("pos", 2),), 2.0): 1.0})
                finally:
                    problem.env.close()


class IncrementalHILPTests(unittest.TestCase):
    """Keep incremental session checks with the existing regressions.

    / 在现有回归文件中保留通用的增量求解会话检查。
    """

    def test_refinement_matches_full_encoding_and_preserves_snapshots(self):
        with TemporaryDirectory() as temporary:
            problem = _terminal_problem(Path(temporary), risk="pos == 3")
            try:
                runtime, interface, evaluator = _terminal_search(problem)
                heuristic = UtilityHeuristic("test", lambda _: 10.0)

                def leaf(item):
                    return _frontier_leaf_record(
                        item, interface, evaluator,
                        heuristic=heuristic, terminal_heuristic=True,
                    )

                frontier = [leaf(item) for item in initialize_root_frontier(runtime, interface)]
                builder = IncrementalPartialTreeILP(runtime, interface, risk_budget=1.0)
                builder.update(expanded_records=[], frontier_records=frontier)
                expanded = []
                checkpoints = []
                while True:
                    tree, delta = builder.snapshot()
                    rebuilt = build_partial_tree_ilp(
                        runtime=runtime, interface=interface,
                        expanded_records=expanded, frontier_records=frontier,
                        risk_budget=1.0,
                    )
                    self.assertEqual(tree, rebuilt)
                    for old_tree, old_delta, saved_tree, saved_delta in checkpoints:
                        self.assertEqual(asdict(old_tree), saved_tree)
                        self.assertEqual(asdict(old_delta), saved_delta)
                    if not frontier:
                        break
                    checkpoints.append((tree, delta, asdict(tree), asdict(delta)))
                    changed = []
                    children = []
                    for record in frontier:
                        expansion = expand_frontier_item(record.item, interface, evaluator)
                        expansion = apply_terminal_heuristic(
                            record.item, expansion, interface, heuristic,
                        )
                        changed.append(replace(
                            record, ilp_metrics=expansion.metrics,
                            continues=bool(expansion.child_frontier),
                            policy_expansion=expansion,
                        ))
                        children.extend(leaf(item) for item in expansion.child_frontier)
                    expanded.extend(changed)
                    frontier = children
                    builder.update(expanded_records=changed, frontier_records=frontier)

                first_tree, first_delta, _, _ = checkpoints[0]
                root = first_tree.root_variable_ids[0]
                self.assertEqual(first_tree.spec.objective[root], 10.0)
                self.assertEqual(tree.spec.objective[root], 1.0)
                self.assertEqual(first_delta.variables, first_tree.spec.variables)
                self.assertEqual(checkpoints[1][1].objective[root], 1.0)
                risk = next(row for row in tree.spec.constraints if row.name == "risk_budget")
                self.assertEqual(sorted(risk.coefficients.values()), [.25, .5])
                self.assertTrue(_only_policy(tree).duration_complete)
            finally:
                problem.env.close()

    def test_selected_frontier_skips_rescoring_but_keeps_terminal_utility(self):
        """Score new F leaves and actual terminal utility, not F→E again.

        / 仅计算新 F 的估值及实际终端效用，F→E 时不重复计算将被丢弃的估值。
        """
        for rounds, calls, solves in ((1, 2, 2), (None, 3, 3)):
            with self.subTest(rounds=rounds), TemporaryDirectory() as temporary:
                problem = _terminal_problem(Path(temporary), transition="0", horizon=2)
                try:
                    runtime, interface, evaluator = _terminal_search(problem)
                    callback = Mock(return_value=10.0)
                    session = Mock(last_model_update_ms=0.0, last_optimize_ms=0.0)

                    def solve(spec, **options):
                        variables = spec.variable_ids()
                        return ILPSolveResult(
                            "optimal", sum(spec.objective.values()),
                            dict.fromkeys(variables, 1.0), variables, 0.0,
                        )

                    session.solve.side_effect = solve
                    planner = HILPPlanner(
                        expansion_rounds=rounds,
                        frontier_heuristic=UtilityHeuristic("test", callback, upper_bound=True),
                        terminal_heuristic=True, risk_budget=1.0,
                        solver_time_limit_ms=None,
                    )
                    decision = planner._choose_action(runtime, interface, evaluator, session)
                    self.assertEqual(callback.call_count, calls)
                    self.assertEqual(session.solve.call_count, solves)
                    objectives = [tuple(call.args[0].objective.values()) for call in session.solve.call_args_list]
                    self.assertEqual(objectives, [(10.0,)] + [(1.0, 10.0)] * (solves - 1))
                    self.assertEqual(decision.value, 11.0)
                    self.assertEqual(decision.complete, rounds is None)
                    self.assertEqual(decision.policy.duration_complete, rounds is None)
                    self.assertEqual(decision.policy.achieved_utility, 11.0 if rounds is None else None)
                finally:
                    problem.env.close()

    def test_warm_start_preserves_binary_values_and_undefined(self):
        """Normalize starts once and clear stale values without binarizing UNDEFINED.

        / 初始值只二值化一次；清除旧值时保持 UNDEFINED，不将其误转为 0 或 1。
        """
        session = GurobiILPSession()
        undefined = 1e101
        session._grb = SimpleNamespace(UNDEFINED=undefined)
        session._variables = {
            name: SimpleNamespace(Start=undefined) for name in ("x_q", "x_qa", "x_qb")
        }
        session._apply_warm_start({"x_q": .9, "x_qa": .1})
        self.assertEqual(session._variables["x_q"].Start, 1.0)
        self.assertEqual(session._variables["x_qa"].Start, 0.0)
        self.assertEqual(session._variables["x_qb"].Start, undefined)
        session._apply_warm_start({"x_qa": 1.0})
        self.assertEqual(session._variables["x_q"].Start, undefined)
        self.assertEqual(session._variables["x_qa"].Start, 1.0)
        self.assertEqual(session._variables["x_qb"].Start, undefined)
        self.assertEqual(session._start_values, {"x_qa": 1.0})

    def test_suppressed_gurobi_errors_are_logged(self):
        """Keep fallback behavior while reporting suppressed solver exceptions.

        / 记录被容错处理的求解器异常，同时保持原来的回退行为。
        """
        session = GurobiILPSession()
        session._model = Mock(dispose=Mock(side_effect=RuntimeError("dispose failed")))
        with self.assertLogs("darp.ilp.gurobi", level="WARNING") as logs:
            session.__del__()
        self.assertIn("finalization", logs.output[0])
        self.assertIsInstance(logs.records[0].exc_info[1], RuntimeError)
        self.assertIsNone(session._model)

        with self.assertLogs("darp.ilp.gurobi", level="WARNING") as logs:
            _set_start(object(), 1.0)
        self.assertIn("MIP start to 1.0", logs.output[0])
        self.assertIsInstance(logs.records[0].exc_info[1], AttributeError)

        with self.assertLogs("darp.ilp.gurobi", level="DEBUG") as logs:
            self.assertIsNone(_optional_attr(object(), "ObjVal"))
        self.assertEqual(logs.records[0].levelname, "DEBUG")
        self.assertIn("ObjVal", logs.output[0])
        self.assertIsInstance(logs.records[0].exc_info[1], AttributeError)

        class AttributeOnlyVariable:
            """Simulate an adapter requiring setAttr. / 模拟必须用 setAttr 的适配器。"""

            __slots__ = ("setAttr",)

            def __init__(self):
                self.setAttr = Mock()

        variable = AttributeOnlyVariable()
        with self.assertLogs("darp.ilp.gurobi", level="DEBUG") as logs:
            _set_objective_coefficient(variable, 2.0)
        variable.setAttr.assert_called_once_with("Obj", 2.0)
        self.assertIn("fallback succeeded", logs.output[0])
        self.assertIsInstance(logs.records[0].exc_info[1], AttributeError)

        with self.assertLogs("darp.ilp.gurobi", level="DEBUG") as logs:
            self.assertIsNone(_optional_float("invalid"))
            self.assertIsNone(_optional_float(float("nan")))
        self.assertIsInstance(logs.records[0].exc_info[1], ValueError)
        self.assertIsNone(logs.records[1].exc_info)

        with self.assertLogs("darp.ilp.gurobi", level="WARNING") as logs:
            _set_param(object(), "TimeLimit", 1.0)
        self.assertIn("TimeLimit=1.0", logs.output[0])

        with self.assertNoLogs("darp.ilp.gurobi", level="DEBUG"):
            variable = SimpleNamespace()
            _set_start(variable, 1.0)
            self.assertEqual(_optional_attr(variable, "Start"), 1.0)
            _set_objective_coefficient(variable, 2.0)
            self.assertEqual(variable.Obj, 2.0)
            model = SimpleNamespace(Params=SimpleNamespace(TimeLimit=0.0))
            _set_param(model, "TimeLimit", 1.0)
            self.assertEqual(model.Params.TimeLimit, 1.0)
            self.assertEqual(_optional_float(2.0), 2.0)
            self.assertIsNone(_optional_float(None))
            session.close()

    def test_lazy_frontier_has_no_executable_expansion(self):
        with TemporaryDirectory() as temporary:
            problem = _terminal_problem(Path(temporary))
            try:
                runtime, interface, evaluator = _terminal_search(problem)
                item, = initialize_root_frontier(runtime, interface)
                with patch("darp.planning.hilp.expand_frontier_item") as expand:
                    record = _frontier_leaf_record(
                        item, interface, evaluator,
                        heuristic=UtilityHeuristic("test", lambda _: 10.0),
                        terminal_heuristic=True,
                    )
                expand.assert_not_called()
                self.assertIsNone(record.policy_expansion)
                tree = build_partial_tree_ilp(
                    runtime=runtime, interface=interface,
                    expanded_records=[], frontier_records=[record], risk_budget=1.0,
                )
                self.assertEqual(tree.variable_expansions, {})
                policy = _only_policy(tree)
                self.assertEqual(policy.nodes[0].assignment, {"slow": False})
                self.assertFalse(policy.duration_complete)
                self.assertIsNone(policy.achieved_utility)
                self.assertIsNone(policy.active_constraint_value)
                self.assertIsNone(policy.feasible)
            finally:
                problem.env.close()

    def test_round_caps_and_timeout_keep_the_last_solved_policy(self):
        for limit, failure, expected_solves, expected_expanded in (
            (0, None, 1, 0),
            (1, None, 2, 1),
            (None, "exception", 2, 0),
            (None, "no-incumbent", 2, 0),
            (None, "infeasible", 2, 0),
            (1, "nonoptimal", 2, 1),
            (None, "no-selected-frontier", 1, 0),
        ):
            with self.subTest(limit=limit, failure=failure), TemporaryDirectory() as temporary:
                problem = _terminal_problem(Path(temporary))
                try:
                    runtime, interface, evaluator = _terminal_search(problem)
                    session = Mock(last_model_update_ms=0.0, last_optimize_ms=0.0)

                    def solve(spec, **options):
                        status = "optimal"
                        if session.solve.call_count == 2:
                            if failure == "exception":
                                raise TimeoutError("test refinement timeout")
                            if failure in ("no-incumbent", "infeasible"):
                                status = "time_limit" if failure == "no-incumbent" else "infeasible"
                                return ILPSolveResult(status, None, {}, (), 0.0)
                            if failure == "nonoptimal":
                                status = "interrupted"
                        variables = spec.variable_ids()
                        return ILPSolveResult(
                            status, sum(spec.objective.values()),
                            dict.fromkeys(variables, 1.0), variables, 0.0,
                        )

                    session.solve.side_effect = solve
                    session.__enter__ = Mock(return_value=session)
                    session.__exit__ = Mock(return_value=False)
                    planner = HILPPlanner(
                        expansion_rounds=limit,
                        frontier_heuristic=UtilityHeuristic("test", lambda _: 10.0, upper_bound=True),
                        terminal_heuristic=True, risk_budget=1.0,
                        solver_time_limit_ms=None,
                    )
                    with patch("darp.planning.hilp.GurobiILPSession", return_value=session), patch.object(
                        planner, "_selected_frontier", wraps=planner._selected_frontier,
                    ) as selected, patch("darp.planning.hilp.logger") as logger:
                        if failure == "no-selected-frontier":
                            selected.return_value = ()
                        if failure == "infeasible":
                            with self.assertRaisesRegex(RuntimeError, "status=infeasible"):
                                planner.choose_action(runtime, interface, evaluator)
                            self.assertEqual(session.solve.call_count, expected_solves)
                            continue
                        decision = planner.choose_action(runtime, interface, evaluator)
                    self.assertEqual(session.solve.call_count, expected_solves)
                    self.assertEqual(decision.timing["expanded_nodes"], expected_expanded)
                    self.assertEqual(decision.timing["expansion_rounds"], expected_expanded)
                    self.assertEqual(decision.timing["ilp_variables"], expected_expanded + 1)
                    self.assertEqual(decision.timing["solver_time_limit_hit"], float(failure in ("exception", "no-incumbent")))
                    self.assertEqual(len(decision.policy.nodes), expected_expanded + 1)
                    self.assertEqual(decision.policy.solver_status, "interrupted" if failure == "nonoptimal" else "optimal")
                    self.assertFalse(decision.complete)
                    self.assertFalse(decision.policy.duration_complete)
                    self.assertEqual(decision.value_kind, "heuristic_objective")
                    if failure in ("exception", "no-incumbent"):
                        self.assertEqual(decision.value, 10.0)
                        logger.warning.assert_called_once()
                        self.assertIn("last solved policy", logger.warning.call_args.args[0])
                        self.assertEqual(bool(logger.warning.call_args.kwargs.get("exc_info")), failure == "exception")
                    else:
                        logger.warning.assert_not_called()
                    if failure == "no-selected-frontier":
                        self.assertEqual(decision.timing["frontier_nodes"], 1)
                        self.assertEqual(decision.timing["frontier_refinement_exhausted"], 1)
                finally:
                    problem.env.close()

    def test_partial_solve_keeps_session_delta_and_warm_start(self):
        """Forward the explicit update and incumbent without rebuilding the model.

        / 直接传递显式差量和上一轮解，不关闭或重建模型。
        """
        spec = ILPModelSpec(
            "incremental-test", (ILPVariable("x_q"),), {"x_q": 0.0},
            (ILPLinearConstraint("root_action", {"x_q": 1.0}, "==", 1.0),),
        )
        tree = PolicyTreeILP(spec, {}, ("x_q",))
        model_delta = ILPModelDelta(spec.variables, spec.objective, spec.constraints)
        builder = Mock()
        builder.snapshot.return_value = tree, model_delta
        session = Mock()
        incumbent = {"x_q": 1.0}
        result = ILPSolveResult("optimal", 0.0, incumbent, ("x_q",), 0.0)
        session.solve.return_value = result

        actual_tree, actual_result = HILPPlanner()._solve_partial_policy_ilp(
            None, None, ilp_builder=builder, expanded_records=[], frontier=[],
            frontier_records={}, ilp_session=session, warm_start=incumbent,
            solver_deadline=None,
        )

        self.assertIs(actual_tree.spec, tree.spec)
        self.assertIs(actual_result, result)
        session.close.assert_not_called()
        session.solve.assert_called_once_with(
            tree.spec, delta=model_delta, time_limit_ms=None, warm_start=incumbent,
        )


class ExperimentLoggingTests(unittest.TestCase):
    """Verify experiment fallbacks log their context without changing results.

    / 验证实验回退记录上下文，但不改变结果格式和后续执行。
    """

    def test_table1_trial_error_is_logged_and_the_matrix_continues(self):
        experiment = import_module("experiments.DARP-table1-grid.run")
        with TemporaryDirectory() as temporary:
            output = Path(temporary) / "trials.csv"
            args = experiment._parser().parse_args([
                "--models", "F", "--horizons", "3", "--deltas", "0.1",
                "--planners", "hilp", "--trials", "2", "--episodes", "1",
                "--output", str(output),
            ])
            successful = dict.fromkeys(experiment.FIELDS, "")
            successful.update(model="F", horizon=3, delta=.1, planner="hilp", trial=2, seed=2024, status="ok")
            with patch.object(experiment, "_parser") as parser, patch.object(
                experiment, "_run_trial", side_effect=[RuntimeError("trial failed"), successful],
            ) as run_trial, patch.object(experiment, "_write_markdown"), patch("builtins.print"), self.assertLogs(
                experiment.__name__, level="ERROR",
            ) as logs:
                parser.return_value.parse_args.return_value = args
                self.assertEqual(experiment.main(), 1)
            self.assertEqual(run_trial.call_count, 2)
            self.assertEqual(len(logs.records), 1)
            self.assertIsInstance(logs.records[0].exc_info[1], RuntimeError)
            for context in ("model=F", "horizon=3", "delta=0.1", "planner=hilp", "trial=1", "seed=2023"):
                self.assertIn(context, logs.output[0])
            with output.open(newline="", encoding="utf-8") as stream:
                reader = csv.DictReader(stream)
                self.assertEqual(tuple(reader.fieldnames), experiment.FIELDS)
                rows = list(reader)
            self.assertEqual([row["status"] for row in rows], ["error", "ok"])
            self.assertEqual(rows[0]["error"], "RuntimeError: trial failed")

    def test_raostar_cache_fallback_logs_and_verifies_destination(self):
        runner = import_module("experiments.DARP-vs-RAOstar-grid.raostar_runner")
        with TemporaryDirectory() as temporary:
            cache = Path(temporary)
            source = runner.RAOSTAR
            destination = cache / f"{source.name}-{source.commit[:12]}"

            def concurrent_checkout(candidate, target):
                target.mkdir()
                raise FileExistsError("another checkout already exists")

            with patch.object(runner, "_command"), patch.object(runner, "_git"), patch.object(
                runner, "_verify", side_effect=lambda path, source: path,
            ) as verify, patch.object(Path, "rename", autospec=True, side_effect=concurrent_checkout), self.assertLogs(
                runner.__name__, level="DEBUG",
            ) as logs:
                self.assertEqual(runner._resolve(source, None, cache), destination)
            self.assertEqual(verify.call_count, 2)
            verify.assert_called_with(destination, source)
            self.assertIn(str(destination), logs.output[0])
            self.assertIsInstance(logs.records[0].exc_info[1], FileExistsError)


if __name__ == "__main__":
    unittest.main()
