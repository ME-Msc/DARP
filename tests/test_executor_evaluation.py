"""Smoke checks for saved-policy execution, risk counting and duration sampling.

Run without Gurobi: ``python -m unittest discover -s tests -p test_executor_evaluation.py``.
"""

import unittest
from dataclasses import replace
from math import sqrt
from pathlib import Path
from random import Random
from statistics import fmean
from tempfile import TemporaryDirectory
from unittest.mock import patch

from darp.adapter.duration import build_duration_evaluator
from darp.adapter.kernel import RDDLKernel
from darp.adapter.loader import load_rddl
from darp.adapter.problem import PyRDDLGymProblem, RDDLLoadError
from darp.adapter.runtime import PyRDDLGymRuntime
from darp.executor import PolicyExecutor
from darp.ilp.model import ILPSolveResult
from darp.model.and_or_tree import ANDORSearchInterface
from darp.planning.heuristic import UtilityHeuristic
from darp.planning.ilp_tree import build_full_tree_ilp
from darp.planning.policy import (
    ConditionalPolicy,
    PolicyNode,
    extract_conditional_policy,
)
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


def _terminal_tree(problem, **options):
    runtime = PyRDDLGymRuntime(problem.env)
    runtime.reset(seed=0)
    view = problem.build_grounded_view()
    interface = view.build_and_or_interface(runtime)
    # One action makes the entire tree the unique policy; no optimizer needed.
    interface = ANDORSearchInterface.from_actions_and_observations(
        actions=tuple(action for action in interface.actions if action.label == "noop"),
        observation_scope=interface.observation_scope,
        kernel=interface.kernel,
    )
    evaluator = build_duration_evaluator(interface.kernel, interface.actions, horizon=runtime.horizon)
    return build_full_tree_ilp(runtime, interface, evaluator, risk_budget=view.grounded_model.risk_budget, **options)


def _only_policy(tree):
    variables = tree.spec.variable_ids()
    return extract_conditional_policy(tree, ILPSolveResult(
        status="test-single-action", objective_value=sum(tree.spec.objective.values()),
        variable_values=dict.fromkeys(variables, 1.0), selected_variables=variables, runtime_ms=0,
    ))


class ExecutorEvaluationTests(unittest.TestCase):
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
                self.assertEqual(tree.initial_chance_risk, .25)
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


if __name__ == "__main__":
    unittest.main()
