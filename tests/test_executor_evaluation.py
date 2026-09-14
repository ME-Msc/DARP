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

from darp.adapter.loader import load_rddl
from darp.executor import PolicyExecutor
from darp.planning.policy import ConditionalPolicy, PolicyNode
from darp.solve import DARPResult

ROOT = Path(__file__).resolve().parents[1]
TABLE1 = ROOT / "experiments" / "DARP-table1-grid"
GRID = ROOT / "experiments" / "DARP-vs-RAOstar-grid" / "rddl"


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


def _environment(directory: Path, duration: str, horizon: int, *, initial_risk=False):
    domain = directory / "domain.rddl"
    instance = directory / "instance.rddl"
    risk = directory / "risk.json"
    domain.write_text(
        """domain executor_test {
    requirements = { partially-observed };
    pvariables {
        hidden : { state-fluent, bool, default = false };
        unsafe : { state-fluent, bool, default = false };
        slow : { action-fluent, bool, default = false };
        seen : { observ-fluent, bool };
    };
    cpfs {
        hidden' = Bernoulli(0.5);
        unsafe' = true;
        seen = false;
    };
    reward = -1;
    duration = """ + duration + ";\n}\n",
        encoding="utf-8",
    )
    instance.write_text(
        """non-fluents test_nf { domain = executor_test; }
instance test_instance {
    domain = executor_test;
    non-fluents = test_nf;
    init-state { unsafe = """ + ("true" if initial_risk else "false") + "; };\n"
        + "    max-nondef-actions = 1;\n"
        + f"    horizon = {horizon};\n    discount = 1.0;\n}}\n",
        encoding="utf-8",
    )
    risk.write_text('{"budget": 1.0, "risky_states": [{"unsafe": true}]}\n', encoding="utf-8")
    return load_rddl(domain, instance).env, risk


class ExecutorEvaluationTests(unittest.TestCase):
    def test_saved_f_e_s_policies_execute(self):
        for model in ("f", "e", "s"):
            with self.subTest(model=model):
                result = DARPResult.load(
                    TABLE1 / "output" / "results" / "smoke-raw"
                    / f"darp-{model}-h3-d0p1-full-ilp-trial01.json"
                )
                env = load_rddl(GRID / "domain.rddl", TABLE1 / "rddl" / f"instance_{model}_h3.rddl").env
                try:
                    agent = PolicyExecutor(result.decision.policy)
                    statistics = agent.evaluate(
                        env, episodes=10, seed=7, risk_path=GRID / "risk.json",
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
                env, risk = _environment(Path(temporary), duration, 2, initial_risk=initial_risk)
                try:
                    policy = _policy((False, True))
                    # Display labels need not uniquely identify RDDL assignments.
                    policy = replace(policy, nodes=tuple(replace(node, action_label="move") for node in policy.nodes))
                    agent = PolicyExecutor(policy)
                    statistics = agent.evaluate(env, episodes=10, seed=19, risk_path=risk)
                    self.assertEqual(statistics["risk_rate"], 1.0)
                    self.assertEqual(statistics["mean"], -2.0)
                    self.assertTrue(agent.at_leaf)
                    if expected is not None:
                        self.assertAlmostEqual(statistics["physical_duration_mean"], expected)
                    else:
                        self.assertTrue(2 <= statistics["physical_duration_mean"] <= 4)
                finally:
                    env.close()


if __name__ == "__main__":
    unittest.main()
