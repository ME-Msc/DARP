"""Experiment isolation and report semantics. / 实验隔离与对比表语义测试。"""

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments import compare, run


class ExperimentLayoutTests(unittest.TestCase):
    def test_changed_input_changes_comparison_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            domain, instance = Path(tmp) / "domain.rddl", Path(tmp) / "instance.rddl"
            domain.write_text("duration = 1;")
            instance.write_text("horizon = 3;")
            before = run.input_digest(domain, instance)
            domain.write_text("duration = 2;")
            self.assertNotEqual(before, run.input_digest(domain, instance))

    def test_report_directory_and_no_false_raostar_cost_comparison(self):
        rows = []
        for method, cost in (("HILP", 10), ("pruningF", 11), ("pruningEF", 12), ("RAOstar", 13)):
            rows.append(dict(case="grid-5x5-h3-d1-r1-b0.1", input_digest="a", instance="fixed-duration/input.rddl",
                             algorithm=method, alpha="1", **{"lambda": ".5"}, trial="1", seed="0",
                             status="ok", objective=str(cost), risk=".1", time_s="1", ilp_variables="10", batch=method))
        with tempfile.TemporaryDirectory() as tmp, patch.object(compare, "ROOT", Path(tmp)):
            configs = {r["batch"]: {} for r in rows}
            tex = compare.write_report(rows[:3], configs, "grid", "test", pdf=False)
            self.assertEqual(tex.parent.name, "grid-HILP-pruningF-pruningEF")
            self.assertIn("10.00", tex.read_text())
            self.assertNotIn(r"\underline{", tex.read_text())  # Unknown timing metadata.
            with self.assertRaises(FileExistsError):
                compare.write_report(rows[:3], configs, "grid", "test", pdf=False)
            tex = compare.write_report([rows[0], rows[3]], configs, "grid", "rao", pdf=False)
            self.assertNotIn("30.00", tex.read_text())
            self.assertIn("native boundary objective", tex.read_text())

    def test_failures_are_not_dropped_from_medians(self):
        common = dict(case="case", input_digest="a", instance="fixed/case.rddl", algorithm="HILP",
                      alpha="1", **{"lambda": "1"}, seed="0", objective="10", risk="0", time_s="2",
                      ilp_variables="10", batch="test")
        rows = [dict(common, trial="1", status="ok"), dict(common, trial="2", status="timeout")]
        with tempfile.TemporaryDirectory() as tmp, patch.object(compare, "ROOT", Path(tmp)):
            tex = compare.write_report(rows, {"test": {}}, "grid", "failed", pdf=False)
            self.assertIn("-- & -- & -- & -- & --", tex.read_text())
            self.assertIn("1 unsuccessful", tex.read_text())

    def test_resume_checks_inputs_and_skips_recorded_trials(self):
        source = run.ROOT / "benchmarks/grid/fixed-duration/grid-5x5-h3-d1-r1-b0.1.rddl"
        with tempfile.TemporaryDirectory() as tmp, patch.object(run, "ROOT", Path(tmp)), patch.object(
            run, "run_trial", return_value={"status": "ok", "time_s": 1.}
        ) as trial:
            kwargs = dict(scene="grid", name="test", instances=[source], algorithms=["HILP"], episodes=0)
            output = run.run_batch(**kwargs)
            run.run_batch(**kwargs, resume=True)
            self.assertEqual(trial.call_count, 1)
            with self.assertRaises(ValueError):
                run.run_batch(**{**kwargs, "seed": 99}, resume=True)
            with (output / "results.csv").open(newline="") as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 1)

    def test_all_migrated_policy_references_exist(self):
        count = 0
        for path in (run.ROOT / "experiments/grid/runs").glob("*/results.csv"):
            if path.parent.name.startswith("validation-"):
                continue
            with path.open(newline="") as stream:
                for row in csv.DictReader(stream):
                    self.assertTrue((run.ROOT / row["instance"]).is_file())
                    if row.get("result_file"):
                        self.assertTrue((path.parent / row["result_file"]).is_file())
                        count += 1
        self.assertGreater(count, 0)


if __name__ == "__main__":
    unittest.main()
