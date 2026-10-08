"""Rank structural and risk regression checks without Gurobi.

/ 不依赖 Gurobi 的 Rank 树结构、效用和风险语义回归测试。
"""

import unittest
from dataclasses import replace
from itertools import product
from types import SimpleNamespace
from unittest.mock import Mock, patch

from darp.ilp.model import (
    ILPLinearConstraint,
    ILPModelDelta,
    ILPModelSpec,
    ILPSolveResult,
    ILPVariable,
)
from darp.planning.hilp import HILPPlanner
from darp.planning.ilp_tree import PolicyTreeILP
from darp.planning.policy import extract_conditional_policy
from darp.planning.rank import restrict_rank_candidates, validate_rank


def _tree(objective, roots, flows=(), *, risk=None, budget=1.0, deadends=()):
    """Create the standard history-tree rows. / 构造标准动作历史树约束。"""
    rows = [ILPLinearConstraint("root_action", dict.fromkeys(roots, 1.0), "==", 1.0)]
    for index, (parent, children) in enumerate(flows):
        rows.append(ILPLinearConstraint(f"flow_{index}", {parent: -1.0, **dict.fromkeys(children, 1.0)}, "==", 0.0))
    for q in deadends:
        rows.append(ILPLinearConstraint(f"deadend_{q}", {q: 1.0}, "==", 0.0))
    if risk is not None:
        rows.append(ILPLinearConstraint("risk_budget", risk, "<=", budget))
    return ILPModelSpec("test_tree", tuple(ILPVariable(q) for q in objective), objective, tuple(rows))


def _feasible(model, assignment):
    """Check tiny models independently of pruning. / 独立于剪枝实现检查微型模型约束。"""
    for row in model.constraints:
        lhs = sum(value * assignment.get(q, 0.0) for q, value in row.coefficients.items())
        if row.sense == "==" and abs(lhs - row.rhs) > 1e-9:
            return False
        if row.sense == "<=" and lhs > row.rhs + 1e-9:
            return False
    return True


class RankTests(unittest.TestCase):
    """Check physical variable removal and full observation coverage.

    / 验证变量真正从模型省略，同时保证所有必需观测分支完整。
    """

    def test_lambda_one_is_identity(self):
        spec = _tree({"q": 1.0}, ("q",), risk={"q": 0.9}, budget=0.1)
        self.assertIs(restrict_rank_candidates(spec, alpha=0.5, lambda_=1.0), spec)

    def test_parameter_validation(self):
        for alpha in (-0.5, float("nan"), float("inf"), True):
            with self.subTest(alpha=alpha), self.assertRaises(ValueError):
                validate_rank(alpha, 0.5)
        for divisor in (0.0, -0.1, 1.1, float("nan"), float("inf"), True):
            with self.subTest(divisor=divisor), self.assertRaises(ValueError):
                validate_rank(1.0, divisor)
        validate_rank(0.0, 1.0)
        validate_rank(100.0, 0.2)

    def test_expanded_utility_includes_tail_unordered_variables(self):
        spec = _tree({"child": 99.0, "frontier": 20.0, "expanded": 1.0}, ("expanded", "frontier"), (("expanded", ("child",)),))
        reduced = restrict_rank_candidates(spec, alpha=0.0, lambda_=0.5)
        self.assertEqual(set(reduced.variable_ids()), {"expanded", "child"})
        self.assertEqual(reduced.objective, {"expanded": 1.0, "child": 99.0})
        self.assertEqual(len(spec.variables), 3)

    def test_each_observation_retains_action_and_risk_row(self):
        spec = _tree(
            {"q": 1.0, "a": 3.0, "b": 2.0, "c": 5.0, "d": 4.0}, ("q",),
            (("q", ("a", "b")), ("q", ("c", "d"))),
            risk={"q": 0.01, "a": 0.02, "b": 0.02, "c": 0.03, "d": 0.03}, budget=0.3,
        )
        reduced = restrict_rank_candidates(
            spec, alpha=1.0, lambda_=0.5,
            frontier_variable_ids={"a", "b", "c", "d"},
        )
        # F contributes its global top half (c,d); structural closure adds a
        # for the other observation branch. / F 全局前二名为 c,d，另一观测分支补全 a。
        self.assertEqual(set(reduced.variable_ids()), {"q", "a", "c", "d"})
        self.assertEqual(reduced.constraints[1].coefficients, {"q": -1.0, "a": 1.0})
        self.assertEqual(reduced.constraints[2].coefficients, {"q": -1.0, "c": 1.0, "d": 1.0})
        self.assertEqual(reduced.constraints[-1].rhs, 0.3)
        self.assertEqual(reduced.constraints[-1].coefficients, {"q": 0.01, "a": 0.02, "c": 0.03, "d": 0.03})

    def test_observation_scores_sum_not_max(self):
        spec = _tree({"q": 0.0, "alternative": 8.0, "a": 5.0, "b": 5.0}, ("q", "alternative"), (("q", ("a",)), ("q", ("b",))))
        reduced = restrict_rank_candidates(spec, alpha=0.0, lambda_=0.25)
        self.assertEqual(set(reduced.variable_ids()), {"q", "a", "b"})

    def test_prefix_risk_filter_also_overrides_warm_start(self):
        spec = _tree({"q": 1.0, "unsafe": 100.0, "safe": 2.0}, ("q",), (("q", ("unsafe", "safe")),), risk={"q": 0.1, "unsafe": 0.11, "safe": 0.05}, budget=0.2)
        reduced = restrict_rank_candidates(spec, alpha=0.0, lambda_=0.5, warm_start={"q": 1.0, "unsafe": 1.0})
        self.assertEqual(set(reduced.variable_ids()), {"q", "safe"})

    def test_prefix_risk_uses_solver_tolerance(self):
        spec = _tree({"q": 1.0, "other": 0.0}, ("q", "other"), risk={"q": 0.2000005}, budget=0.2)
        self.assertEqual(restrict_rank_candidates(spec, alpha=0.0, lambda_=0.5).variable_ids(), ("q",))

    def test_dead_end_child_propagates_to_ancestor(self):
        spec = _tree({"q": 100.0, "good": 1.0, "bad": 100.0}, ("q", "good"), (("q", ("bad",)),), deadends=("bad",))
        reduced = restrict_rank_candidates(spec, alpha=0.0, lambda_=0.5)
        self.assertEqual(reduced.variable_ids(), ("good",))
        self.assertEqual(tuple(row.name for row in reduced.constraints), ("root_action",))

    def test_unselected_parent_never_leaves_orphan_descendants(self):
        spec = _tree({"q": -100.0, "good": 1.0, "child": 50.0}, ("q", "good"), (("q", ("child",)),))
        reduced = restrict_rank_candidates(spec, alpha=0.0, lambda_=0.5)
        # Ranking is approximate; whichever branch remains must be connected.
        # 评分不保证最优；无论保留哪个分支，都不能留下孤立后代。
        self.assertEqual("child" in reduced.variable_ids(), "q" in reduced.variable_ids())

    def test_no_risk_row_and_alpha_larger_than_one(self):
        spec = _tree({"q": 2.0, "other": 1.0}, ("q", "other"))
        self.assertEqual(restrict_rank_candidates(spec, alpha=10.0, lambda_=0.5).variable_ids(), ("q",))
        risk_spec = _tree({"q": 2.0, "other": 1.0}, ("q", "other"), risk={"q": 0.5})
        self.assertEqual(restrict_rank_candidates(risk_spec, alpha=10.0, lambda_=0.5).variable_ids(), ("other",))

    def test_empty_candidates_keep_infeasible_root_row(self):
        spec = _tree({"q": 1.0}, ("q",), risk={"q": 0.2}, budget=0.1)
        reduced = restrict_rank_candidates(spec, alpha=1.0, lambda_=0.5)
        self.assertEqual(reduced.variables, ())
        self.assertEqual(reduced.constraints[0], ILPLinearConstraint("root_action", {}, "==", 1.0))
        self.assertEqual(reduced.constraints[1], ILPLinearConstraint("risk_budget", {}, "<=", 0.1))

    def test_negative_residual_budget_is_not_lost(self):
        spec = _tree({"q": 1.0}, ("q",), risk={}, budget=-0.1)
        reduced = restrict_rank_candidates(spec, alpha=0.0, lambda_=0.5)
        self.assertEqual(reduced.variables, ())
        self.assertEqual(reduced.constraints[-1].rhs, -0.1)

    def test_warm_start_retains_prior_choice_beyond_fraction(self):
        spec = _tree({"q": 3.0, "prior": 2.0, "other": 1.0}, ("q", "prior", "other"))
        reduced = restrict_rank_candidates(spec, alpha=0.0, lambda_=1/3,
                                          frontier_variable_ids=set(spec.variable_ids()),
                                          warm_start={"prior": 1.0})
        self.assertEqual(reduced.variable_ids(), ("q", "prior"))

    def test_warm_start_retains_ancestors_and_all_observation_choices(self):
        spec = _tree(
            {"best": 100.0, "prior": 0.0, "a": 3.0, "b": 2.0, "c": 5.0, "d": 4.0},
            ("best", "prior"), (("prior", ("a", "b")), ("prior", ("c", "d"))),
        )
        incumbent = {"prior": 1.0, "b": 1.0, "d": 1.0}
        reduced = restrict_rank_candidates(spec, alpha=0.0, lambda_=0.5, warm_start=incumbent)
        self.assertTrue({"prior", "b", "d"} <= set(reduced.variable_ids()))
        self.assertTrue(_feasible(reduced, incumbent))

    def test_flat_expanded_subtrees_and_frontier_use_separate_targets(self):
        spec = _tree(
            {"e1": 9.0, "e2": 8.0, "f1": 7.0, "f2": 6.0, "f3": 5.0},
            ("e1", "e2", "f1", "f2", "f3"),
        )
        reduced = restrict_rank_candidates(
            spec, alpha=0.0, lambda_=0.5,
            frontier_variable_ids={"f1", "f2", "f3"},
        )
        self.assertEqual(reduced.variable_ids(), ("e1", "f1", "f2"))

    def test_protected_frontier_prevents_ancestor_subtree_deletion(self):
        # f is protected despite its low-scoring ancestor; b and g can be omitted.
        # f 分数最高，必须保护其低分祖先 a；b 与其后代 g 可以整棵省略。
        spec = _tree({"a": -100.0, "b": 0.0, "f": 10.0, "g": 1.0},
                     ("a", "b"), (("a", ("f",)), ("b", ("g",))))
        reduced = restrict_rank_candidates(spec, alpha=1.0, lambda_=0.2,
                                          frontier_variable_ids={"f", "g"})
        self.assertEqual(set(reduced.variable_ids()), {"a", "f"})

    def test_impossible_ancestor_cannot_protect_high_score_frontier(self):
        spec = _tree({"a": 0.0, "b": 0.0, "f": 100.0, "bad": 0.0, "g": 1.0},
                     ("a", "b"), (("a", ("f",)), ("a", ("bad",)), ("b", ("g",))),
                     deadends=("bad",))
        reduced = restrict_rank_candidates(spec, alpha=1.0, lambda_=0.2,
                                          frontier_variable_ids={"f", "g"})
        self.assertEqual(set(reduced.variable_ids()), {"b", "g"})

    def test_protected_single_action_chain_survives_unattainable_target(self):
        spec = _tree({"a": 1.0, "b": 1.0, "f": 1.0}, ("a",),
                     (("a", ("b",)), ("b", ("f",))))
        self.assertIs(restrict_rank_candidates(spec, alpha=1.0, lambda_=0.01,
                                              frontier_variable_ids={"f"}), spec)

    def test_bottom_up_prunes_side_branch_but_preserves_frontier_path(self):
        """A protected subtree root may still lose an unprotected inner branch.
        / 受保护子树根不能删除，但其未保护的内部侧枝应可以删除。
        """
        spec = _tree(
            {"r": 0., "a": 0., "b": 0., "f": 10., "g": 0.},
            ("r",), (("r", ("a", "b")), ("a", ("f",)), ("b", ("g",))),
        )
        reduced = restrict_rank_candidates(spec, alpha=1., lambda_=0.3,
                                          frontier_variable_ids={"f", "g"})
        self.assertEqual(set(reduced.variable_ids()), {"r", "a", "f"})
        self.assertTrue(_feasible(reduced, {"r": 1., "a": 1., "f": 1.}))

    def test_last_observation_action_survives_unreachable_quota(self):
        """Observation branches are AND requirements, not alternatives.
        / 不同观测必须全部覆盖，不能为了达到比例删掉唯一后续。
        """
        spec = _tree(
            {"r": 0., "a": 0., "b": 0., "f": 10., "g": 0.},
            ("r",), (("r", ("a",)), ("r", ("b",)), ("a", ("f",)), ("b", ("g",))),
        )
        reduced = restrict_rank_candidates(spec, alpha=1., lambda_=0.3,
                                          frontier_variable_ids={"f", "g"})
        self.assertIs(reduced, spec)

    def test_frontier_with_children_is_rejected(self):
        spec = _tree({"a": 1., "b": 1.}, ("a",), (("a", ("b",)),))
        with self.assertRaisesRegex(ValueError, "no child-flow"):
            restrict_rank_candidates(spec, alpha=1., lambda_=0.5, frontier_variable_ids={"a"})

    def test_ceil_uses_original_group_size(self):
        spec = _tree({"a": 3.0, "b": 2.0, "c": 1.0}, ("a", "b", "c"))
        self.assertEqual(restrict_rank_candidates(spec, alpha=0.0, lambda_=0.5).variable_ids(), ("a", "b"))

    def test_unknown_constraints_are_not_silently_discarded(self):
        spec = _tree({"q": 1.0}, ("q",))
        spec = replace(spec, constraints=spec.constraints + (ILPLinearConstraint("capacity", {"q": 1.0}, "<=", 0.0),))
        with self.assertRaisesRegex(ValueError, "Unsupported p-ILP constraint"):
            restrict_rank_candidates(spec, alpha=0.0, lambda_=0.5)

    def test_invalid_tree_shapes_are_rejected(self):
        cases = (
            _tree({"q": 1.0, "orphan": 2.0}, ("q",)),
            _tree({"q": 1.0, "child": 2.0}, ("q",), (("q", ("child",)), ("q", ("child",)))),
            _tree({"root": 1.0, "q": 1.0, "child": 2.0}, ("root",), (("q", ("child",)), ("child", ("q",)))),
        )
        for spec in cases:
            with self.subTest(spec=spec), self.assertRaises(ValueError):
                restrict_rank_candidates(spec, alpha=0.0, lambda_=0.5)

    def test_undeclared_coefficients_are_rejected_even_when_zero(self):
        spec = _tree({"q": 1.0}, ("q",))
        for value in (0.0, 1.0):
            invalid = [replace(spec, objective={"q": 1.0, "unknown": value})]
            for name in ("root_action", "risk_budget", "deadend_q", "flow_q"):
                row = ILPLinearConstraint(name, {"unknown": value}, "==", 0.0)
                invalid.append(replace(spec, constraints=(row,)))
            for model in invalid:
                with self.subTest(model=model), self.assertRaisesRegex(ValueError, "undeclared"):
                    restrict_rank_candidates(model, alpha=0.0, lambda_=0.5)

    def test_duplicate_and_nonfinite_model_entries_are_rejected(self):
        spec = _tree({"q": 1.0}, ("q",))
        invalid = (
            replace(spec, variables=spec.variables * 2),
            replace(spec, constraints=spec.constraints * 2),
            replace(spec, objective={"q": float("nan")}),
            _tree({"q": 1.0}, ("q",), risk={"q": float("inf")}),
            _tree({"q": 1.0}, ("q",), risk={"q": -0.1}),
            _tree({"q": 1.0}, ("q",), risk={}, budget=float("nan")),
        )
        for model in invalid:
            with self.subTest(model=model), self.assertRaises(ValueError):
                restrict_rank_candidates(model, alpha=0.0, lambda_=0.5)

    def test_restricted_model_equals_original_with_omitted_variables_zero(self):
        spec = _tree(
            {"q": 1.0, "unused": 0.0, "a": 3.0, "b": 2.0, "c": 4.0, "d": 1.0},
            ("q", "unused"), (("q", ("a", "b")), ("q", ("c", "d"))),
            risk={"q": 0.1, "a": 0.1, "c": 0.1}, budget=0.25,
        )
        reduced = restrict_rank_candidates(spec, alpha=0.0, lambda_=0.5)

        for values in product((0.0, 1.0), repeat=len(reduced.variables)):
            assignment = dict(zip(reduced.variable_ids(), values))
            self.assertEqual(_feasible(spec, assignment), _feasible(reduced, assignment))
            self.assertEqual(
                sum(value * assignment.get(q, 0.0) for q, value in spec.objective.items()),
                sum(value * assignment[q] for q, value in reduced.objective.items()),
            )

    def test_deep_tree_does_not_use_python_recursion(self):
        ids = tuple(f"q{index}" for index in range(1500))
        spec = _tree(dict.fromkeys(reversed(ids), 1.0), (ids[0],), tuple((parent, (child,)) for parent, child in zip(ids, ids[1:])))
        reduced = restrict_rank_candidates(spec, alpha=0.0, lambda_=0.5)
        self.assertEqual(reduced.variable_ids(), spec.variable_ids())

    def test_nonuniform_and_or_tree_preserves_fixed_zero_model_and_feasible_incumbents(self):
        """Enumerate feasible policies with unequal depths and multiple observations.

        / 穷举不等深、多观测树，验证补零等价性以及同模型可行 incumbent 的保护。
        """
        spec = _tree(
            dict(zip("abcdefghij", (0., -1., 4., 2., 3., -2., 2., 1., 4., 1.))),
            ("a", "b"), (("a", ("c", "d")), ("a", ("e", "f")),
                          ("b", ("g", "h")), ("c", ("i", "j"))),
            risk={"a": .01, "b": .02, "c": .1, "d": .03, "e": .05,
                  "f": .01, "g": .2, "h": .01, "i": .1, "j": .02}, budget=.21,
        )
        frontier = set("dfghij")  # e is a completed E leaf. / e 是已完成的 E 叶节点。
        feasible = [dict(zip(spec.variable_ids(), bits))
                    for bits in product((0., 1.), repeat=len(spec.variables))
                    if _feasible(spec, dict(zip(spec.variable_ids(), bits)))]
        self.assertTrue(feasible)
        for alpha, divisor in product((0., 1., 4.), (.3, .5, .7, .9)):
            reduced = restrict_rank_candidates(spec, alpha=alpha, lambda_=divisor,
                                              frontier_variable_ids=frontier)
            for bits in product((0., 1.), repeat=len(reduced.variables)):
                assignment = dict(zip(reduced.variable_ids(), bits))
                self.assertEqual(_feasible(spec, assignment), _feasible(reduced, assignment))
                self.assertAlmostEqual(
                    sum(spec.objective[q] * x for q, x in assignment.items()),
                    sum(reduced.objective[q] * x for q, x in assignment.items()))
            for incumbent in feasible:
                protected = restrict_rank_candidates(spec, alpha=alpha, lambda_=divisor,
                                                     frontier_variable_ids=frontier,
                                                     warm_start=incumbent)
                self.assertTrue(_feasible(protected, incumbent))
                self.assertTrue({q for q, x in incumbent.items() if x} <= set(protected.variable_ids()))


class RankHILPTests(unittest.TestCase):
    """Exercise HILP integration without importing or licensing Gurobi.

    / 通过模拟求解器验证 HILP 集成，无需导入 Gurobi 或使用许可证。
    """

    def setUp(self):
        self.spec = _tree({"x_n0": 3.0, "x_n1": 2.0}, ("x_n0", "x_n1"))
        self.tree = PolicyTreeILP(
            self.spec, {q: object() for q in self.spec.variable_ids()}, self.spec.variable_ids(),
            frontier_variable_ids=self.spec.variable_ids(),
        )
        self.delta = ILPModelDelta(self.spec.variables, self.spec.objective, self.spec.constraints)
        self.builder = Mock()
        self.builder.snapshot.return_value = (self.tree, self.delta)
        self.session = Mock(last_model_update_ms=0.0, last_optimize_ms=0.0)
        self.session.solve.side_effect = lambda spec, **kwargs: self._result(spec)

    @staticmethod
    def _result(spec, status="optimal"):
        """Select the first root in a synthetic result. / 构造选中第一个根动作的模拟解。"""
        selected = spec.variable_ids()[:1] if status == "optimal" else ()
        return ILPSolveResult(
            status, sum(spec.objective[q] for q in selected) if selected else None,
            {q: float(q in selected) for q in spec.variable_ids()}, selected, 0.0,
        )

    def _solve(self, *, lambda_=0.5, deadline=None, warm_start=None):
        """Run the real partial-solve path with a fake backend. / 用模拟后端执行真实 partial-solve 流程。"""
        return HILPPlanner(rank_lambda=lambda_)._solve_partial_policy_ilp(
            None, None, ilp_builder=self.builder, expanded_records=[], frontier=[],
            frontier_records={}, ilp_session=self.session, warm_start=warm_start,
            solver_deadline=deadline,
        )

    def test_reduced_solver_keeps_full_policy_lookup_maps(self):
        tree, result = self._solve()
        self.assertEqual(tree.spec.variable_ids(), ("x_n0",))
        self.assertIs(tree.variable_items, self.tree.variable_items)
        self.assertIs(tree.variable_expansions, self.tree.variable_expansions)
        self.assertEqual(result.selected_variables, ("x_n0",))
        self.session.close.assert_called_once()
        self.assertIsNone(self.session.solve.call_args.kwargs["delta"])

    def test_default_path_keeps_original_delta_and_does_not_close(self):
        tree, _ = self._solve(lambda_=1.0)
        self.assertIs(tree.spec, self.spec)
        self.session.close.assert_not_called()
        self.assertIs(self.session.solve.call_args.kwargs["delta"], self.delta)

    def test_restricted_infeasible_retries_original_model(self):
        self.session.solve.side_effect = lambda spec, **kwargs: self._result(
            spec, "infeasible" if len(spec.variables) == 1 else "optimal",
        )
        tree, result = self._solve()
        self.assertIs(tree.spec, self.spec)
        self.assertEqual(result.status, "optimal")
        self.assertEqual(self.session.solve.call_count, 2)
        self.assertEqual(self.session.close.call_count, 2)
        self.assertTrue(all(call.kwargs["delta"] is None for call in self.session.solve.call_args_list))

    def test_joint_observation_risk_falls_back_to_feasible_original_model(self):
        spec = _tree(
            {"q": 1.0, "a": 4.0, "b": 1.0, "c": 3.0, "d": 1.0}, ("q",),
            (("q", ("a", "b")), ("q", ("c", "d"))),
            risk={"q": 0.1, "a": 0.1, "c": 0.1}, budget=0.25,
        )
        self.builder.snapshot.return_value = (
            PolicyTreeILP(spec, dict.fromkeys(spec.variable_ids()), ("q",)), self.delta,
        )

        def solve(model, **kwargs):
            feasible = [
                assignment for values in product((0.0, 1.0), repeat=len(model.variables))
                if _feasible(model, assignment := dict(zip(model.variable_ids(), values)))
            ]
            if not feasible:
                return self._result(model, "infeasible")
            utility = lambda assignment: sum(model.objective[q] * value for q, value in assignment.items())
            best = max(feasible, key=utility)
            return ILPSolveResult("optimal", utility(best), best, tuple(q for q, value in best.items() if value), 0.0)

        self.session.solve.side_effect = solve
        tree, result = self._solve()
        self.assertEqual(self.session.solve.call_args_list[0].args[0].variable_ids(), ("q", "a", "c"))
        self.assertIs(tree.spec, spec)
        self.assertEqual(self.session.solve.call_count, 2)
        self.assertEqual(result.selected_variables, ("q", "a", "d"))
        self.assertTrue(_feasible(spec, result.variable_values))

    def test_fallback_uses_remaining_global_deadline(self):
        with patch("darp.planning.hilp.perf_counter", return_value=1.0) as clock:
            def solve(spec, **kwargs):
                """Consume time on the first solve. / 模拟第一次求解耗时。"""
                if len(spec.variables) == 1:
                    clock.return_value = 1.4
                    return self._result(spec, "infeasible")
                return self._result(spec)

            self.session.solve.side_effect = solve
            self._solve(deadline=2.0)
        calls = self.session.solve.call_args_list
        self.assertAlmostEqual(calls[0].kwargs["time_limit_ms"], 1000.0)
        self.assertAlmostEqual(calls[1].kwargs["time_limit_ms"], 600.0)

    def test_expired_fallback_does_not_start_second_solver(self):
        with patch("darp.planning.hilp.perf_counter", return_value=1.0) as clock:
            def solve(spec, **kwargs):
                """Exhaust the shared deadline. / 模拟耗尽共享总时限。"""
                clock.return_value = 2.1
                return self._result(spec, "infeasible")

            self.session.solve.side_effect = solve
            with self.assertRaises(TimeoutError):
                self._solve(deadline=2.0)
        self.session.solve.assert_called_once()

    def test_warm_start_is_forwarded_and_selected_choices_retained(self):
        warm_start = {"x_n1": 1.0, "previously_omitted": 0.0}
        tree, _ = self._solve(warm_start=warm_start)
        self.assertEqual(tree.spec.variable_ids(), self.spec.variable_ids())
        self.assertIs(self.session.solve.call_args.kwargs["warm_start"], warm_start)

    def test_actual_policy_extraction_accepts_omitted_action_siblings(self):
        observation = (("seen", True),)
        items = {
            q: SimpleNamespace(
                node=SimpleNamespace(node_id=q, assignment={"move": q}),
                observation_keys=() if q == "q" else (observation,),
                action_label=q,
            )
            for q in ("q", "a", "b")
        }
        expansions = {
            "q": SimpleNamespace(
                metrics=SimpleNamespace(utility=1.0, chance_risk=0.0),
                observation_frontiers=(SimpleNamespace(
                    should_expand=True, observation=observation,
                    child_frontier=(items["a"], items["b"]),
                ),),
            ),
            "a": SimpleNamespace(
                metrics=SimpleNamespace(utility=2.0, chance_risk=0.05),
                observation_frontiers=(SimpleNamespace(
                    should_expand=False, observation=(("seen", False),), child_frontier=(),
                ),),
            ),
        }
        spec = _tree({"q": 1.0, "a": 2.0, "b": 1.0}, ("q",), (("q", ("a", "b")),), risk={"a": 0.05}, budget=0.1)
        reduced = restrict_rank_candidates(spec, alpha=0.0, lambda_=0.5)
        tree = PolicyTreeILP(
            reduced, items, ("q",), variable_expansions=expansions,
            variable_continues={"q": True, "a": False}, risk_budget=0.1,
        )
        result = ILPSolveResult("optimal", 3.0, {"q": 1.0, "a": 1.0}, ("q", "a"), 0.0)
        policy = extract_conditional_policy(tree, result)
        self.assertTrue(policy.duration_complete)
        self.assertTrue(policy.feasible)
        self.assertEqual(policy.achieved_utility, 3.0)
        self.assertEqual(policy.active_constraint_value, 0.05)
        self.assertEqual(len(policy.nodes), 2)
        self.assertFalse(extract_conditional_policy(
            replace(tree, variable_items={q: items[q] for q in ("q", "a")}), result,
        ).duration_complete)

    def test_search_certificate_depends_on_actual_candidate_restriction(self):
        items = {
            q: SimpleNamespace(
                node=SimpleNamespace(
                    node_index=index, assignment={"move": True},
                    history=SimpleNamespace(label=lambda: "move"),
                ),
                action_label="move",
            )
            for index, q in enumerate(self.spec.variable_ids())
        }
        policy = SimpleNamespace(duration_complete=True, feasible=True, achieved_utility=3.0)
        cases = ((1.0, False, True), (0.5, False, False), (0.9, False, True), (0.5, True, True))
        for lambda_, fallback, expected_complete in cases:
            with self.subTest(lambda_=lambda_, fallback=fallback):
                spec = self.spec if fallback else restrict_rank_candidates(
                    self.spec, alpha=1.0, lambda_=lambda_,
                )
                tree = replace(self.tree, spec=spec, variable_items=items, frontier_variable_ids=())
                planner = HILPPlanner(expansion_rounds=0, rank_lambda=lambda_)
                with (
                    patch("darp.planning.hilp.initialize_root_frontier", return_value=tuple(items.values())),
                    patch("darp.planning.hilp.IncrementalPartialTreeILP"),
                    patch("darp.planning.hilp.extract_conditional_policy", return_value=policy),
                    patch.object(planner, "_count_globally_expandable_frontier", return_value=0),
                    patch.object(planner, "_solve_partial_policy_ilp", return_value=(tree, self._result(spec))),
                ):
                    decision = planner._choose_action(None, None, None, self.session)
                self.assertEqual(decision.complete, expected_complete)
                self.assertTrue(decision.policy.duration_complete)
                self.assertEqual(decision.timing["rank_restricted"], float(not expected_complete))


if __name__ == "__main__":
    unittest.main()
