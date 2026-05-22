"""Tests for Phase 8 Gurobi-backed ILP planning."""

from __future__ import annotations

from itertools import product
import sys
from types import SimpleNamespace
from typing import Any, Mapping

from darp.ilp.gurobi import GurobiILPSolver
from darp.ilp.model import ILPLinearConstraint, ILPModelSpec, ILPVariable
from darp.model.and_or_tree import ANDORSearchInterface, ActionChoice, ObservationScope
from darp.model.duration_sidecar import build_duration_sidecar
from darp.planning.full_ilp import FullILPPlanner
from darp.planning.hilp import HILPPlanner
from darp.planning.ilp_tree import build_generated_full_tree_ilp
from darp.planning.rollout import action_label
from test_phase7_search import _phase7_inputs


def test_gurobi_solver_selects_best_binary_variable(monkeypatch):
    """Check the Gurobi adapter solves a tiny binary ILP. / 检查 Gurobi adapter 能解小型二元 ILP。"""
    _install_fake_gurobi(monkeypatch)
    spec = ILPModelSpec(
        name="tiny_binary",
        variables=(
            ILPVariable("low", "low"),
            ILPVariable("high", "high"),
        ),
        objective={"low": 1.0, "high": 3.0},
        constraints=(
            ILPLinearConstraint(
                name="choose_one",
                coefficients={"low": 1.0, "high": 1.0},
                sense="==",
                rhs=1.0,
            ),
        ),
    )

    result = GurobiILPSolver().solve(spec)

    assert result.status == "optimal"
    assert result.objective_value == 3.0
    assert result.selected_variables == ("high",)


def test_gurobi_solver_reports_infeasible_status(monkeypatch):
    """Check infeasible ILPs return status without crashing. / 检查 infeasible ILP 能返回状态而不崩溃。"""
    _install_fake_gurobi(monkeypatch)
    spec = ILPModelSpec(
        name="infeasible_binary",
        variables=(ILPVariable("x", "x"),),
        objective={"x": 1.0},
        constraints=(
            ILPLinearConstraint(name="must_zero", coefficients={"x": 1.0}, sense="==", rhs=0.0),
            ILPLinearConstraint(name="must_one", coefficients={"x": 1.0}, sense="==", rhs=1.0),
        ),
    )

    result = GurobiILPSolver().solve(spec)

    assert result.status == "infeasible"
    assert result.objective_value is None
    assert result.selected_variables == ()


def test_generated_full_tree_ilp_contains_root_flow_and_risk_rows():
    """Check generated policy-tree ILP rows before solving. / 检查生成式 policy-tree ILP 的 root/flow/risk 行。"""
    runtime, interface, duration = _phase7_inputs()

    tree_ilp = build_generated_full_tree_ilp(
        runtime,
        interface,
        duration,
        depth=2,
        risk_budget=0.0,
    )

    names = [constraint.name for constraint in tree_ilp.spec.constraints]
    assert names[0] == "root_action"
    assert "risk_budget" in names
    assert any(name.startswith("flow_") for name in names)
    assert len(tree_ilp.root_variable_ids) == len(interface.actions)
    assert set(tree_ilp.root_variable_ids).issubset(set(tree_ilp.spec.variable_ids()))


def test_full_ilp_planner_uses_gurobi_when_available(monkeypatch):
    """Check full-tree planning uses Gurobi-selected root variables. / 检查 full-tree planner 使用 Gurobi 选择根变量。"""
    _install_fake_gurobi(monkeypatch)
    runtime, interface, duration = _two_action_inputs()
    planner = FullILPPlanner(lookahead_depth=1, require_gurobi=True)

    decision = planner.choose_action(runtime, interface, duration, remaining_depth=runtime.horizon)

    assert decision.label == "go"
    assert decision.complete is True
    assert planner.last_ilp_result is not None
    assert planner.last_ilp_result.is_optimal


def test_hilp_frontier_selection_uses_gurobi_when_available(monkeypatch):
    """Check HILP calls the Gurobi p-ILP frontier selector. / 检查 HILP 调用 Gurobi p-ILP frontier selector。"""
    _install_fake_gurobi(monkeypatch)
    runtime, interface, duration = _two_action_inputs()
    planner = HILPPlanner(lookahead_depth=1, max_iterations=1, frontier_width=1, require_gurobi=True)

    decision = planner.choose_action(runtime, interface, duration, remaining_depth=runtime.horizon)

    assert decision.label == "go"
    assert planner.last_stats is not None
    assert planner.last_stats.used_gurobi is True


def _two_action_inputs():
    """Build a tiny deterministic problem where action `go` dominates. / 构建 `go` 动作明显更优的小问题。"""
    runtime = _TwoActionRuntime()
    interface = ANDORSearchInterface.from_actions_and_observations(
        actions=tuple(
            ActionChoice(label=action_label(action), assignment=action)
            for action in runtime.action_candidates()
        ),
        observation_scope=ObservationScope(mode="mdp-state", variables=("at_goal",)),
    )
    duration = build_duration_sidecar({"kind": "fixed", "default": 1}).evaluator(horizon=runtime.horizon)
    return runtime, interface, duration


class _TwoActionRuntime:
    """Small runtime with one rewarding boolean action. / 只有一个高 reward 布尔动作的小 runtime。"""

    horizon = 1
    discount = 1.0

    def __init__(self, at_goal: bool = False) -> None:
        """Initialize the runtime state. / 初始化 runtime state。"""
        self.at_goal = at_goal
        self.state = {"at_goal": at_goal}

    def clone(self) -> "_TwoActionRuntime":
        """Return an isolated runtime copy. / 返回隔离 runtime 副本。"""
        return _TwoActionRuntime(self.at_goal)

    def action_candidates(self) -> tuple[dict[str, bool], ...]:
        """Return noop and go actions. / 返回 noop 与 go 动作。"""
        return ({"go": False}, {"go": True})

    def step(self, action: Mapping[str, Any]):
        """Apply one transition and reward. / 执行一次转移和 reward。"""
        reward = 5.0 if bool(action.get("go")) else 0.0
        self.at_goal = bool(action.get("go")) or self.at_goal
        self.state = {"at_goal": self.at_goal}
        return dict(self.state), reward, False, False, {}


class _FakeGRB:
    """Minimal constants used by the fake gurobipy module. / fake gurobipy 使用的最小常量。"""

    BINARY = "BINARY"
    MAXIMIZE = 1
    MINIMIZE = -1
    OPTIMAL = 2
    INFEASIBLE = 3
    INF_OR_UNBD = 4
    UNBOUNDED = 5
    TIME_LIMIT = 9
    INTERRUPTED = 11


class _FakeVar:
    """Tiny fake Gurobi variable. / 极简 fake Gurobi 变量。"""

    def __init__(self, name: str) -> None:
        """Store a variable name and solution value. / 保存变量名和解值。"""
        self.VarName = name
        self.X = 0.0


class _FakeConstraint:
    """Tiny fake linear constraint. / 极简 fake 线性约束。"""

    def __init__(self, expr: "_FakeLinExpr", sense: str, rhs: float) -> None:
        """Store the expression, sense, and right-hand side. / 保存表达式、方向和右端项。"""
        self.expr = expr
        self.sense = sense
        self.rhs = rhs


class _FakeLinExpr:
    """Tiny fake Gurobi linear expression. / 极简 fake Gurobi 线性表达式。"""

    def __init__(self) -> None:
        """Initialize an empty expression. / 初始化空表达式。"""
        self.terms: list[tuple[float, _FakeVar]] = []

    def addTerms(self, coeff: float, var: _FakeVar) -> None:
        """Add one coefficient-variable term. / 添加一个系数-变量项。"""
        self.terms.append((float(coeff), var))

    def value(self, assignment: Mapping[_FakeVar, int]) -> float:
        """Evaluate under a binary assignment. / 在二元赋值下求值。"""
        return sum(coeff * assignment[var] for coeff, var in self.terms)

    def __eq__(self, rhs: object) -> "_FakeConstraint":  # type: ignore[override]
        """Build an equality constraint. / 构建等式约束。"""
        return _FakeConstraint(self, "==", float(rhs))

    def __le__(self, rhs: object) -> "_FakeConstraint":
        """Build a less-or-equal constraint. / 构建小于等于约束。"""
        return _FakeConstraint(self, "<=", float(rhs))

    def __ge__(self, rhs: object) -> "_FakeConstraint":
        """Build a greater-or-equal constraint. / 构建大于等于约束。"""
        return _FakeConstraint(self, ">=", float(rhs))


class _FakeModel:
    """Brute-force fake Gurobi model for small test ILPs. / 用穷举解小测试 ILP 的 fake Gurobi model。"""

    def __init__(self, name: str) -> None:
        """Initialize model storage. / 初始化 model 存储。"""
        self.name = name
        self.Params = SimpleNamespace(OutputFlag=0, TimeLimit=None, MIPGap=None)
        self._vars: list[_FakeVar] = []
        self._constraints: list[_FakeConstraint] = []
        self._objective = _FakeLinExpr()
        self._sense = _FakeGRB.MAXIMIZE
        self.Status = None
        self.ObjVal = None
        self.MIPGap = 0.0

    def addVar(self, *, vtype: object, name: str) -> _FakeVar:
        """Add one binary variable. / 添加一个二元变量。"""
        assert vtype == _FakeGRB.BINARY
        var = _FakeVar(name)
        self._vars.append(var)
        return var

    def update(self) -> None:
        """Match the real Gurobi API. / 对齐真实 Gurobi API。"""

    def addConstr(self, constraint: _FakeConstraint, *, name: str | None = None) -> None:
        """Add one linear constraint. / 添加一个线性约束。"""
        self._constraints.append(constraint)

    def setObjective(self, expr: _FakeLinExpr, sense: int) -> None:
        """Set the objective expression and sense. / 设置目标表达式和方向。"""
        self._objective = expr
        self._sense = sense

    def optimize(self) -> None:
        """Solve by enumerating all binary assignments. / 通过枚举所有二元赋值求解。"""
        best_assignment: dict[_FakeVar, int] | None = None
        best_value: float | None = None
        for values in product((0, 1), repeat=len(self._vars)):
            assignment = dict(zip(self._vars, values, strict=True))
            if not self._satisfies_all(assignment):
                continue
            value = self._objective.value(assignment)
            if best_value is None or (
                value > best_value if self._sense == _FakeGRB.MAXIMIZE else value < best_value
            ):
                best_value = value
                best_assignment = assignment
        if best_assignment is None:
            self.Status = _FakeGRB.INFEASIBLE
            self.ObjVal = None
            return
        self.Status = _FakeGRB.OPTIMAL
        self.ObjVal = best_value
        for var, value in best_assignment.items():
            var.X = float(value)

    def _satisfies_all(self, assignment: Mapping[_FakeVar, int]) -> bool:
        """Return whether all constraints hold. / 返回是否满足全部约束。"""
        for constraint in self._constraints:
            value = constraint.expr.value(assignment)
            if constraint.sense == "==" and abs(value - constraint.rhs) > 1e-9:
                return False
            if constraint.sense == "<=" and value > constraint.rhs + 1e-9:
                return False
            if constraint.sense == ">=" and value < constraint.rhs - 1e-9:
                return False
        return True


def _install_fake_gurobi(monkeypatch) -> None:
    """Install a fake `gurobipy` module for deterministic tests. / 安装 fake `gurobipy` 模块以便稳定测试。"""
    fake_module = SimpleNamespace(GRB=_FakeGRB, Model=_FakeModel, LinExpr=_FakeLinExpr)
    monkeypatch.setitem(sys.modules, "gurobipy", fake_module)
