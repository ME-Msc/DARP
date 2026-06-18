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
from darp.planning.ilp_tree import build_full_tree_ilp
from darp.planning.rollout import action_label


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


def test_policy_tree_ilp_contains_root_flow_and_risk_rows():
    """Check policy-tree ILP rows before solving. / 检查 policy-tree ILP 的 root/flow/risk 行。"""
    runtime, interface, duration = _policy_tree_inputs()

    tree_ilp = build_full_tree_ilp(
        runtime,
        interface,
        duration,
        risk_budget=0.0,
    )

    names = [constraint.name for constraint in tree_ilp.spec.constraints]
    assert names[0] == "root_action"
    assert "risk_budget" in names
    assert any(name.startswith("flow_") for name in names)
    assert len(tree_ilp.root_variable_ids) == len(interface.actions)
    assert set(tree_ilp.root_variable_ids).issubset(set(tree_ilp.spec.variable_ids()))


def test_policy_tree_ilp_declares_every_flow_variable():
    """Check flow rows only reference declared duration-feasible children. / 检查 flow 行只引用已声明且 duration 可继续的子变量。"""
    runtime, interface, duration = _policy_tree_inputs()

    for horizon in (1, 2, 3):
        runtime = _TwoActionRuntime(horizon=horizon)
        interface, duration = _interface_and_duration(runtime)
        tree_ilp = build_full_tree_ilp(runtime, interface, duration)

        tree_ilp.spec.validate()


def test_full_ilp_planner_uses_gurobi_when_available(monkeypatch):
    """Check full-tree planning uses Gurobi-selected root variables. / 检查 full-tree planner 使用 Gurobi 选择根变量。"""
    _install_fake_gurobi(monkeypatch)
    runtime, interface, duration = _two_action_inputs()
    planner = FullILPPlanner()

    decision = planner.choose_action(runtime, interface, duration, remaining_depth=runtime.horizon)

    assert decision.label == "go"
    assert decision.complete is True
    assert planner.last_ilp_result is not None
    assert planner.last_ilp_result.is_optimal


def test_full_ilp_planner_expands_to_remaining_depth(monkeypatch):
    """Check full-ILP expands to the remaining horizon. / 检查 full-ILP 展开到当前剩余 horizon。"""
    _install_fake_gurobi(monkeypatch)
    runtime, interface, duration = _policy_tree_inputs()
    planner = FullILPPlanner()

    decision = planner.choose_action(runtime, interface, duration, remaining_depth=runtime.horizon)

    assert decision.remaining_depth == runtime.horizon
    assert planner.last_policy_tree is not None
    assert max(item.node.history.depth for item in planner.last_policy_tree.variable_items.values()) == runtime.horizon


def test_full_ilp_uses_explicit_root_belief_over_runtime_state():
    """Check full-ILP root belief can override representative runtime state. / 检查 full-ILP root belief 可覆盖代表 runtime state。"""
    runtime = _TwoActionRuntime(at_goal=False, horizon=1)
    interface, duration = _interface_and_duration(runtime)
    kernel = interface.exact_kernel
    root_belief = {kernel._state_key(True): 1.0}

    tree_ilp = build_full_tree_ilp(
        runtime,
        interface,
        duration,
        root_belief=root_belief,
    )

    noop_id = next(
        var_id
        for var_id, item in tree_ilp.variable_items.items()
        if item.action_label == "noop"
    )
    assert "at_goal" in tree_ilp.variable_metrics[noop_id].state_label
    assert "not_at_goal" not in tree_ilp.variable_metrics[noop_id].state_label


def test_hilp_partial_tree_uses_gurobi_when_available(monkeypatch):
    """Check HILP solves the current partial-tree p-ILP. / 检查 HILP 求解当前 partial-tree p-ILP。"""
    _install_fake_gurobi(monkeypatch)
    runtime, interface, duration = _two_action_inputs()
    planner = HILPPlanner(lookahead_depth=1, max_iterations=1, frontier_width=1)

    decision = planner.choose_action(runtime, interface, duration, remaining_depth=runtime.horizon)

    assert decision.label == "go"
    assert planner.last_stats is not None
    assert planner.last_ilp_result is not None
    assert planner.last_stats.partial_variable_count > 0


def test_hilp_keeps_partial_tree_below_full_horizon(monkeypatch):
    """Check HILP selects from a partial tree instead of full-ILP fallback. / 检查 HILP 直接从 partial tree 选动作。"""
    _install_fake_gurobi(monkeypatch)
    runtime, interface, duration = _policy_tree_inputs()
    planner = HILPPlanner(lookahead_depth=2, max_iterations=1, frontier_width=1)

    decision = planner.choose_action(runtime, interface, duration, remaining_depth=runtime.horizon)
    full_tree = build_full_tree_ilp(runtime, interface, duration)

    assert decision.label == "go"
    assert planner.last_partial_tree is not None
    partial_depth = max(item.node.history.depth for item in planner.last_partial_tree.variable_items.values())
    full_depth = max(item.node.history.depth for item in full_tree.variable_items.values())
    assert partial_depth == 2
    assert full_depth == runtime.horizon
    assert len(planner.last_partial_tree.spec.variables) < len(full_tree.spec.variables)


def _two_action_inputs():
    """Build a tiny deterministic problem where action `go` dominates. / 构建 `go` 动作明显更优的小问题。"""
    runtime = _TwoActionRuntime()
    interface, duration = _interface_and_duration(runtime)
    return runtime, interface, duration


def _policy_tree_inputs():
    """Build a tiny runtime that expands beyond the root. / 构建一个会展开到根节点之后的小 runtime。"""
    runtime = _TwoActionRuntime(horizon=3)
    interface, duration = _interface_and_duration(runtime)
    return runtime, interface, duration


def _interface_and_duration(runtime):
    """Build common planner inputs for tiny runtime tests. / 为小型 runtime 测试构建通用 planner 输入。"""
    interface = ANDORSearchInterface.from_actions_and_observations(
        actions=tuple(
            ActionChoice(label=action_label(action), assignment=action)
            for action in runtime.action_candidates()
        ),
        observation_scope=ObservationScope(mode="mdp-state", variables=("at_goal",)),
        exact_kernel=_TinyExactKernel(),
    )
    duration = build_duration_sidecar({"kind": "fixed", "default": 1}).evaluator(horizon=runtime.horizon)
    return interface, duration


class _TwoActionRuntime:
    """Small runtime with one rewarding boolean action. / 只有一个高 reward 布尔动作的小 runtime。"""

    discount = 1.0

    def __init__(self, at_goal: bool = False, horizon: int = 1) -> None:
        """Initialize the runtime state. / 初始化 runtime state。"""
        self.at_goal = at_goal
        self.horizon = horizon
        self.state = {"at_goal": at_goal}

    def clone(self) -> "_TwoActionRuntime":
        """Return an isolated runtime copy. / 返回隔离 runtime 副本。"""
        return _TwoActionRuntime(self.at_goal, self.horizon)

    def action_candidates(self) -> tuple[dict[str, bool], ...]:
        """Return noop and go actions. / 返回 noop 与 go 动作。"""
        return ({"go": False}, {"go": True})

    def step(self, action: Mapping[str, Any]):
        """Apply one transition and reward. / 执行一次转移和 reward。"""
        reward = 5.0 if bool(action.get("go")) else 0.0
        self.at_goal = bool(action.get("go")) or self.at_goal
        self.state = {"at_goal": self.at_goal}
        return dict(self.state), reward, False, False, {}


class _TinyExactKernel:
    """Exact finite kernel for the tiny two-action tests. / 小型双动作测试用 exact finite kernel。"""

    def initial_belief_from_state(self, state: Mapping[str, Any]):
        """Return a singleton belief. / 返回单点 belief。"""
        return {self._state_key(bool(state.get("at_goal", False))): 1.0}

    def fluent_belief(self, belief: Mapping[Any, float]):
        """Return state fluent marginals. / 返回 state fluent 边缘概率。"""
        return {"at_goal": sum(prob for state, prob in belief.items() if dict(state).get("at_goal"))}

    def state_label(self, state):
        """Return a compact state label. / 返回紧凑 state 标签。"""
        return "at_goal" if dict(state).get("at_goal") else "not_at_goal"

    def expand_action(self, belief: Mapping[Any, float], action: Mapping[str, Any]):
        """Return exact transition, observation, reward, and risk constants. / 返回 exact 转移、观测、奖励和风险常量。"""
        prior: dict[Any, float] = {}
        utility = 0.0
        for state, probability in belief.items():
            at_goal = bool(dict(state).get("at_goal"))
            next_at_goal = at_goal or bool(action.get("go"))
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

    def expand_safe_action(self, safe_belief: Mapping[Any, float], action: Mapping[str, Any]):
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
        """Convert a state key to a mapping. / 将 state key 转成 mapping。"""
        return dict(key)

    def transition_distribution(self, state: Mapping[str, Any], action: Mapping[str, Any]):
        """Return the deterministic tiny transition. / 返回确定性小模型转移。"""
        next_at_goal = bool(state.get("at_goal")) or bool(action.get("go"))
        return {self._state_key(next_at_goal): 1.0}

    def observation_probability(self, observation, state, action):
        """Return the MDP-state observation likelihood. / 返回 MDP-state 观测似然。"""
        del action
        return 1.0 if observation == (("__state__", state),) else 0.0

    def _state_key(self, at_goal: bool):
        """Return a stable exact-state key. / 返回稳定 exact state key。"""
        return (("at_goal", at_goal),)


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
