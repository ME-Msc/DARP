"""Shared file-based entry point for CLI and experiments.

命令行与实验共用的文件输入求解入口。
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

from darp.adapter.duration import build_duration_evaluator
from darp.adapter.kernel import RDDLKernel, StateKey
from darp.adapter.loader import load_rddl
from darp.adapter.runtime import PyRDDLGymRuntime
from darp.planning.decision import ActionDecision
from darp.planning.full_ilp import FullILPPlanner
from darp.planning.heuristic import UtilityHeuristic
from darp.planning.hilp import HILPPlanner

PlannerName = Literal["hilp", "full-ilp"]
RootBeliefFactory = Callable[
    [RDDLKernel],
    Mapping[StateKey, float],
]


@dataclass(frozen=True, slots=True)
class DARPResult:
    """One decision with planner-only timing. 一个决策及仅含规划阶段的耗时。"""

    decision: ActionDecision
    elapsed_s: float
    risk_budget: float | None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready result. 返回可保存为 JSON 的可复用求解结果。"""
        return {
            "format": "darp-result",
            "version": 1,
            "elapsed_s": self.elapsed_s,
            "risk_budget": self.risk_budget,
            "decision": self.decision.to_dict(),
        }

    def save(self, path: str | Path) -> Path:
        """Write portable UTF-8 JSON. 将结果保存为 UTF-8 JSON 文件。"""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
        return target

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> DARPResult:
        """Restore :meth:`to_dict` output. 从 to_dict 产生的数据恢复结果。"""
        if value.get("format") != "darp-result" or value.get("version") != 1:
            raise ValueError("Unsupported DARP result format or version.")
        budget = value.get("risk_budget")
        decision = value["decision"]
        if not isinstance(decision, Mapping):
            raise TypeError("DARPResult decision must be an object.")
        return cls(
            decision=ActionDecision.from_dict(decision),
            elapsed_s=float(value["elapsed_s"]),
            risk_budget=None if budget is None else float(budget),
        )

    @classmethod
    def load(cls, path: str | Path) -> DARPResult:
        """Load :meth:`save` output. 读取 save 保存的结果文件。"""
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(value, Mapping):
            raise TypeError("DARP result JSON must contain an object.")
        return cls.from_dict(value)


def solve_rddl(
    domain: str | Path,
    instance: str | Path,
    *,
    planner: PlannerName = "hilp",
    seed: int = 0,
    risk_budget: float | None = None,
    expansion_rounds: int | None = None,
    frontier_width: int | None = None,
    heuristic: UtilityHeuristic | None = None,
    terminal_heuristic: bool = False,
    timeout_s: float | None = 60.0,
    full_ilp_max_tree_nodes: int | None = 100_000,
    root_belief_factory: RootBeliefFactory | None = None,
) -> DARPResult:
    """Load one RDDL problem and search. 加载一个 RDDL 问题并执行一次搜索。"""

    _validate_options(planner, heuristic, terminal_heuristic, timeout_s)
    problem = load_rddl(domain, instance)
    runtime = PyRDDLGymRuntime(problem.env)
    runtime.reset(seed=seed)
    interface = problem.build_grounded_view().build_and_or_interface(runtime)
    kernel = interface.kernel
    if kernel is None:
        raise ValueError("RDDL duration evaluation requires DARP's RDDL kernel.")
    evaluator = build_duration_evaluator(
        kernel,
        interface.actions,
        horizon=runtime.horizon,
    )
    budget = risk_budget if risk_budget is not None else kernel.grounded_model.risk_budget

    root_belief = None
    if root_belief_factory is not None:
        root_belief = root_belief_factory(kernel)
        if not isinstance(root_belief, Mapping):
            raise TypeError("A root-belief factory must return a mapping.")
    limit_ms = None if timeout_s is None else timeout_s * 1000.0
    selected = (
        FullILPPlanner(
            risk_budget=budget,
            max_tree_nodes=full_ilp_max_tree_nodes,
            solver_time_limit_ms=limit_ms,
            terminal_heuristic=heuristic if terminal_heuristic else None,
        )
        if planner == "full-ilp"
        else HILPPlanner(
            expansion_rounds=expansion_rounds,
            frontier_width=frontier_width,
            frontier_heuristic=heuristic,
            terminal_heuristic=terminal_heuristic,
            risk_budget=budget,
            solver_time_limit_ms=limit_ms,
        )
    )
    started = perf_counter()
    decision = selected.choose_action(
        runtime,
        interface,
        evaluator,
        root_belief=root_belief,
    )
    return DARPResult(
        decision=decision,
        elapsed_s=perf_counter() - started,
        risk_budget=budget,
    )


def _validate_options(
    planner: str,
    heuristic: UtilityHeuristic | None,
    terminal_heuristic: bool,
    timeout_s: float | None,
) -> None:
    if planner not in ("hilp", "full-ilp"):
        raise ValueError(f"Unknown DARP planner: {planner!r}")
    if timeout_s is not None and timeout_s <= 0:
        raise ValueError("timeout_s must be positive when provided")
    if terminal_heuristic and heuristic is None:
        raise ValueError("terminal_heuristic requires an external heuristic")
    if planner == "full-ilp" and heuristic is not None and not terminal_heuristic:
        raise ValueError(
            "Full-ILP accepts an external heuristic only as a terminal value."
        )


__all__ = [
    "DARPResult",
    "PlannerName",
    "RootBeliefFactory",
    "solve_rddl",
]
