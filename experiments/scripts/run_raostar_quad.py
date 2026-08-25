"""Run fixed-horizon DARP/RAO* comparisons on upstream's native QuadModel."""

from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
from math import isclose, isfinite
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping

from darp.planning.full_ilp import FullILPPlanner
from darp.planning.hilp import HILPPlanner
from experiments.external_raostar import load_manifest, run_external_raostar, validate_checkout
from experiments.benchmarks.raostar_quad import (
    build_upstream_quad_problem,
    deserialize_quad_policy,
    evaluate_quad_policy,
    load_upstream_quad_model,
    serialize_quad_policy,
)


DEFAULT_MANIFEST = Path(__file__).resolve().parents[1] / "manifests" / "raostar_quad.json"
_EXPERIMENT_MATRIX = load_manifest(DEFAULT_MANIFEST)["scenarios"]["quad"]["experiment_matrix"]
DEFAULT_HORIZONS = tuple(map(int, _EXPERIMENT_MATRIX["fixed_horizons"]))
DEFAULT_RISK_BUDGETS = tuple(map(float, _EXPERIMENT_MATRIX["risk_budgets"]))
DEFAULT_FULL_ILP_MAX_HORIZON = int(_EXPERIMENT_MATRIX["full_ilp_max_horizon"])
DEFAULT_REPETITIONS = int(_EXPERIMENT_MATRIX["smoke_repetitions"])
HILP_EXPANSION_ROUNDS_PER_HORIZON = int(
    _EXPERIMENT_MATRIX["hilp_expansion_rounds_per_horizon"]
)
BRIDGE_SOURCE = Path(__file__).resolve().parents[1] / "benchmarks" / "raostar_quad.py"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _optional_bool(value: Any) -> bool | None:
    return None if value is None else bool(value)


def _evaluation_is_feasible(evaluation: Any, risk_budget: float) -> bool:
    """Compare the independently recomputed risk without float rounding."""

    return Fraction(evaluation.first_entry_risk_exact) <= Fraction.from_float(
        float(risk_budget)
    )


def _base(algorithm: str, horizon: int, risk_budget: float, manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "benchmark": "upstream-raostar-quad",
        "algorithm": algorithm,
        "algorithm_provenance": (
            manifest["algorithm"]["provenance"]
            if algorithm == manifest["algorithm"]["label"]
            else "DARP repository implementation"
        ),
        "upstream_license_status": manifest["license"]["status"],
        "upstream_pinned_commit": manifest["algorithm"]["commit"],
        "horizon": horizon,
        "risk_budget": risk_budget,
        "status": "error",
        "error": None,
    }


def _run_darp(algorithm: str, model: Any, horizon: int, risk_budget: float, timeout: float, manifest: Mapping[str, Any]) -> dict[str, Any]:
    row = _base(algorithm, horizon, risk_budget, manifest)
    try:
        runtime, interface, duration = build_upstream_quad_problem(model, horizon)
        expansion_rounds = HILP_EXPANSION_ROUNDS_PER_HORIZON * horizon
        planner = (
            FullILPPlanner(risk_budget=risk_budget, solver_time_limit_ms=timeout * 1000.0)
            if algorithm == "darp-full-ilp"
            else HILPPlanner(
                risk_budget=risk_budget,
                expansion_rounds=expansion_rounds,
                heuristic_mode="one-step-greedy",
                solver_time_limit_ms=timeout * 1000.0,
            )
        )
        started = perf_counter()
        decision = planner.choose_action(runtime, interface, duration)
        elapsed = perf_counter() - started
        timing = dict(decision.timing)
        policy_duration_complete = decision.policy.duration_complete
        protocol_payload = None
        evaluation = None
        policy = {tuple(rule.observations): str(rule.action_label) for rule in decision.policy.rules}
        if len(policy) != len(decision.policy.rules):
            raise RuntimeError("DARP produced duplicate observation histories")
        protocol_payload = serialize_quad_policy(policy, horizon=horizon)
        if policy_duration_complete:
            evaluation = evaluate_quad_policy(model, policy, horizon)
        if evaluation is not None and decision.policy.achieved_utility is not None and not isclose(
            evaluation.darp_utility, float(decision.policy.achieved_utility), rel_tol=1e-9, abs_tol=1e-9
        ):
            raise RuntimeError("independent evaluator disagrees with DARP utility certificate")
        if evaluation is not None and decision.policy.active_constraint_value is not None and not isclose(
            evaluation.first_entry_risk, float(decision.policy.active_constraint_value), rel_tol=1e-9, abs_tol=1e-9
        ):
            raise RuntimeError("independent evaluator disagrees with DARP risk certificate")
        if decision.policy.feasible is False:
            raise RuntimeError("DARP returned a policy that violates the risk budget")
        if evaluation is not None and not _evaluation_is_feasible(evaluation, risk_budget):
            raise RuntimeError("independent evaluator found a DARP risk-budget violation")
        row.update(
            status="ok" if policy_duration_complete else "partial-policy",
            search_complete=bool(decision.complete),
            policy_duration_complete=policy_duration_complete,
            solver_status=decision.policy.solver_status,
            solver_numerically_optimal=_optional_bool(
                timing.get("solver_numerically_optimal")
            ),
            numerical_zero_gap=_optional_bool(timing.get("numerical_zero_gap")),
            mathematically_optimal=False,
            objective_coefficients_exact=_optional_bool(timing.get("objective_coefficients_exact")),
            constraint_coefficients_exact=_optional_bool(timing.get("constraint_coefficients_exact")),
            solver_time_limit_hit=_optional_bool(timing.get("solver_time_limit_hit")),
            frontier_refinement_exhausted=_optional_bool(
                timing.get("frontier_refinement_exhausted")
            ),
            certifying_utility_bound=_optional_bool(
                timing.get("certifying_utility_bound")
            ),
            global_expandable_frontier=timing.get("global_expandable_frontier"),
            expanded_nodes=timing.get("expanded_nodes"),
            frontier_nodes=timing.get("frontier_nodes"),
            iterations=timing.get("expansion_rounds"),
            ilp_variables=timing.get("ilp_variables"),
            ilp_constraints=timing.get("ilp_constraints"),
            tree_ilp_build_ms=timing.get("tree_ilp_build_ms"),
            gurobi_solve_ms=timing.get("gurobi_solve_ms"),
            root_action=decision.label,
            elapsed_s=elapsed,
            native_cost=None if evaluation is None else evaluation.native_cost,
            darp_utility=None if evaluation is None else evaluation.darp_utility,
            first_entry_risk=None if evaluation is None else evaluation.first_entry_risk,
            policy_feasible=None if evaluation is None else _evaluation_is_feasible(evaluation, risk_budget),
            policy_protocol=protocol_payload,
            policy_protocol_sha256=None if protocol_payload is None else protocol_payload["sha256"],
            expansion_round_limit=(
                None if algorithm == "darp-full-ilp" else expansion_rounds
            ),
        )
    except Exception as error:
        row["error"] = f"{type(error).__name__}: {error}"
    return row


def _run_raostar(checkout: Path, python: Path, horizon: int, risk_budget: float, timeout: float, manifest_path: Path, manifest: Mapping[str, Any], model: Any, accept_no_license: bool) -> dict[str, Any]:
    algorithm = str(manifest["algorithm"]["label"])
    row = _base(algorithm, horizon, risk_budget, manifest)
    try:
        result = run_external_raostar(
            checkout=checkout,
            python=python,
            chance_constraint=risk_budget,
            horizon=horizon,
            time_limit=timeout,
            manifest_path=manifest_path,
            accept_no_license=accept_no_license,
        )
        if result.get("status") != "ok":
            raise RuntimeError(str(result.get("message", result.get("error_type", result["status"]))))
        evaluation = None
        payload = result.get("policy_protocol")
        if result.get("search_complete"):
            if not isinstance(payload, Mapping):
                raise RuntimeError("complete upstream search did not export a policy protocol")
            policy = deserialize_quad_policy(payload)
            evaluation = evaluate_quad_policy(model, policy, horizon)
            if not isclose(evaluation.native_cost, float(result["root_value"]), rel_tol=1e-9, abs_tol=1e-9):
                raise RuntimeError("independent evaluator disagrees with upstream root value")
            if not isclose(evaluation.first_entry_risk, float(result["root_execution_risk"]), rel_tol=1e-9, abs_tol=1e-9):
                raise RuntimeError("independent evaluator disagrees with upstream execution risk")
            if not _evaluation_is_feasible(evaluation, risk_budget):
                raise RuntimeError("independent evaluator found an upstream risk-budget violation")
        row.update(
            status="ok",
            search_complete=bool(result.get("search_complete")),
            termination_reason=result.get("termination_reason"),
            root_action=result.get("root_action"),
            elapsed_s=result.get("elapsed_s"),
            graph_nodes=result.get("graph_nodes"),
            iterations=result.get("iterations"),
            admissibility_warning_count=result.get("admissibility_warning_count"),
            native_cost=None if evaluation is None else evaluation.native_cost,
            darp_utility=None if evaluation is None else evaluation.darp_utility,
            first_entry_risk=None if evaluation is None else evaluation.first_entry_risk,
            policy_feasible=None if evaluation is None else _evaluation_is_feasible(evaluation, risk_budget),
            policy_protocol=payload,
            policy_protocol_sha256=None if payload is None else payload.get("sha256"),
            manifest_sha256=result["manifest"]["sha256"],
            frozen_worker_sha256=result["local_source_snapshot"]["worker_sha256"],
        )
    except Exception as error:
        row["error"] = f"{type(error).__name__}: {error}"
    return row


def run_matrix(*, checkout: Path, python: Path, manifest_path: Path = DEFAULT_MANIFEST, timeout: float = 60.0, include_full_ilp: bool = False, full_ilp_max_horizon: int = DEFAULT_FULL_ILP_MAX_HORIZON, accept_no_license: bool = False, horizons: tuple[int, ...] = DEFAULT_HORIZONS, risk_budgets: tuple[float, ...] = DEFAULT_RISK_BUDGETS, repetitions: int = DEFAULT_REPETITIONS) -> list[dict[str, Any]]:
    if full_ilp_max_horizon <= 0:
        raise ValueError("full_ilp_max_horizon must be positive")
    if repetitions <= 0:
        raise ValueError("repetitions must be positive")
    if not horizons or any(
        isinstance(horizon, bool) or not isinstance(horizon, int) or horizon <= 0
        for horizon in horizons
    ):
        raise ValueError("horizons must be non-empty positive integers")
    if not risk_budgets or any(
        not isfinite(float(budget)) or not 0.0 <= float(budget) <= 1.0
        for budget in risk_budgets
    ):
        raise ValueError("risk_budgets must be non-empty probabilities in [0, 1]")
    manifest = load_manifest(manifest_path)
    if manifest["license"].get("status") == "license-not-provided" and not accept_no_license:
        raise ValueError(
            "The pinned RAO* checkout provides no license; pass "
            "--accept-no-license only after reviewing the manifest notice."
        )
    validate_checkout(checkout, manifest)
    source_hashes = {
        "runner_source_sha256": _sha256(Path(__file__).resolve()),
        "bridge_source_sha256": _sha256(BRIDGE_SOURCE),
        "manifest_source_sha256": _sha256(Path(manifest_path).resolve()),
    }
    model = load_upstream_quad_model(checkout)
    rows = []
    for repetition in range(1, repetitions + 1):
        for horizon in horizons:
            algorithms = ["darp-hilp", str(manifest["algorithm"]["label"])]
            if include_full_ilp and horizon <= full_ilp_max_horizon:
                algorithms.insert(1, "darp-full-ilp")
            for budget in risk_budgets:
                for algorithm in algorithms:
                    if algorithm == manifest["algorithm"]["label"]:
                        row = _run_raostar(checkout, python, horizon, budget, timeout, manifest_path, manifest, model, accept_no_license)
                    else:
                        row = _run_darp(algorithm, model, horizon, budget, timeout, manifest)
                    row["repetition"] = repetition
                    rows.append(row)
    validate_checkout(checkout, manifest)
    source_stable = source_hashes == {
        "runner_source_sha256": _sha256(Path(__file__).resolve()),
        "bridge_source_sha256": _sha256(BRIDGE_SOURCE),
        "manifest_source_sha256": _sha256(Path(manifest_path).resolve()),
    }
    for row in rows:
        row.update(source_hashes)
        row["source_stable"] = source_stable
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--horizons", nargs="+", type=int, default=DEFAULT_HORIZONS)
    parser.add_argument(
        "--risk-budgets", nargs="+", type=float, default=DEFAULT_RISK_BUDGETS
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=DEFAULT_REPETITIONS,
        help="number of fresh planner/process repetitions per matrix cell",
    )
    parser.add_argument(
        "--include-full-ilp",
        action="store_true",
        help="include the exhaustive full-ILP oracle only for small horizons",
    )
    parser.add_argument(
        "--full-ilp-max-horizon",
        type=int,
        default=DEFAULT_FULL_ILP_MAX_HORIZON,
        help="largest horizon eligible for --include-full-ilp (default: 2)",
    )
    parser.add_argument("--accept-no-license", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        rows = run_matrix(
            checkout=args.checkout,
            python=args.python,
            manifest_path=args.manifest,
            timeout=args.timeout,
            include_full_ilp=args.include_full_ilp,
            full_ilp_max_horizon=args.full_ilp_max_horizon,
            accept_no_license=args.accept_no_license,
            horizons=tuple(args.horizons),
            risk_budgets=tuple(args.risk_budgets),
            repetitions=args.repetitions,
        )
    except ValueError as error:
        parser.error(str(error))
    text = "\n".join(json.dumps(row, sort_keys=True, allow_nan=False) for row in rows) + "\n"
    if args.output is None:
        print(text, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    return 0 if all(row["status"] == "ok" for row in rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())
