"""Independent benchmark runs; no report generation. / 独立运行实验，不生成报告。"""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import logging
import platform
import re
import sys
from contextlib import nullcontext
from itertools import product
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from darp.adapter.loader import load_rddl
from darp.executor import PolicyExecutor
from darp.planning.heuristic import load_utility_heuristic
from darp.planning.rank import validate_rank
from darp.solve import DARPResult, solve_rddl

logger = logging.getLogger(__name__)
ALGORITHMS = ("HILP", "FullILP", "RAOstar", "pruningF", "pruningEF")
FIELDS = ("case", "input_digest", "domain", "instance", "algorithm", "alpha", "lambda",
          "trial", "seed", "status", "objective", "risk", "complete", "time_s",
          "ilp_variables", "full_partial_ilp_variables", "rank_filter_ms", "rank_fallbacks",
          "episodes", "risk_rate", "physical_duration_mean", "rollout_time_s", "result_file", "error")


def path_label(path: Path) -> str:
    """Portable repository path, absolute for external inputs. / 仓库内相对路径，外部绝对路径。"""
    path = path.resolve()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def input_digest(domain: Path, instance: Path) -> str:
    """Match actual input contents, not filenames. / 按实际输入内容匹配，不猜测文件名。"""
    return hashlib.sha256(domain.read_bytes() + b"\0" + instance.read_bytes()).hexdigest()


def safe_name(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", value):
        raise ValueError(f"Use letters, digits, '-' and '_' for directory names: {value!r}")
    return value


def describe_input(domain: Path, instance: Path) -> dict:
    """Read the problem through the real parser. / 使用实际解析器检查问题输入。"""
    problem = load_rddl(domain, instance)
    try:
        return {"domain": path_label(domain), "instance": path_label(instance),
                "input_digest": input_digest(domain, instance),
                "horizon": int(problem.env.horizon),
                "risk_budget": getattr(problem.native_ast.instance, "risk_budget", 0.0)}
    finally:
        problem.env.close()


def run_trial(domain: Path, instance: Path, algorithm: str, alpha: float, fraction: float,
              seed: int, timeout: float, episodes: int, policy_path: Path,
              heuristic: str | None, terminal_heuristic: bool, raostar=None) -> dict:
    """Solve then replay outside planning timing. / 先求解，再在规划计时之外回放。"""
    if algorithm == "RAOstar":
        from experiments.grid.raostar import run_instance
        return run_instance(raostar, domain, instance, timeout)
    # Preserve the existing F-only ablation without modifying the solver.
    # 保留已有 F-only 消融方法，不修改求解算法。
    context = patch("darp.planning.rank._prune_expanded", return_value=None) if algorithm == "pruningF" else nullcontext()
    with context:
        result = solve_rddl(domain, instance,
                            planner="full-ilp" if algorithm == "FullILP" else "hilp",
                            seed=seed, timeout_s=timeout,
                            heuristic=load_utility_heuristic(heuristic) if heuristic else None,
                            terminal_heuristic=terminal_heuristic,
                            full_ilp_max_tree_nodes=800_000,
                            rank_alpha=alpha, rank_lambda=fraction)
    policy = result.decision.policy
    valid = (policy.duration_complete and policy.feasible is True
             and policy.achieved_utility is not None
             and (algorithm.startswith("pruning") or result.decision.complete))
    result.save(policy_path)
    row = {"status": "ok" if valid else "incomplete", "time_s": result.elapsed_s,
           "objective": -policy.achieved_utility if policy.achieved_utility is not None else "",
           "risk": policy.active_constraint_value, "complete": result.decision.complete,
           "result_file": "policies/" + policy_path.name}
    row.update({name: result.decision.timing.get(name, "") for name in FIELDS
                if name in result.decision.timing})
    if valid and episodes:
        env = load_rddl(domain, instance).env
        try:
            restored = DARPResult.load(policy_path)
            stats = PolicyExecutor(restored.decision.policy).evaluate(env, episodes=episodes, seed=seed)
            row.update({key: stats[key] for key in ("episodes", "risk_rate", "physical_duration_mean", "rollout_time_s")})
        finally:
            env.close()
    return row


def run_batch(*, scene: str, name: str, instances: list[Path], algorithms: list[str],
              domain: Path | None = None, alpha: float = 1., lambdas: list[float] | None = None,
              trials: int = 1, seed: int = 2023, timeout: float = 120., episodes: int = 10,
              heuristic: str | None = None, terminal_heuristic: bool = False,
              resume: bool = False, constrained_repo: Path | None = None,
              raostar_repo: Path | None = None, full_ilp_max_horizon: int | None = None) -> Path:
    """Checkpoint each trial and refuse incompatible resume. / 逐次保存并拒绝不兼容的续跑。"""
    safe_name(scene)
    safe_name(name)
    if not instances or not algorithms or set(algorithms) - set(ALGORITHMS) or len(set(algorithms)) != len(algorithms):
        raise ValueError("Select at least one input and a supported algorithm.")
    if trials < 1 or episodes < 0 or timeout <= 0:
        raise ValueError("Require trials>=1, episodes>=0, timeout>0.")
    lambdas = list(dict.fromkeys(lambdas or [.5]))
    for fraction in lambdas:
        validate_rank(alpha, fraction)
    inputs = [(domain or path.with_name("domain.rddl"), path) for path in instances]
    descriptions = [describe_input(d, i) for d, i in inputs]
    if len({d["input_digest"] for d in descriptions}) != len(descriptions):
        raise ValueError("Duplicate problem inputs in one batch.")
    if scene == "grid" and heuristic is None:
        heuristic = "benchmarks.grid.heuristic:MANHATTAN"
        terminal_heuristic = True
    if terminal_heuristic and not heuristic:
        raise ValueError("terminal_heuristic requires an explicit heuristic.")
    import gurobipy
    settings = {"schema": 1, "scene": scene, "inputs": descriptions,
                "algorithms": algorithms, "alpha": alpha, "lambdas": lambdas,
                "trials": trials, "seed": seed, "timeout_s": timeout, "episodes": episodes,
                "heuristic": heuristic, "terminal_heuristic": terminal_heuristic,
                "timing_scope": "planner-only", "MIPGap": 1e-6,
                "full_ilp_max_tree_nodes": 800_000, "full_ilp_max_horizon": full_ilp_max_horizon,
                "threads": "Gurobi default", "python": platform.python_version(),
                "platform": platform.platform(), "gurobi": list(gurobipy.gurobi.version())}
    if "RAOstar" in algorithms:
        if scene != "grid":
            raise ValueError("RAOstar has an adapter only for the grid benchmark.")
        from experiments.grid.raostar import RAOStarRunner, CONSTRAINED_POMDP_COMMIT, RAOSTAR_COMMIT
        raostar = RAOStarRunner.create(constrained_pomdp_repo=constrained_repo,
                                      raostar_repo=raostar_repo, cache_root=ROOT / ".cache/baselines")
        settings.update(constrained_pomdp_commit=CONSTRAINED_POMDP_COMMIT, raostar_commit=RAOSTAR_COMMIT)
    else:
        raostar = None
    output = ROOT / "experiments" / scene / "runs" / name
    previous = []
    if output.exists():
        if not resume:
            raise FileExistsError(f"Batch already exists; choose a new --name or --resume: {output}")
        if json.loads((output / "config.json").read_text()) != settings:
            raise ValueError("Resume input contents or settings differ from the saved batch.")
        if (output / "results.csv").exists():
            with (output / "results.csv").open(newline="") as stream:
                previous = list(csv.DictReader(stream))
    else:
        output.mkdir(parents=True)
        (output / "config.json").write_text(json.dumps(settings, indent=2) + "\n")
    def key(row):
        return (row["input_digest"], row["algorithm"], float(row["alpha"]), float(row["lambda"]), int(row["trial"]))
    done = {key(row) for row in previous}
    methods = [(method, alpha if method.startswith("pruning") else 1., fraction)
               for method in algorithms for fraction in (lambdas if method.startswith("pruning") else [1.])]
    with (output / "results.csv").open("a" if previous else "w", newline="") as stream:
        writer = csv.DictWriter(stream, FIELDS, lineterminator="\n")
        if not previous:
            writer.writeheader()
        index = len(previous)
        for ((domain_path, instance), info), trial in product(zip(inputs, descriptions), range(1, trials + 1)):
            for method, weight, fraction in methods:
                if method == "FullILP" and full_ilp_max_horizon is not None and info["horizon"] > full_ilp_max_horizon:
                    continue
                row = {"case": instance.stem, "input_digest": info["input_digest"],
                       "domain": info["domain"], "instance": info["instance"],
                       "algorithm": method, "alpha": weight, "lambda": fraction,
                       "trial": trial, "seed": seed + trial - 1}
                if key(row) in done:
                    continue
                index += 1
                started = perf_counter()
                try:
                    row.update(run_trial(domain_path, instance, method, weight, fraction,
                                         row["seed"], timeout, episodes,
                                         output / "policies" / f"policy-{index:05d}.json",
                                         heuristic, terminal_heuristic, raostar))
                except Exception as exc:
                    logger.exception("Trial failed: case=%s algorithm=%s trial=%s seed=%s", row["case"], method, trial, row["seed"])
                    row.update(status="timeout" if isinstance(exc, TimeoutError) else "error",
                               error=f"{type(exc).__name__}: {exc}", time_s=perf_counter() - started)
                writer.writerow(row)
                stream.flush()
                done.add(key(row))
                print(f"{row['case']} {method} lambda={fraction:g}: {row['status']}", flush=True)
    return output


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene")
    parser.add_argument("--name", required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--instances", type=Path, nargs="+")
    parser.add_argument("--domain", type=Path)
    parser.add_argument("--algorithms", choices=ALGORITHMS, nargs="+")
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--lambdas", type=float, nargs="+")
    parser.add_argument("--trials", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--timeout", type=float)
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--full-ilp-max-horizon", type=int)
    parser.add_argument("--heuristic")
    parser.add_argument("--terminal-heuristic", action="store_true", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--constrained-repo", type=Path)
    parser.add_argument("--raostar-repo", type=Path)
    args = vars(parser.parse_args(argv))
    config_path = args.pop("config")
    config = json.loads(config_path.read_text()) if config_path else {}
    config.update({key: value for key, value in args.items() if value is not None})
    paths = []
    for pattern in config.get("instances", []):
        pattern = Path(pattern).expanduser()
        matches = sorted(glob.glob(str(pattern if pattern.is_absolute() else ROOT / pattern)))
        if not matches:
            raise ValueError(f"No input matches {pattern}")
        paths.extend(Path(p) for p in matches)
    config["instances"] = paths
    if config.get("domain"):
        config["domain"] = Path(config["domain"]).expanduser().resolve()
    config.setdefault("scene", "grid")
    config.setdefault("algorithms", ["HILP"])
    output = run_batch(**config)
    print(output)
    with (output / "results.csv").open(newline="") as stream:
        return int(any(row["status"] != "ok" for row in csv.DictReader(stream)))


if __name__ == "__main__":
    raise SystemExit(main())
