"""Paired HILP/Rank timings and F-only ablation. / 同批计时、仅 F 筛选消融与历史 RAO* 参考。"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import logging
import platform
import re
from contextlib import nullcontext
from itertools import product
from pathlib import Path
from statistics import median
from time import perf_counter
from unittest.mock import patch

from darp.adapter.loader import load_rddl
from darp.executor import PolicyExecutor
from darp.planning.rank import validate_rank
from darp.solve import DARPResult, solve_rddl

BASE = importlib.import_module("experiments.DARP-vs-RAOstar-grid.darp_runner")
DIRECTORY = Path(__file__).resolve().parent
VERSION = "bottom-up-retention-v2"
TIMINGS = ("rank_filter_ms", "gurobi_model_update_ms", "gurobi_optimize_ms",
           "rank_fallbacks", "ilp_variables", "full_partial_ilp_variables",
           "partial_ilp_solves", "ilp_e_sent_total", "ilp_e_full_total",
           "ilp_f_sent_total", "ilp_f_full_total")
FIELDS = ("version", "size", "horizon", "delta", "trial", "seed", "method",
          "alpha", "lambda", "timeout_s", "status", "objective", "risk", "time_s",
          "complete", "histories", *TIMINGS, "episodes", "risk_rate",
          "physical_duration_mean", "rollout_time_s", "result_file", "error")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[5, 100])
    parser.add_argument("--horizons", type=int, nargs="+", default=[3, 4, 5, 6])
    parser.add_argument("--deltas", type=float, nargs="+", default=[0.1, 0.2, 0.3])
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--lambdas", type=float, nargs="+", default=[.3, .5, .7, .9])
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--timeout", type=float, default=120.)
    parser.add_argument("--episodes", type=int, default=1000, help="replay trial-1 saved policies outside timing")
    parser.add_argument("--ablation", action=argparse.BooleanOptionalAction, default=True,
                        help="include the F-only ablation (disable only E-subtree deletion); --no-ablation omits it")
    parser.add_argument("--output", type=Path, default=DIRECTORY / "output/pruning-comparison-raw.csv")
    parser.add_argument("--summary", type=Path, default=DIRECTORY / "output/letax/pruning-comparison.tex")
    parser.add_argument("--baseline", type=Path, default=BASE.RDDL_DIR.parent / "output/table2-raw.csv")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.summary.suffix != ".tex":
        raise ValueError("The experiment summary must be a .tex file.")
    if args.trials < 1 or args.episodes < 0 or args.timeout <= 0:
        raise ValueError("Require trials>=1, episodes>=0 and timeout>0.")
    for fraction in args.lambdas:
        validate_rank(args.alpha, fraction)
        if fraction == 1.:
            raise ValueError("Use 0<lambda<1; HILP lambda=1 is included automatically.")
    rows = _read_csv(args.output) if args.resume or args.summary_only else []
    if args.summary_only and not rows:
        raise ValueError(f"No experiment results found: {args.output}")
    if len({_key(row) for row in rows}) != len(rows):
        raise ValueError("Duplicate experiment keys in the raw results.")
    if rows and (set(rows[0]) != set(FIELDS) or any(row["version"] != VERSION for row in rows)):
        raise ValueError("Incompatible output version/schema; use a new output path.")
    if any(float(row["alpha"]) != args.alpha or float(row["timeout_s"]) != args.timeout
           or int(row["seed"]) != args.seed for row in rows):
        raise ValueError("Resume settings differ from recorded alpha/timeout/seed.")
    if any(row.get("episodes") and int(row["episodes"]) != args.episodes for row in rows):
        raise ValueError("Resume replay settings differ from recorded episodes.")
    methods = [("HILP", 1.)] + [("Rank", fraction) for fraction in args.lambdas]
    if args.ablation:
        methods += [("F-only", fraction) for fraction in args.lambdas]
    cases = list(product(args.sizes, args.horizons, args.deltas))
    if not args.summary_only:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        # Warm up imports/license before measured trials. / 正式计时前预热导入与许可证。
        BASE.run_darp(BASE.RDDL_DIR / "instance_5_h3.rddl", delta=.1, seed=args.seed,
                      timeout_s=args.timeout, rank_lambda=.5)
        _metadata(args, len(rows))
        done = {_key(row) for row in rows}
        with args.output.open("a" if rows and args.resume else "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, FIELDS, lineterminator="\n")
            if not rows:
                writer.writeheader()
            for index, (size, horizon, delta) in enumerate(cases):
                instance = _instance(size, horizon, args.output.parent)
                for trial in range(1, args.trials + 1):
                    # Rotate order, run serially to avoid contention. / 轮换方法顺序，串行求解避免资源竞争。
                    offset = (index + trial - 1) % len(methods)
                    for method, fraction in methods[offset:] + methods[:offset]:
                        key = (size, horizon, delta, trial, method, fraction)
                        if key in done:
                            continue
                        row = _run(instance, size, horizon, delta, trial, method, fraction, args)
                        writer.writerow(row)
                        stream.flush()
                        rows.append({name: str(row.get(name, "")) for name in FIELDS})
                        done.add(key)
                        print(f"{key}: {row['status']} {row['time_s']:.3f}s cost={row.get('objective', '')}", flush=True)
                _summary(rows, _read_csv(args.baseline), args.summary)
    _summary(rows, _read_csv(args.baseline), args.summary)
    print(f"raw: {args.output}\nsummary: {args.summary}")
    return 0


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _key(row: dict) -> tuple:
    return (int(row["size"]), int(row["horizon"]), float(row["delta"]),
            int(row["trial"]), row["method"], float(row["lambda"]))


def _instance(size: int, horizon: int, output: Path) -> Path:
    """Reuse inputs; larger h changes horizon only. / 复用场景，扩大 h 只改 horizon。"""
    if size not in (5, 100) or horizon < 1:
        raise ValueError("This experiment supports size=5/100 and positive horizons.")
    path = BASE.RDDL_DIR / f"instance_{size}_h{horizon}.rddl"
    if path.is_file():
        return path
    text = (BASE.RDDL_DIR / f"instance_{size}_h6.rddl").read_text(encoding="utf-8")
    text = text.replace(f"_{size}_h6", f"_{size}_h{horizon}")
    text, count = re.subn(r"horizon\s*=\s*6\s*;", f"horizon = {horizon};", text)
    if count != 1:
        raise ValueError("Expected one horizon declaration in reference instance.")
    target = output / "inputs" / f"instance_{size}_h{horizon}.rddl"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return target


def _run(instance: Path, size: int, horizon: int, delta: float, trial: int,
         method: str, fraction: float, args: argparse.Namespace) -> dict:
    """Same entry point, recorded failures, replay outside timing. / 同一入口，保留失败，回放不计时。"""
    row = dict(version=VERSION, size=size, horizon=horizon, delta=delta, trial=trial,
               seed=args.seed, method=method, alpha=args.alpha, **{"lambda": fraction},
               timeout_s=args.timeout, status="error", time_s=0.)
    started = perf_counter()
    try:
        # Ablation disables only E selection, not scoring or model rebuilds.
        # 消融仅禁用 E 子树删除，保留评分、F 保护及模型重建；不增加求解器公开模式。
        ablation = patch("darp.planning.rank._prune_expanded", return_value=None) if method == "F-only" else nullcontext()
        with ablation:
            result = solve_rddl(BASE.DOMAIN, instance, planner="hilp", seed=args.seed,
                                risk_budget=delta, heuristic=BASE.MANHATTAN,
                                terminal_heuristic=True, timeout_s=args.timeout,
                                rank_alpha=args.alpha, rank_lambda=fraction)
        decision, policy = result.decision, result.decision.policy
        valid = (policy.duration_complete and policy.feasible is True
                 and policy.achieved_utility is not None
                 and (method != "HILP" or decision.complete))
        row.update(status="ok" if valid else "incomplete", time_s=result.elapsed_s,
                   objective=-policy.achieved_utility if valid else "",
                   risk=policy.active_constraint_value, complete=decision.complete,
                   histories=decision.timing["expanded_nodes"] + decision.timing["frontier_nodes"])
        row.update({name: decision.timing[name] for name in TIMINGS})
        if trial == 1:
            label = f"{method}-{size}-h{horizon}-d{delta:g}-a{args.alpha:g}-l{fraction:g}"
            path = args.output.parent / "results" / args.output.stem / f"{label}.json"
            result.save(path)
            row["result_file"] = str(path.relative_to(args.output.parent))
            if valid and args.episodes:
                restored = DARPResult.load(path)
                env = load_rddl(BASE.DOMAIN, instance).env
                try:
                    statistics = PolicyExecutor(restored.decision.policy).evaluate(env, episodes=args.episodes, seed=args.seed)
                finally:
                    env.close()
                row.update({name: statistics[name] for name in (
                    "episodes", "risk_rate", "physical_duration_mean", "rollout_time_s")})
    except Exception as exc:
        logging.exception("Experiment failed: %s", row)
        row.update(status="timeout" if isinstance(exc, TimeoutError) else "error",
                   error=f"{type(exc).__name__}: {exc}")
        if not row["time_s"]:
            row["time_s"] = perf_counter() - started
    return row


def _metadata(args: argparse.Namespace, existing_rows: int) -> None:
    """Record runtime settings, never credentials. / 记录运行设置，不记录许可证凭据。"""
    import gurobipy
    import importlib.metadata
    data = {"algorithm": VERSION, "python": platform.python_version(),
            "platform": platform.platform(), "gurobi": gurobipy.gurobi.version(),
            "pyRDDLGym": importlib.metadata.version("pyRDDLGym"), "MIPGap": 1e-6,
            "FeasibilityTol": 1e-6, "threads": "Gurobi default",
            "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}}
    from datetime import datetime, timezone
    path = args.output.with_suffix(".metadata.json")
    # Retain the initial batch when adding ablations later. / 补跑消融时保留首批设置。
    previous = json.loads(path.read_text(encoding="utf-8")) if existing_rows and path.is_file() else {}
    batches = previous.get("batches", [])
    if previous and not batches:
        batches.append({"arguments": previous.get("arguments", {}), "existing_rows": 0})
    batches.append({"started_at_utc": datetime.now(timezone.utc).isoformat(),
                    "arguments": data["arguments"], "existing_rows": existing_rows})
    data["batches"] = batches
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _summary(rows: list[dict], historical: list[dict], output: Path) -> None:
    """Write the LaTeX results table and pruning diagnostics. / 只输出 LaTeX 结果表与剪枝诊断。"""
    grouped: dict[tuple, list[dict]] = {}
    for row in rows:
        size, horizon, delta, _, method, fraction = _key(row)
        grouped.setdefault((size, horizon, delta, method, fraction), []).append(row)
    output.parent.mkdir(parents=True, exist_ok=True)
    _latex(grouped, historical, output)


def _latex(grouped: dict, historical: list[dict], output: Path) -> None:
    """Group lambda columns and split pruning modes into legible paper tables.
    / 按 lambda 分组列，E+F 与 F-only 分页呈现；百分比基于同配置 HILP。
    """
    scenarios = sorted({key[:3] for key in grouped})
    modes = [method for method in ("Rank", "F-only") if any(key[3] == method for key in grouped)]
    labels = {"Rank": "prun-HILP (E+F)", "F-only": "prun-HILP (F-only)"}

    def number(group, field):
        if not group or any(row.get("status", "ok") != "ok" for row in group):
            return None
        return median(float(row[field]) for row in group)

    def paired(base, group):
        return (bool(base) and bool(group)
                and {r["trial"] for r in base} == {r["trial"] for r in group})

    def cell(group, field, baseline=None):
        value = number(group, field)
        if value is None:
            return "--"
        text = f"{value:.4f}" if field == "objective" else (
            f"{value:.3f}" if field == "time_s" else f"{value:.0f}")
        base = number(baseline, field) if baseline and paired(baseline, group) else None
        if base is not None and field == "objective" and abs(value - base) > 1e-6:
            text = r"\textbf{" + text + "}"
        if base is not None and field == "time_s" and value < base:
            text = r"\underline{" + text + "}"
        return text

    alpha = sorted({float(r["alpha"]) for g in grouped.values() for r in g})
    trials = sorted({int(r["trial"]) for g in grouped.values() for r in g})
    timeouts = sorted({float(r["timeout_s"]) for g in grouped.values() for r in g})
    episodes = sorted({int(r["episodes"]) for g in grouped.values() for r in g if r.get("episodes")})
    lines = [r"\documentclass[10pt]{article}",
             r"\usepackage[a4paper,landscape,margin=12mm]{geometry}",
             r"\usepackage{booktabs,multirow,graphicx,amsmath}",
             r"\pagestyle{plain}\begin{document}"]
    for size, mode in product(sorted({s[0] for s in scenarios}), modes):
        if len(lines) > 4:
            lines.append(r"\clearpage")
        fractions = sorted({key[4] for key in grouped if key[0] == size and key[3] == mode})
        lines += [rf"\begin{{center}}\Large $ {size}\times{size}$ grid: HILP versus {labels[mode]}\end{{center}}",
                  rf"\noindent $\alpha={','.join(f'{a:g}' for a in alpha)}$; "
                  r"$\lambda$ is the retained fraction. Fixed duration $=1$; Manhattan heuristic. "
                  rf"{len(trials)} search trials per complete configuration; median wall time includes filtering, model construction and optimization.\par",
                  r"\noindent \underline{Underlined time}: lower than the recorded HILP baseline. "
                  r"\textbf{Bold Obj.}: differs from HILP; lower cost is better. "
                  r"RAO* uses historical measurements and a different native objective boundary.\par",
                  r"\begin{table}[ht]\centering",
                  rf"\caption{{Simulation results with heuristics: {labels[mode]}.}}",
                  r"\scriptsize\setlength{\tabcolsep}{3pt}",
                  r"\resizebox{\textwidth}{!}{\begin{tabular}{cc" + "rrrr" * (2 + len(fractions)) + "}",
                  r"\toprule",
                  " & ".join([r"\multirow{2}{*}{$h$}", r"\multirow{2}{*}{$\Delta$}",
                              r"\multicolumn{4}{c}{HILP}",
                              r"\multicolumn{4}{c}{RAO* (historical)}"] +
                             [rf"\multicolumn{{4}}{{c}}{{{labels[mode]} ($\lambda={f:g}$)}}" for f in fractions]) + r" \\"]
        lines.append(" & & " + " & ".join(["Obj.", "Time (s)", "$n$", "Iter."] * (2 + len(fractions))) + r" \\")
        for i in range(2 + len(fractions)):
            lines.append(rf"\cmidrule(lr){{{3+4*i}-{6+4*i}}}")
        for horizon in sorted({s[1] for s in scenarios if s[0] == size}):
            deltas = sorted({s[2] for s in scenarios if s[:2] == (size, horizon)})
            for j, delta in enumerate(deltas):
                scenario = (size, horizon, delta)
                base = grouped.get((*scenario, "HILP", 1.), [])
                old = [r for r in historical if r["algorithm"] == "RAO*" and
                       (int(r["size"]), int(r["horizon"]), float(r["delta"])) == scenario]
                entries = [rf"\multirow{{{len(deltas)}}}{{*}}{{{horizon}}}" if j == 0 else "", f"{delta:g}"]
                entries += [cell(base, f) for f in ("objective", "time_s", "histories", "partial_ilp_solves")]
                entries += [cell(old, f) for f in ("objective", "time_s", "n", "iterations")]
                for fraction in fractions:
                    group = grouped.get((*scenario, mode, fraction), [])
                    entries += [cell(group, f, base)
                                for f in ("objective", "time_s", "histories", "partial_ilp_solves")]
                lines.append(" & ".join(entries) + r" \\")
            if horizon != max(s[1] for s in scenarios if s[0] == size):
                lines.append(r"\midrule")
        lines += [r"\bottomrule\end{tabular}}\end{table}",
                  r"\noindent\small DARP $n=|E\cup F|$ counts generated action histories; RAO* $n$ counts belief graph nodes. "
                  r"Iter. counts p-ILP calls for DARP and expansions for RAO*. "
                  r"An executable pruned policy does not certify global optimality; failed groups are --.\par",
                  r"\medskip\noindent\textbf{Pruning diagnostics.}\par",
                  r"\begin{center}\small\begin{tabular}{crrrrrrr}\toprule",
                  r"$\lambda$ & Success/runs & Speedup & Max. cost increase (\%) & E sent (\%) & F sent (\%) & Filter (s) & Fallbacks \\ \midrule"]
        for fraction in fractions:
            data = [r for key, g in grouped.items() if key[0] == size and key[3:] == (mode, fraction) for r in g]
            good = [r for r in data if r["status"] == "ok"]
            speeds, relative = [], []
            for scenario in (s for s in scenarios if s[0] == size):
                base, group = grouped.get((*scenario, "HILP", 1.), []), grouped.get((*scenario, mode, fraction), [])
                if number(base, "time_s") and number(group, "time_s") and paired(base, group):
                    speeds.append(number(base, "time_s") / number(group, "time_s"))
                    base_cost = number(base, "objective")
                    if abs(base_cost) > 1e-12:
                        relative.append(100 * (number(group, "objective") - base_cost) / abs(base_cost))
            def sent(part):
                total = sum(float(r[f"ilp_{part}_full_total"]) for r in good)
                return 100 * sum(float(r[f"ilp_{part}_sent_total"]) for r in good) / total if total else None
            time_text = rf"{median(speeds):.2f}$\times$" if speeds else "--"
            if speeds and median(speeds) > 1:
                time_text = r"\underline{" + time_text + "}"
            e, f = sent("e"), sent("f")
            entries = [f"{fraction:g}", f"{len(good)}/{len(data)}", time_text,
                       f"{max(relative):+.2f}" if relative else "--",
                       f"{e:.1f}" if e is not None else "--", f"{f:.1f}" if f is not None else "--",
                       f"{median(float(r['rank_filter_ms']) for r in good)/1000:.3f}" if good else "--",
                       str(sum(int(float(r["rank_fallbacks"])) for r in good))]
            lines.append(" & ".join(entries) + r" \\")
        lines += [r"\bottomrule\end{tabular}\end{center}",
                  r"\noindent\small Cost increase is $100(C_{\rm pruned}-C_{\rm HILP})/|C_{\rm HILP}|$; "
                  r"the diagnostic column reports the largest configuration-level increase, using median costs. "
                  r"Speedup is the median configuration-level time ratio. E/F sent ratios include fallback submissions.\par",
                  r"\noindent\small F-only skips E-subtree pruning; necessary dead-end, prefix-risk and unreachable-history "
                  r"filtering still applies. Submitted coefficients and global risk constraints are preserved.\par",
                  r"\noindent\small HILP and E+F reuse the completed initial batch; F-only was added in a supplementary "
                  r"serial batch. Underlining shows observed differences, not statistical significance. "
                  rf"MIPGap $=10^{{-6}}$; search limit $={','.join(f'{t:g}' for t in timeouts)}$ s. "
                  + (f"Replay ({','.join(map(str, episodes))} episodes for trial 1) is outside search timing." if episodes
                     else "No recorded policy replay.") + r"\par"]
    lines.append(r"\end{document}")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
