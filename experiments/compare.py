"""Select existing runs or run new inputs, then export LaTeX/PDF. / 选择已有结果或补跑输入后生成表格。"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.run import ALGORITHMS, path_label, run_batch, safe_name


def choose(items: list, prompt: str, *, multiple: bool = True) -> list:
    """Numbered terminal selection with validation. / 校验终端编号选择。"""
    if not items:
        return []
    for index, item in enumerate(items, 1):
        print(f"  {index}. {item}")
    while True:
        answer = input(prompt + (" [编号，逗号分隔；all 全选]: " if multiple else " [编号]: ")).strip()
        try:
            indices = list(range(len(items))) if answer == "all" and multiple else [int(v.strip()) - 1 for v in answer.split(",")]
            if not indices or min(indices) < 0 or max(indices) >= len(items) or (not multiple and len(indices) != 1):
                raise ValueError
            return [items[i] for i in dict.fromkeys(indices)]
        except ValueError:
            print("请输入有效编号。")


def interactive() -> tuple[str, list[Path]]:
    """Select reusable runs; explicitly confirm any new solve. / 复用已有结果，补跑须显式确认。"""
    scenes = sorted({p.parent.parent.name for p in (ROOT / "experiments").glob("*/runs/*")}
                    | {p.name for p in (ROOT / "benchmarks").iterdir() if p.is_dir() and not p.name.startswith("_")})
    scene = choose(scenes + ["指定其他场景"], "选择场景", multiple=False)[0]
    if scene == "指定其他场景":
        scene = safe_name(input("场景名称: ").strip())
    available = sorted((ROOT / "experiments" / scene / "runs").glob("*/results.csv"))
    selected = choose([str(p.parent.relative_to(ROOT)) for p in available], "选择已有运行批次") if available else []
    if not selected or input("需要补跑其他实验吗？[y/N]: ").strip().lower() == "y":
        source = input("输入来源：1=benchmarks，2=外部 RDDL [1]: ").strip() or "1"
        if source == "1":
            instances = sorted(p for p in (ROOT / "benchmarks" / scene).rglob("*.rddl") if p.name != "domain.rddl")
            instances = [Path(p) for p in choose([str(p.relative_to(ROOT)) for p in instances], "选择 instance")]
            instances = [ROOT / p for p in instances]
            domain = None
        elif source == "2":
            domain = Path(input("domain.rddl 路径: ").strip()).expanduser().resolve()
            instances = [Path(input("instance.rddl 路径: ").strip()).expanduser().resolve()]
        else:
            raise ValueError("Input source must be 1 or 2.")
        methods = choose(list(ALGORITHMS if scene == "grid" and source == "1" else ("HILP", "FullILP", "pruningF", "pruningEF")), "选择算法")
        name = safe_name(input("新运行批次名称: ").strip())
        alpha = float(input("alpha [1]: ").strip() or "1")
        fractions = [float(v) for v in (input("lambda，逗号分隔 [0.5]: ").strip() or ".5").split(",")]
        trials = int(input("每配置运行次数 [1]: ").strip() or "1")
        timeout = float(input("单次求解时限（秒）[120]: ").strip() or "120")
        heuristic = None
        terminal = False
        if source == "2":
            heuristic = input("heuristic module:object（留空使用默认）: ").strip() or None
            terminal = bool(heuristic) and input("终端计入 heuristic？[y/N]: ").strip().lower() == "y"
            if scene == "grid":
                raise ValueError("External RDDL must use a distinct scene name to avoid applying grid defaults.")
        count = len(instances) * trials * sum(len(fractions) if m.startswith("pruning") else 1 for m in methods)
        print(f"将运行 {count} 次求解，保存到 experiments/{scene}/runs/{name}")
        if input("确认开始求解？[y/N]: ").strip().lower() == "y":
            output = run_batch(scene=scene, name=name, instances=instances, domain=domain,
                               algorithms=methods, alpha=alpha, lambdas=fractions, trials=trials,
                               timeout=timeout, heuristic=heuristic, terminal_heuristic=terminal)
            selected.append(str(output))
    return scene, [Path(p) if Path(p).is_absolute() else ROOT / p for p in selected]


def escape(value) -> str:
    """Escape user labels for LaTeX. / 转义用户输入的 LaTeX 特殊字符。"""
    mapping = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
               "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(mapping.get(c, c) for c in str(value))


def read_runs(paths: list[Path], scene: str) -> tuple[list[dict], dict]:
    """Validate provenance and keep different batches separate. / 校验来源，不混合不同批次。"""
    rows, configs = [], {}
    for path in paths:
        path = path.resolve()
        label = path_label(path)
        if label in configs:
            raise ValueError(f"Duplicate run: {path}")
        config = json.loads((path / "config.json").read_text())
        if config["scene"] != scene:
            raise ValueError(f"Different scene in {path}")
        configs[label] = config
        with (path / "results.csv").open(newline="") as stream:
            values = list(csv.DictReader(stream))
        seen = set()
        for row in values:
            key = (row["input_digest"], row["algorithm"], float(row["alpha"]), float(row["lambda"]), int(row["trial"]))
            if key in seen:
                raise ValueError(f"Duplicate configuration/trial in {path}: {key}")
            seen.add(key)
            row["batch"] = label
            rows.append(row)
    return rows, configs


def timing_signature(config: dict):
    """Unknown legacy settings cannot certify timing comparability. / 旧数据缺少设置时不认定计时可比。"""
    keys = ("heuristic", "terminal_heuristic", "timing_scope", "MIPGap", "threads", "timeout_s", "python", "platform", "gurobi")
    if any(config.get(key) is None for key in keys):
        return None
    return json.dumps({k: config[k] for k in keys}, sort_keys=True)


def write_report(rows: list[dict], configs: dict, scene: str, name: str,
                 *, pdf: bool = True, overwrite: bool = False) -> Path:
    """Reusable result-only reporting with explicit missing/failure cells. / 仅汇总已有结果，明确失败及缺失值。"""
    if not rows:
        raise ValueError("No selected result rows.")
    safe_name(name)
    methods = [a for a in ALGORITHMS if any(r["algorithm"] == a for r in rows)]
    output = ROOT / "experiments/output" / (safe_name(scene) + "-" + "-".join(methods))
    tex = output / f"{name}.tex"
    selection = output / f"{name}.selection.json"
    if not overwrite and any(p.exists() for p in (tex, tex.with_suffix(".pdf"), selection)):
        raise FileExistsError(f"Report exists: {tex}; use another --name or --overwrite.")
    grouped = defaultdict(list)
    labels = {}
    for row in rows:
        variant = (row["algorithm"], float(row["alpha"]), float(row["lambda"]), row["batch"])
        grouped[row["input_digest"], variant].append(row)
        labels[row["input_digest"]] = Path(row["instance"]).parent.name + "/" + row["case"]
    cases = sorted(labels, key=labels.get)
    variants = sorted({key[1] for key in grouped}, key=lambda v: (ALGORITHMS.index(v[0]), v[1:]))
    def value(group, field):
        if not group or any(r["status"] != "ok" or r.get(field, "") == "" for r in group):
            return None
        numbers = [float(r[field]) for r in group]
        return median(numbers) if all(math.isfinite(x) for x in numbers) else None
    def fmt(number, digits=3):
        return "--" if number is None else f"{number:.{digits}f}"
    lines = [r"\documentclass[10pt]{article}", r"\usepackage[a4paper,landscape,margin=14mm]{geometry}",
             r"\usepackage{booktabs,array,amsmath}", r"\begin{document}"]
    # Two variants per panel keep readable type without scaling whole tables.
    # 每面板最多两个算法配置，保持字号，不整体缩小表格。
    for offset in range(0, len(variants), 2):
        panel = variants[offset:offset + 2]
        for start in range(0, len(cases), 24):
            if len(lines) > 4:
                lines.append(r"\clearpage")
            lines += [r"\section*{" + escape(scene + ": " + name) + "}",
                      r"\noindent Median over recorded trials; cost is negative expected utility. Missing or unsuccessful results: --.\par",
                      r"\small\begin{tabular}{p{65mm}" + "rrrrr" * len(panel) + "}", r"\toprule"]
            titles = []
            for algorithm, alpha, fraction, batch in panel:
                label = algorithm + (f" (a={alpha:g}, l={fraction:g})" if algorithm.startswith("pruning") else "")
                titles.append(r"\multicolumn{5}{c}{" + escape(label) + "}")
            lines += ["Input & " + " & ".join(titles) + r" \\",
                      " & " + " & ".join([r"Cost & Risk & Time (s) & Vars & $\Delta$cost\%"] * len(panel)) + r" \\", r"\midrule"]
            for case in cases[start:start + 24]:
                bases = [v for v in variants if v[0] == "HILP" and (case, v) in grouped]
                base = grouped[case, bases[0]] if len(bases) == 1 else []
                base_cost = value(base, "objective")
                cells = []
                for variant in panel:
                    group = grouped.get((case, variant), [])
                    cost, risk, elapsed, count = [value(group, f) for f in ("objective", "risk", "time_s", "ilp_variables")]
                    cost_text = fmt(cost, 4)
                    comparable_cost = variant[0] != "RAOstar" and not configs[variant[3]].get("unverified_legacy_inputs", False)
                    increase = None if not comparable_cost or cost is None or base_cost in (None, 0) else 100 * (cost - base_cost) / abs(base_cost)
                    if comparable_cost and cost is not None and base_cost is not None and abs(cost - base_cost) > 1e-6:
                        cost_text = r"\textbf{" + cost_text + "}"
                    time_text = fmt(elapsed)
                    signature = timing_signature(configs[variant[3]])
                    paired = (variant[0] != "RAOstar" and base and signature is not None and signature == timing_signature(configs[bases[0][3]])
                              and {(r["trial"], r["seed"]) for r in base} == {(r["trial"], r["seed"]) for r in group})
                    base_time = value(base, "time_s")
                    if paired and elapsed is not None and base_time is not None and elapsed < base_time:
                        time_text = r"\underline{" + time_text + "}"
                    cells.extend([cost_text, fmt(risk), time_text, fmt(count, 0), fmt(increase, 2)])
                # Permit line breaks after the variant directory and filename hyphens.
                # 在类型目录和文件名连字符处允许换行。
                label = escape(labels[case]).replace("/", r"/\allowbreak ").replace("-", r"-\allowbreak ")
                lines.append(label + " & " + " & ".join(cells) + r" \\")
            lines += [r"\bottomrule\end{tabular}\par", r"\medskip\footnotesize",
                      r"Bold: cost differs from the unique HILP baseline. Underline: lower time with matching recorded settings and trial/seed pairs. "
                      r"$\Delta$cost = $100(C-C_{HILP})/|C_{HILP}|$; undefined for zero/missing/ambiguous baseline. "
                      r"Historical or mismatched timing is descriptive only; no speedup claim is implied.\par"]
            if "RAOstar" in methods:
                lines.append(r"RAOstar reports its native boundary objective, not the DARP terminal replacement; no cost increase is computed for RAOstar.\par")
            for variant in panel:
                trials = [r for r in rows if (r["algorithm"], float(r["alpha"]), float(r["lambda"]), r["batch"]) == variant]
                lines.append(escape(f"{variant[0]}: {variant[3]}; {len(trials)} rows, {sum(r['status'] != 'ok' for r in trials)} unsuccessful.") + r"\par")
    lines.append(r"\end{document}")
    output.mkdir(parents=True, exist_ok=True)
    tex.write_text("\n".join(lines) + "\n")
    selection.write_text(json.dumps({"scene": scene, "batches": configs,
                                     "inputs": cases, "variants": variants,
                                     "aggregation": "median", "rows": len(rows)}, indent=2) + "\n")
    if pdf:
        compile_pdf(tex)
    return tex


def compile_pdf(tex: Path) -> Path:
    """Keep compiler intermediates out of tracked output. / 编译缓存放临时目录。"""
    executable = shutil.which("pdflatex")
    if not executable:
        raise RuntimeError(f"pdflatex is unavailable; source was saved at {tex}")
    with tempfile.TemporaryDirectory(prefix="darp-table-") as temporary:
        for _ in range(2):
            completed = subprocess.run([executable, "-no-shell-escape", "-interaction=nonstopmode", "-halt-on-error",
                                        "-output-directory", temporary, str(tex.resolve())], capture_output=True, text=True)
            if completed.returncode:
                raise RuntimeError("LaTeX compilation failed:\n" + completed.stdout[-4000:])
        shutil.copy2(Path(temporary) / tex.with_suffix(".pdf").name, tex.with_suffix(".pdf"))
    return tex.with_suffix(".pdf")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+")
    parser.add_argument("--scene", default="grid")
    parser.add_argument("--name", default="comparison")
    parser.add_argument("--cases", nargs="+", help="optional exact instance stems")
    parser.add_argument("--tex-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    scene, paths = (args.scene, args.runs) if args.runs else interactive()
    if not paths:
        print("未选择实验结果。")
        return 0
    rows, configs = read_runs(paths, scene)
    if args.cases:
        rows = [r for r in rows if r["case"] in args.cases]
    elif not args.runs:
        cases = sorted({r["instance"] for r in rows})
        selected = set(choose(cases, "选择比较的配置"))
        rows = [r for r in rows if r["instance"] in selected]
        args.name = safe_name(input("表格名称 [comparison]: ").strip() or "comparison")
    print(write_report(rows, configs, scene, args.name, pdf=not args.tex_only, overwrite=args.overwrite))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
