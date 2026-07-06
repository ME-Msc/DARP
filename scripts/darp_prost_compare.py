"""Reusable DARP-heuristics-vs-PROST comparison runner.

This module owns process orchestration, log parsing, metric aggregation, and
terminal table rendering. Scenario scripts only provide RDDL paths and any
scenario-specific PROST state parser.
"""

# TODO(phase-9.2): Promote this experiment runner into the package once benchmark APIs stabilize.

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROST_ROOT = Path(os.environ.get("PROST_ROOT", str(REPO_ROOT.parent / "prost-planner")))
DEFAULT_RDDLSIM_ROOT = Path(os.environ.get("RDDLSIM_ROOT", str(REPO_ROOT.parent / "rddlsim")))
DEFAULT_PROST_PYTHON = Path(os.environ.get("PROST_PYTHON", sys.executable))

ProstStateTraceParser = Callable[[str, tuple[str, ...]], tuple[str, ...]]

RUN_CSV_COLUMNS = (
    "scenario",
    "seed",
    "system",
    "variant",
    "planner",
    "heuristic",
    "domain",
    "instance",
    "duration",
    "total_reward",
    "turns",
    "runtime_s",
    "decision_ms",
    "planner_elapsed_ms",
    "rddl_load_ms",
    "grounding_ms",
    "and_or_interface_ms",
    "initial_belief_ms",
    "frontier_expand_ms",
    "heuristic_eval_ms",
    "ilp_encode_ms",
    "tree_ilp_build_ms",
    "gurobi_call_ms",
    "postprocess_ms",
    "expanded_nodes",
    "performed_trials",
    "ilp_vars",
    "ilp_constraints",
    "gurobi_ms",
    "prost_parsing_ms",
    "prost_instantiating_ms",
    "prost_simplifying_ms",
    "prost_analyzing_ms",
    "risk_budget",
    "constraint_violation",
    "returncode",
    "actions",
    "states",
)

SUMMARY_GROUP_COLUMNS = ("scenario", "system", "variant", "planner", "heuristic")
SUMMARY_NUMERIC_COLUMNS = (
    "total_reward",
    "turns",
    "runtime_s",
    "decision_ms",
    "planner_elapsed_ms",
    "grounding_ms",
    "expanded_nodes",
    "performed_trials",
    "ilp_vars",
    "ilp_constraints",
    "gurobi_ms",
)


@dataclass(frozen=True)
class DARPProstExperimentSpec:
    """Describe one DARP-vs-PROST comparison experiment."""

    name: str
    description: str
    domain_path: Path
    instance_path: Path
    duration_path: Path | None
    prost_instance_name: str
    default_output_dir: Path
    prost_state_trace_parser: ProstStateTraceParser | None = None
    prost_state_note: str = "PROST state trace is not parsed for this scenario"
    default_planner: str = "hilp"
    default_heuristic_lookahead_depth: int = 4
    default_expansion_rounds: int | None = None
    default_frontier_width: int = 1
    default_hilp_heuristics: tuple[str, ...] = ("reachable-bellman",)
    default_prost_config: str = "[Prost -s {seed} -se [IPC2014]]"


@dataclass(frozen=True)
class DARPVariantSummary:
    """Store one DARP variant's compact comparison result."""

    label: str
    heuristic: str
    result: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly variant summary."""
        return {
            "label": self.label,
            "heuristic": self.heuristic,
            **self.result,
        }


@dataclass(frozen=True)
class RunSummary:
    """Store one seed's compact comparison result."""

    seed: int
    darp_variants: tuple[DARPVariantSummary, ...]
    prost_reward: float | None
    prost_round_rewards: tuple[float, ...]
    prost_turns: tuple[int, ...]
    prost_actions: tuple[str, ...]
    prost_state_trace: tuple[str, ...]
    prost_runtime_seconds: float | None
    prost_search_ms: float | None
    prost_metrics: dict[str, Any]
    darp_returncode: int | None
    prost_returncode: int | None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly summary row."""
        return {
            "seed": self.seed,
            "darp_variants": [variant.to_dict() for variant in self.darp_variants],
            "prost_reward": self.prost_reward,
            "prost_round_rewards": list(self.prost_round_rewards),
            "prost_turns": list(self.prost_turns),
            "prost_actions": list(self.prost_actions),
            "prost_state_trace": list(self.prost_state_trace),
            "prost_runtime_seconds": self.prost_runtime_seconds,
            "prost_search_ms": self.prost_search_ms,
            "prost_metrics": self.prost_metrics,
            "darp_returncode": self.darp_returncode,
            "prost_returncode": self.prost_returncode,
        }


def run_experiment(spec: DARPProstExperimentSpec, argv: list[str] | None = None) -> int:
    """Run one configured DARP-vs-PROST comparison experiment."""
    args = build_parser(spec).parse_args(argv)
    seeds = tuple(_parse_seeds(args.seeds))
    darp_variants = _darp_variants(args)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[RunSummary] = []
    for seed in seeds:
        run_dir = output_dir / f"seed_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        if args.open_terminals:
            if len(seeds) != 1:
                raise ValueError("--open-terminals supports exactly one seed.")
            clear_open_terminal_outputs(args, run_dir)
            started_at = time.monotonic()
            open_experiment_terminals(spec, args, seed, run_dir)
            summaries.append(wait_for_open_terminal_summary(spec, args, seed, run_dir, started_at, darp_variants))
            continue
        darp_results = () if args.skip_darp else run_darp_variants(spec, args, seed, run_dir, darp_variants)
        prost_result = None if args.skip_prost else run_prost(spec, args, seed, run_dir)
        summaries.append(summary_from_results(seed, darp_results, prost_result))

    artifacts = write_experiment_artifacts(spec, summaries, output_dir)
    print_summary(spec, summaries, artifacts["summary_json"])
    return 0


def write_experiment_artifacts(
    spec: DARPProstExperimentSpec,
    summaries: list[RunSummary],
    output_dir: Path,
) -> dict[str, Path]:
    """Write JSON and CSV files for one experiment."""
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_json = output_dir / "summary.json"
    summary_json.write_text(
        json.dumps([summary.to_dict() for summary in summaries], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    run_rows = run_rows_from_summaries(spec, summaries)
    runs_csv = output_dir / "runs.csv"
    write_csv_rows(runs_csv, run_rows, RUN_CSV_COLUMNS)
    summary_csv = output_dir / "summary.csv"
    write_csv_rows(summary_csv, summary_rows_from_run_rows(run_rows), summary_csv_columns())
    return {
        "summary_json": summary_json,
        "runs_csv": runs_csv,
        "summary_csv": summary_csv,
    }


def build_parser(spec: DARPProstExperimentSpec) -> argparse.ArgumentParser:
    """Build a comparison-script parser from an experiment spec."""
    parser = argparse.ArgumentParser(description=spec.description)
    parser.add_argument("--seeds", default="0", help="comma-separated seeds, e.g. 0,1,2")
    parser.add_argument(
        "--output-dir",
        default=str(spec.default_output_dir),
        help="directory for raw logs and summary JSON",
    )
    parser.add_argument("--skip-darp", action="store_true", help="do not run DARP")
    parser.add_argument("--skip-prost", action="store_true", help="do not run PROST/rddlsim")
    parser.add_argument(
        "--open-terminals",
        action="store_true",
        help="open terminal windows for DARP, rddlsim server, and PROST client instead of running headless",
    )
    parser.add_argument(
        "--darp-python",
        default=sys.executable,
        help="Python executable used to run `python -m darp`",
    )
    parser.add_argument(
        "--prost-python",
        default=str(DEFAULT_PROST_PYTHON),
        help="Python executable used to run PROST helper scripts",
    )
    parser.add_argument("--prost-root", default=str(DEFAULT_PROST_ROOT), help="PROST checkout")
    parser.add_argument(
        "--rddlsim-root",
        default=str(DEFAULT_RDDLSIM_ROOT),
        help="rddlsim checkout",
    )
    parser.add_argument("--port", type=int, default=2323, help="rddlsim server port")
    parser.add_argument("--host", default="127.0.0.1", help="rddlsim host")
    parser.add_argument("--timeout", type=float, default=120.0, help="per-command timeout in seconds")
    parser.add_argument("--planner", default=spec.default_planner, help="DARP planner name")
    parser.add_argument(
        "--heuristic-lookahead-depth",
        type=int,
        default=spec.default_heuristic_lookahead_depth,
        help="DARP reachable-bellman heuristic future Bellman layers",
    )
    parser.add_argument(
        "--expansion-rounds",
        type=int,
        default=spec.default_expansion_rounds,
        help="optional HILP frontier expansion budget; omit for horizon-bounded exhaustive refinement",
    )
    parser.add_argument("--frontier-width", type=int, default=spec.default_frontier_width, help="DARP HILP frontier width")
    parser.add_argument(
        "--hilp-heuristics",
        default=",".join(spec.default_hilp_heuristics),
        help="comma-separated DARP HILP frontier heuristic modes",
    )
    parser.add_argument(
        "--prost-config",
        default=spec.default_prost_config,
        help="PROST config string; {seed} is replaced per run",
    )
    return parser


def _darp_variants(args: argparse.Namespace) -> tuple[dict[str, str], ...]:
    """Return configured DARP heuristic variants."""
    allowed = {"one-step-greedy", "reachable-bellman"}
    heuristics = tuple(piece.strip() for piece in args.hilp_heuristics.split(",") if piece.strip())
    if not heuristics:
        raise ValueError("--hilp-heuristics must name at least one heuristic mode.")
    unknown = sorted(set(heuristics) - allowed)
    if unknown:
        raise ValueError(f"Unsupported HILP heuristic modes: {', '.join(unknown)}")
    return tuple(
        {
            "heuristic": heuristic,
            "key": _variant_key(heuristic),
            "label": f"DARP-{heuristic}",
        }
        for heuristic in heuristics
    )


def _variant_key(value: str) -> str:
    """Return a filesystem-safe key for one variant."""
    return re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_")


def open_experiment_terminals(
    spec: DARPProstExperimentSpec,
    args: argparse.Namespace,
    seed: int,
    run_dir: Path,
) -> None:
    """Open terminal tabs that run the comparison commands visibly."""
    terminal = shutil.which("gnome-terminal")
    if terminal is None:
        raise RuntimeError("`gnome-terminal` was not found; run without --open-terminals or install it.")

    if not args.skip_darp:
        for variant in _darp_variants(args):
            key = variant["key"]
            darp_script = run_dir / f"run_darp_{key}.sh"
            _write_shell_script(
                darp_script,
                [
                    "set -e",
                    f"cd {shlex.quote(str(REPO_ROOT))}",
                    "export PYTHONPATH=src",
                    f"date +%s.%N > {shlex.quote(str(run_dir / f'darp_{key}_started_at.txt'))}",
                    _shell_command(_darp_command(spec, args, seed, run_dir, variant))
                    + f" 2>&1 | tee {shlex.quote(str(run_dir / f'darp_{key}_terminal.log'))}",
                    f"date +%s.%N > {shlex.quote(str(run_dir / f'darp_{key}_finished_at.txt'))}",
                    "echo",
                    f"echo 'DARP {variant['heuristic']} finished. Press Enter to close this terminal.'",
                    "read",
                ],
            )
            _open_terminal(terminal, f"DARP {variant['heuristic']}", darp_script)

    if not args.skip_prost:
        prost_root = Path(args.prost_root).resolve()
        rddlsim_root = Path(args.rddlsim_root).resolve()
        benchmark_dir = run_dir / "prost_rddl"
        log_dir = run_dir / "prost_rddlsim_logs"
        prost_work_dir = _prepare_prost_client_workdir(args, run_dir)
        _prepare_benchmark(spec, benchmark_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        server_script = run_dir / "run_prost_server.sh"
        client_script = run_dir / "run_prost_client.sh"
        _write_shell_script(
            server_script,
            [
                "set -e",
                f"cd {shlex.quote(str(prost_root / 'testbed'))}",
                f"export RDDLSIM_ROOT={shlex.quote(str(rddlsim_root))}",
                _shell_command(_prost_server_command(spec, args, seed, benchmark_dir, log_dir))
                + f" 2>&1 | tee {shlex.quote(str(run_dir / 'prost_server_terminal.log'))}",
                "echo",
                "echo 'rddlsim server finished. Press Enter to close this terminal.'",
                "read",
            ],
        )
        _write_shell_script(
            client_script,
            [
                "set -e",
                f"cd {shlex.quote(str(prost_work_dir))}",
                "echo 'Waiting for rddlsim server log...'",
                (
                    "while ! grep -q 'RDDL Server Initialized' "
                    f"{shlex.quote(str(run_dir / 'prost_server_terminal.log'))} "
                    "2>/dev/null; do sleep 0.2; done"
                ),
                f"date +%s.%N > {shlex.quote(str(run_dir / 'prost_client_started_at.txt'))}",
                _shell_command(_prost_client_command(spec, args, seed))
                + f" 2>&1 | tee {shlex.quote(str(run_dir / 'prost_client_terminal.log'))}",
                f"date +%s.%N > {shlex.quote(str(run_dir / 'prost_client_finished_at.txt'))}",
                "echo",
                "echo 'PROST client finished. Press Enter to close this terminal.'",
                "read",
            ],
        )
        _open_terminal(terminal, "PROST rddlsim server", server_script)
        _open_terminal(terminal, "PROST client IPC2014", client_script)

    print(f"Opened terminal windows for seed {seed}. Waiting for logs in {run_dir}...")


def run_darp_variants(
    spec: DARPProstExperimentSpec,
    args: argparse.Namespace,
    seed: int,
    run_dir: Path,
    variants: tuple[dict[str, str], ...],
) -> tuple[DARPVariantSummary, ...]:
    """Run all configured DARP heuristic variants."""
    summaries: list[DARPVariantSummary] = []
    for variant in variants:
        result = run_darp(spec, args, seed, run_dir, variant)
        summaries.append(
            DARPVariantSummary(
                label=variant["label"],
                heuristic=variant["heuristic"],
                result=result,
            )
        )
    return tuple(summaries)


def run_darp(
    spec: DARPProstExperimentSpec,
    args: argparse.Namespace,
    seed: int,
    run_dir: Path,
    variant: dict[str, str],
) -> dict[str, Any]:
    """Run DARP and return parsed reward/actions/state metrics."""
    key = variant["key"]
    trace_path = run_dir / f"darp_{key}_trace.json"
    stdout_path = run_dir / f"darp_{key}_stdout.txt"
    stderr_path = run_dir / f"darp_{key}_stderr.txt"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    started_at = time.monotonic()
    timed_out = False
    try:
        completed = subprocess.run(
            _darp_command(spec, args, seed, run_dir, variant),
            cwd=REPO_ROOT,
            env=env,
            text=True,
            capture_output=True,
            timeout=args.timeout,
            check=False,
        )
        stdout = completed.stdout
        stderr = completed.stderr
        returncode = completed.returncode
    except subprocess.TimeoutExpired as error:
        timed_out = True
        stdout = _timeout_output(error.stdout)
        stderr = _timeout_output(error.stderr) + f"\nDARP timed out after {args.timeout} seconds.\n"
        returncode = -1
    runtime_seconds = time.monotonic() - started_at
    stdout_path.write_text(stdout, encoding="utf-8")
    stderr_path.write_text(stderr, encoding="utf-8")
    result = parse_darp_trace(trace_path)
    result["heuristic"] = variant["heuristic"]
    result["returncode"] = returncode
    result["runtime_seconds"] = runtime_seconds
    result["timed_out"] = timed_out
    return result


def run_prost(
    spec: DARPProstExperimentSpec,
    args: argparse.Namespace,
    seed: int,
    run_dir: Path,
) -> dict[str, Any]:
    """Run rddlsim server plus PROST client and parse reward/action metrics."""
    prost_root = Path(args.prost_root).resolve()
    rddlsim_root = Path(args.rddlsim_root).resolve()
    benchmark_dir = run_dir / "prost_rddl"
    log_dir = run_dir / "prost_rddlsim_logs"
    prost_work_dir = _prepare_prost_client_workdir(args, run_dir)
    _prepare_benchmark(spec, benchmark_dir)
    if log_dir.exists():
        shutil.rmtree(log_dir)
    log_dir.mkdir(parents=True)

    server_stdout = run_dir / "prost_server_stdout.txt"
    client_stdout = run_dir / "prost_client_stdout.txt"
    client_stderr = run_dir / "prost_client_stderr.txt"
    server_env = dict(os.environ)
    server_env["RDDLSIM_ROOT"] = str(rddlsim_root)
    with server_stdout.open("w", encoding="utf-8") as server_file:
        prost_started_at = time.monotonic()
        server = subprocess.Popen(
            _prost_server_command(spec, args, seed, benchmark_dir, log_dir),
            cwd=prost_root / "testbed",
            env=server_env,
            text=True,
            stdout=server_file,
            stderr=subprocess.STDOUT,
        )
        try:
            _wait_for_server_log(server_stdout, timeout=min(30.0, args.timeout))
            client_started_at = time.monotonic()
            timed_out = False
            try:
                client = subprocess.run(
                    _prost_client_command(spec, args, seed),
                    cwd=prost_work_dir,
                    text=True,
                    capture_output=True,
                    timeout=args.timeout,
                    check=False,
                )
                client_stdout_text = client.stdout
                client_stderr_text = client.stderr
                client_returncode = client.returncode
            except subprocess.TimeoutExpired as error:
                timed_out = True
                client_stdout_text = _timeout_output(error.stdout)
                client_stderr_text = _timeout_output(error.stderr) + f"\nPROST timed out after {args.timeout} seconds.\n"
                client_returncode = -1
            client_runtime_seconds = time.monotonic() - client_started_at
            client_stdout.write_text(client_stdout_text, encoding="utf-8")
            client_stderr.write_text(client_stderr_text, encoding="utf-8")
            try:
                server.wait(timeout=15.0)
            except subprocess.TimeoutExpired:
                server.terminate()
                server.wait(timeout=10.0)
            parsed = parse_rddlsim_log(log_dir / f"logs-{args.port}.log")
            parsed.update(parse_prost_client_output(spec, client_stdout))
            parsed["returncode"] = client_returncode
            parsed["runtime_seconds"] = client_runtime_seconds
            parsed["timed_out"] = timed_out
            return parsed
        except TimeoutError as error:
            client_stdout.write_text("", encoding="utf-8")
            client_stderr.write_text(str(error) + "\n", encoding="utf-8")
            parsed = parse_rddlsim_log(log_dir / f"logs-{args.port}.log")
            parsed.update(parse_prost_client_output(spec, client_stdout))
            parsed["returncode"] = -1
            parsed["runtime_seconds"] = time.monotonic() - prost_started_at
            parsed["timed_out"] = True
            return parsed
        finally:
            if server.poll() is None:
                server.terminate()
                try:
                    server.wait(timeout=10.0)
                except subprocess.TimeoutExpired:
                    server.kill()


def parse_rddlsim_log(log_path: Path) -> dict[str, Any]:
    """Parse rddlsim XML-fragment logs for compact reward statistics."""
    if not log_path.exists():
        return {"reward": None, "round_rewards": (), "turns": ()}
    text = log_path.read_text(encoding="utf-8", errors="replace")
    round_rewards = tuple(float(value) for value in re.findall(r"<round-reward>([-0-9.]+)</round-reward>", text))
    turns = tuple(int(value) for value in re.findall(r"<turns-used>([0-9]+)</turns-used>", text))
    session_rewards = tuple(float(value) for value in re.findall(r"<total-reward>([-0-9.]+)</total-reward>", text))
    reward = session_rewards[-1] if session_rewards else (sum(round_rewards) if round_rewards else None)
    return {"reward": reward, "round_rewards": round_rewards, "turns": turns}


def _timeout_output(value: str | bytes | None) -> str:
    """Normalize subprocess timeout output."""
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def parse_prost_client_output(spec: DARPProstExperimentSpec, output_path: Path) -> dict[str, Any]:
    """Parse PROST client stdout for action trace, state trace, and search time."""
    if not output_path.exists():
        return {"actions": (), "state_trace": (), "search_ms": None}
    text = output_path.read_text(encoding="utf-8", errors="replace")
    actions = tuple(
        _normalize_prost_action(action)
        for action in re.findall(r"^Submitted action:\s*(.*?)\s*$", text, re.M)
    )
    state_trace = spec.prost_state_trace_parser(text, actions) if spec.prost_state_trace_parser else ()
    search_times = tuple(float(value) for value in re.findall(r"Search time:\s*([0-9.eE+-]+)", text))
    search_ms = sum(search_times) * 1000.0 if search_times else None
    return {
        "actions": actions,
        "state_trace": state_trace,
        "search_ms": search_ms,
        "parser_complete_ms": _seconds_to_ms(_last_float(r"PROST parser complete running time:\s*([0-9.eE+-]+)s?", text)),
        "prost_complete_ms": _seconds_to_ms(_last_float(r"PROST complete running time:\s*([0-9.eE+-]+)", text)),
        "parsing_ms": _seconds_to_ms(_prost_top_level_phase_seconds(text, "Parsing")),
        "instantiating_ms": _seconds_to_ms(_prost_top_level_phase_seconds(text, "Instantiating")),
        "simplifying_ms": _seconds_to_ms(_prost_top_level_phase_seconds(text, "Simplifying")),
        "determinizing_ms": _seconds_to_ms(_prost_top_level_phase_seconds(text, "Determinizing")),
        "analyzing_ms": _seconds_to_ms(_prost_top_level_phase_seconds(text, "Analyzing task")),
        "created_search_nodes": _sum_float_matches(r"Created search nodes:\s*([0-9.eE+-]+)", text),
        "performed_trials": _sum_float_matches(r"Performed trials:\s*([0-9.eE+-]+)", text),
    }


def wait_for_open_terminal_summary(
    spec: DARPProstExperimentSpec,
    args: argparse.Namespace,
    seed: int,
    run_dir: Path,
    started_at: float,
    variants: tuple[dict[str, str], ...],
) -> RunSummary:
    """Wait for terminal-launched runs to write logs, then parse metrics."""
    darp_results: tuple[DARPVariantSummary, ...] = ()
    prost_result = None
    if not args.skip_darp:
        parsed_variants: list[DARPVariantSummary] = []
        for variant in variants:
            key = variant["key"]
            darp_result = _wait_for_darp_trace(run_dir / f"darp_{key}_trace.json", timeout=args.timeout)
            darp_result["heuristic"] = variant["heuristic"]
            darp_result["runtime_seconds"] = _runtime_from_stamp_files(
                run_dir / f"darp_{key}_started_at.txt",
                run_dir / f"darp_{key}_finished_at.txt",
            ) or (time.monotonic() - started_at)
            darp_result["returncode"] = None
            parsed_variants.append(
                DARPVariantSummary(
                    label=variant["label"],
                    heuristic=variant["heuristic"],
                    result=darp_result,
                )
            )
        darp_results = tuple(parsed_variants)
    if not args.skip_prost:
        prost_result = _wait_for_prost_log(run_dir / "prost_rddlsim_logs" / f"logs-{args.port}.log", timeout=args.timeout)
        prost_result.update(
            _wait_for_prost_client_output(
                spec,
                run_dir / "prost_client_terminal.log",
                timeout=args.timeout,
            )
        )
        prost_result["runtime_seconds"] = _runtime_from_stamp_files(
            run_dir / "prost_client_started_at.txt",
            run_dir / "prost_client_finished_at.txt",
        ) or None
        prost_result["returncode"] = None
    return summary_from_results(seed, darp_results, prost_result)


def summary_from_results(
    seed: int,
    darp_results: tuple[DARPVariantSummary, ...],
    prost_result: dict[str, Any] | None,
) -> RunSummary:
    """Build a summary row from optional DARP and PROST result dictionaries."""
    return RunSummary(
        seed=seed,
        darp_variants=darp_results,
        prost_reward=prost_result.get("reward") if prost_result else None,
        prost_round_rewards=tuple(prost_result.get("round_rewards", ())) if prost_result else (),
        prost_turns=tuple(prost_result.get("turns", ())) if prost_result else (),
        prost_actions=tuple(prost_result.get("actions", ())) if prost_result else (),
        prost_state_trace=tuple(prost_result.get("state_trace", ())) if prost_result else (),
        prost_runtime_seconds=prost_result.get("runtime_seconds") if prost_result else None,
        prost_search_ms=prost_result.get("search_ms") if prost_result else None,
        prost_metrics=dict(prost_result) if prost_result else {},
        darp_returncode=None,
        prost_returncode=(
            int(prost_result["returncode"])
            if prost_result and prost_result.get("returncode") is not None
            else None
        ),
    )


def parse_darp_trace(trace_path: Path) -> dict[str, Any]:
    """Parse a DARP JSON trace into comparable metrics."""
    payload = json.loads(trace_path.read_text(encoding="utf-8")) if trace_path.exists() else {}
    steps = tuple(payload.get("steps", ()))
    top_timing = payload.get("timing", {})
    top_timing = top_timing if isinstance(top_timing, dict) else {}
    return {
        "reward": payload.get("total_reward"),
        "actions": tuple(str(step.get("action", "")) for step in steps),
        "state_trace": _state_trace_from_darp_steps(steps),
        "rddl_load_ms": _optional_float(top_timing.get("rddl_load_ms")),
        "duration_load_ms": _optional_float(top_timing.get("duration_load_ms")),
        "grounding_ms": _optional_float(top_timing.get("grounding_ms")),
        "and_or_interface_ms": _optional_float(top_timing.get("and_or_interface_ms")),
        "initial_belief_ms": _optional_float(top_timing.get("initial_belief_ms")),
        "planner_elapsed_ms": _sum_darp_planner_elapsed_ms(steps),
        "decision_ms": _sum_darp_timing(steps, "decision_ms"),
        "tree_ilp_build_ms": _sum_darp_timing(steps, "tree_ilp_build_ms"),
        "frontier_expand_ms": _sum_darp_timing(steps, "frontier_expand_ms"),
        "heuristic_eval_ms": _sum_darp_timing(steps, "heuristic_eval_ms"),
        "ilp_encode_ms": _sum_darp_timing(steps, "ilp_encode_ms"),
        "gurobi_ms": _sum_darp_timing(steps, "gurobi_solve_ms"),
        "gurobi_call_ms": _sum_darp_timing(steps, "gurobi_call_ms"),
        "postprocess_ms": _sum_darp_timing(steps, "postprocess_ms"),
        "ilp_variables": _sum_darp_timing(steps, "ilp_variables"),
        "ilp_constraints": _sum_darp_timing(steps, "ilp_constraints"),
        "expanded_nodes": _sum_darp_timing(steps, "expanded_nodes"),
    }


def print_summary(spec: DARPProstExperimentSpec, summaries: list[RunSummary], summary_path: Path) -> None:
    """Print compact terminal metric tables."""
    print(f"Wrote summary: {summary_path}")
    print(f"Wrote run CSV: {summary_path.parent / 'runs.csv'}")
    print(f"Wrote summary CSV: {summary_path.parent / 'summary.csv'}")
    rows: list[dict[str, str]] = []
    for summary in summaries:
        rows.extend(_metric_rows(spec, summary))
    darp_headers = _darp_headers(summaries)
    _print_table(rows, headers=["seed", "metric", *darp_headers, "PROST", "note"])


def run_rows_from_summaries(
    spec: DARPProstExperimentSpec,
    summaries: list[RunSummary],
) -> list[dict[str, str]]:
    """Flatten seed summaries into one machine-readable CSV row per solver variant."""
    rows: list[dict[str, str]] = []
    for summary in summaries:
        for variant in summary.darp_variants:
            rows.append(_darp_run_row(spec, summary.seed, variant))
        if summary.prost_metrics or summary.prost_reward is not None or summary.prost_actions:
            rows.append(_prost_run_row(spec, summary))
    return rows


def summary_rows_from_run_rows(run_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Aggregate long-form run rows into mean/std rows for paper tables."""
    grouped: dict[tuple[str, ...], list[dict[str, str]]] = {}
    for row in run_rows:
        key = tuple(row.get(column, "") for column in SUMMARY_GROUP_COLUMNS)
        grouped.setdefault(key, []).append(row)
    summary_rows: list[dict[str, str]] = []
    for key, rows in sorted(grouped.items()):
        summary = dict(zip(SUMMARY_GROUP_COLUMNS, key, strict=True))
        summary["seeds"] = str(len(rows))
        successes = [
            row
            for row in rows
            if row.get("returncode", "") in {"", "0", "0.0"}
        ]
        summary["successes"] = str(len(successes))
        for column in SUMMARY_NUMERIC_COLUMNS:
            values = [_csv_float(row.get(column, "")) for row in rows]
            clean_values = [value for value in values if value is not None]
            summary[f"{column}_mean"] = _csv_number(mean(clean_values)) if clean_values else ""
            summary[f"{column}_std"] = _csv_number(stdev(clean_values)) if len(clean_values) > 1 else ""
        summary_rows.append(summary)
    return summary_rows


def write_csv_rows(path: Path, rows: list[dict[str, str]], columns: tuple[str, ...]) -> None:
    """Write stable-column CSV rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def write_latex_csv_table(path: Path, csv_filename: str) -> None:
    """Write a LaTeX table snippet that reads data from an external CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "% Auto-generated by scripts/darp_prost_compare.py.",
                "% Requires: \\usepackage{booktabs,pgfplotstable}",
                "% Keep data in CSV; override \\darpSummaryCsv before \\input if needed.",
                f"\\providecommand{{\\darpSummaryCsv}}{{{csv_filename}}}",
                "\\pgfplotstableset{",
                "  col sep=comma,",
                "  string type,",
                "  columns/scenario/.style={column name=Scenario},",
                "  columns/system/.style={column name=System},",
                "  columns/variant/.style={column name=Variant},",
                "  columns/total_reward_mean/.style={column name={$\\bar R$}},",
                "  columns/total_reward_std/.style={column name={$\\sigma_R$}},",
                "  columns/turns_mean/.style={column name={Turns}},",
                "  columns/decision_ms_mean/.style={column name={Decision ms}},",
                "  columns/runtime_s_mean/.style={column name={Runtime s}},",
                "}",
                "\\pgfplotstabletypeset[",
                "  columns={scenario,system,variant,total_reward_mean,total_reward_std,turns_mean,decision_ms_mean,runtime_s_mean},",
                "  every head row/.style={before row=\\toprule, after row=\\midrule},",
                "  every last row/.style={after row=\\bottomrule}",
                "]{\\darpSummaryCsv}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def write_latex_preview(path: Path) -> None:
    """Write a complete LaTeX document for previewing the generated table snippet."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "\\documentclass{article}",
                "",
                "\\usepackage[margin=1in]{geometry}",
                "\\usepackage{booktabs}",
                "\\usepackage{pgfplotstable}",
                "\\usepackage[strings]{underscore}",
                "",
                "\\begin{document}",
                "",
                "\\input{summary_table.tex}",
                "",
                "\\end{document}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def clear_open_terminal_outputs(args: argparse.Namespace, run_dir: Path) -> None:
    """Remove stale files that could make --open-terminals parse an old run."""
    paths = [
        run_dir / "prost_server_terminal.log",
        run_dir / "prost_client_terminal.log",
    ]
    for variant in _darp_variants(args):
        key = variant["key"]
        paths.extend(
            [
                run_dir / f"darp_{key}_trace.json",
                run_dir / f"darp_{key}_terminal.log",
            ]
        )
    for path in paths:
        if path.exists():
            path.unlink()
    if not args.skip_prost:
        log_dir = run_dir / "prost_rddlsim_logs"
        if log_dir.exists():
            shutil.rmtree(log_dir)


def _darp_command(
    spec: DARPProstExperimentSpec,
    args: argparse.Namespace,
    seed: int,
    run_dir: Path,
    variant: dict[str, str],
) -> list[str]:
    """Return the DARP command for one seed."""
    command = [
        args.darp_python,
        "-m",
        "darp",
        "--domain",
        str(spec.domain_path),
        "--instance",
        str(spec.instance_path),
    ]
    if spec.duration_path is not None:
        command.extend(["--duration", str(spec.duration_path)])
    command.extend(
        [
            "--planner",
            args.planner,
            "--heuristic-lookahead-depth",
            str(args.heuristic_lookahead_depth),
            "--frontier-width",
            str(args.frontier_width),
            "--hilp-heuristic",
            variant["heuristic"],
            "--seed",
            str(seed),
            "--output",
            str(run_dir / f"darp_{variant['key']}_trace.json"),
        ]
    )
    if args.expansion_rounds is not None:
        command.extend(["--expansion-rounds", str(args.expansion_rounds)])
    return command


def _prost_server_command(
    spec: DARPProstExperimentSpec,
    args: argparse.Namespace,
    seed: int,
    benchmark_dir: Path,
    log_dir: Path,
) -> list[str]:
    """Return the rddlsim server command."""
    prost_root = Path(args.prost_root).resolve()
    return [
        args.prost_python,
        str(prost_root / "testbed" / "run-server.py"),
        "-b",
        str(benchmark_dir),
        "-p",
        str(args.port),
        "-r",
        "1",
        "-s",
        str(seed),
        "--separate-session",
        "-l",
        str(log_dir),
    ]


def _prost_client_command(spec: DARPProstExperimentSpec, args: argparse.Namespace, seed: int) -> list[str]:
    """Return the PROST client command."""
    return [
        "./search-release",
        "-h",
        args.host,
        "-p",
        str(args.port),
        spec.prost_instance_name,
        args.prost_config.format(seed=seed),
    ]


def _prepare_benchmark(spec: DARPProstExperimentSpec, target: Path) -> None:
    """Copy shared RDDL files into a per-run benchmark directory."""
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    shutil.copy2(spec.domain_path, target / spec.domain_path.name)
    shutil.copy2(spec.instance_path, target / spec.instance_path.name)


def _prepare_prost_client_workdir(args: argparse.Namespace, run_dir: Path) -> Path:
    """Copy PROST release binaries into a writable per-run client directory."""
    prost_root = Path(args.prost_root).resolve()
    work_dir = run_dir / "prost_client_work"
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True)
    shutil.copy2(prost_root / "builds" / "release" / "search" / "search", work_dir / "search-release")
    shutil.copy2(
        prost_root / "builds" / "release" / "rddl_parser" / "rddl-parser",
        work_dir / "rddl-parser-release",
    )
    return work_dir


def _wait_for_darp_trace(trace_path: Path, *, timeout: float) -> dict[str, Any]:
    """Wait for the DARP terminal process to finish writing JSON trace."""
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if trace_path.exists():
            try:
                result = parse_darp_trace(trace_path)
                if result.get("reward") is not None:
                    return result
            except (json.JSONDecodeError, OSError) as exc:
                last_error = exc
        time.sleep(0.2)
    if last_error is not None:
        raise TimeoutError(f"Timed out waiting for readable DARP trace {trace_path}: {last_error}") from last_error
    raise TimeoutError(f"Timed out waiting for DARP trace {trace_path}.")


def _wait_for_prost_log(log_path: Path, *, timeout: float) -> dict[str, Any]:
    """Wait for the PROST/rddlsim terminal processes to write session metrics."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        parsed = parse_rddlsim_log(log_path)
        if parsed.get("reward") is not None:
            return parsed
        time.sleep(0.2)
    raise TimeoutError(f"Timed out waiting for PROST/rddlsim log {log_path}.")


def _wait_for_prost_client_output(
    spec: DARPProstExperimentSpec,
    output_path: Path,
    *,
    timeout: float,
) -> dict[str, Any]:
    """Wait for PROST terminal stdout to include final session text, then parse it."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if output_path.exists():
            text = output_path.read_text(encoding="utf-8", errors="replace")
            if "END OF SESSION" in text:
                return parse_prost_client_output(spec, output_path)
        time.sleep(0.2)
    raise TimeoutError(f"Timed out waiting for PROST client output {output_path}.")


def _runtime_from_stamp_files(start_path: Path, finish_path: Path) -> float | None:
    """Return elapsed seconds from terminal-written timestamp files."""
    if not start_path.exists() or not finish_path.exists():
        return None
    try:
        started = float(start_path.read_text(encoding="utf-8").strip())
        finished = float(finish_path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None
    elapsed = finished - started
    return elapsed if elapsed >= 0.0 else None


def _wait_for_server_log(log_path: Path, *, timeout: float) -> None:
    """Wait until rddlsim reports readiness without consuming its client socket."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if log_path.exists() and "RDDL Server Initialized" in log_path.read_text(
            encoding="utf-8",
            errors="replace",
        ):
            return
        time.sleep(0.2)
    raise TimeoutError(f"Timed out waiting for rddlsim readiness in {log_path}.")


def _write_shell_script(path: Path, lines: list[str]) -> None:
    """Write an executable shell script used by --open-terminals."""
    path.write_text("#!/usr/bin/env bash\n" + "\n".join(lines) + "\n", encoding="utf-8")
    path.chmod(0o755)


def _open_terminal(terminal: str, title: str, script: Path) -> None:
    """Open one GNOME Terminal window for a generated shell script."""
    subprocess.Popen(
        [
            terminal,
            "--title",
            title,
            "--",
            "bash",
            "-lc",
            str(script),
        ],
        cwd=REPO_ROOT,
    )


def _shell_command(parts: list[str]) -> str:
    """Quote a command list for generated shell scripts."""
    return " ".join(shlex.quote(part) for part in parts)


def _parse_seeds(value: str) -> list[int]:
    """Parse a comma-separated seed list."""
    return [int(piece.strip()) for piece in value.split(",") if piece.strip()]


def _metric_rows(spec: DARPProstExperimentSpec, summary: RunSummary) -> list[dict[str, str]]:
    """Return transposed DARP-vs-PROST rows for one seed."""
    darp_maps = {
        variant.label: _darp_metric_map(variant.result)
        for variant in summary.darp_variants
    }
    prost = {
        "total_reward": _format_optional_float(summary.prost_reward),
        "turns": str(summary.prost_turns[-1]) if summary.prost_turns else "",
        "runtime_s": _format_optional_float(summary.prost_runtime_seconds),
        "rddl_load_ms": _format_optional_float(_prost_metric(summary, "parser_complete_ms")),
        "grounding_ms": _format_optional_float(_prost_metric(summary, "instantiating_ms")),
        "and_or_interface_ms": "",
        "initial_belief_ms": "",
        "planner_elapsed_ms": "",
        "decision_ms": _format_optional_float(summary.prost_search_ms),
        "frontier_expand_ms": "",
        "heuristic_eval_ms": "",
        "ilp_encode_ms": "",
        "tree_ilp_build_ms": "",
        "gurobi_call_ms": "",
        "postprocess_ms": "",
        "actions": ",".join(summary.prost_actions),
        "states": "->".join(summary.prost_state_trace),
        "expanded_nodes": _format_optional_float(_prost_metric(summary, "created_search_nodes")),
        "performed_trials": _format_optional_float(_prost_metric(summary, "performed_trials")),
        "ilp_vars": "",
        "ilp_constraints": "",
        "gurobi_ms": "",
        "prost_parsing_ms": _format_optional_float(_prost_metric(summary, "parsing_ms")),
        "prost_simplifying_ms": _format_optional_float(_prost_metric(summary, "simplifying_ms")),
        "prost_analyzing_ms": _format_optional_float(_prost_metric(summary, "analyzing_ms")),
        "risk_budget": "",
        "constraint_violation": "",
    }
    notes = {
        "planner_elapsed_ms": "DARP sums per-step choose_action wall-clock elapsed_ms; excludes env.step",
        "decision_ms": "DARP uses timing.decision_ms; PROST sums reported Search time lines",
        "rddl_load_ms": "DARP pyRDDLGym load; PROST parser complete running time",
        "grounding_ms": "DARP pyRDDLGym RDDLGrounder; PROST instantiating phase as closest counterpart",
        "and_or_interface_ms": "DARP builds AND-OR search interface; PROST counterpart not parsed",
        "initial_belief_ms": "DARP exact initial/root belief construction",
        "frontier_expand_ms": "DARP time spent calling Algorithm 2 Expand for HILP frontier leaves",
        "heuristic_eval_ms": "DARP time spent evaluating HILP frontier heuristic",
        "ilp_encode_ms": "DARP time spent encoding the current p-ILP schema",
        "tree_ilp_build_ms": "DARP total partial-tree build time: expand + heuristic + ILP encoding overhead",
        "gurobi_call_ms": "DARP wall-clock time inside the Gurobi adapter call",
        "postprocess_ms": "DARP time spent extracting the selected root action",
        "states": spec.prost_state_note,
        "expanded_nodes": "DARP sums expanded action histories; PROST sums Created search nodes",
        "performed_trials": "PROST sums Performed trials; DARP does not use THTS trials",
        "ilp_vars": "DARP sums per-step ILP variable counts; PROST not parsed yet",
        "ilp_constraints": "DARP sums per-step ILP constraint counts; PROST not parsed yet",
        "gurobi_ms": "DARP sums per-step Gurobi solver-call timings; PROST does not use Gurobi",
        "prost_parsing_ms": "PROST parser Parsing phase; DARP equivalent included in rddl_load_ms",
        "prost_simplifying_ms": "PROST simplifier phase; DARP counterpart not separately exposed",
        "prost_analyzing_ms": "PROST task analysis phase; DARP counterpart not separately exposed",
        "risk_budget": "not used by this experiment unless the scenario adds it",
        "constraint_violation": "not collected yet",
    }
    metrics = (
        "total_reward",
        "turns",
        "runtime_s",
        "rddl_load_ms",
        "grounding_ms",
        "and_or_interface_ms",
        "initial_belief_ms",
        "planner_elapsed_ms",
        "decision_ms",
        "frontier_expand_ms",
        "heuristic_eval_ms",
        "ilp_encode_ms",
        "tree_ilp_build_ms",
        "gurobi_call_ms",
        "postprocess_ms",
        "actions",
        "states",
        "expanded_nodes",
        "performed_trials",
        "ilp_vars",
        "ilp_constraints",
        "gurobi_ms",
        "prost_parsing_ms",
        "prost_simplifying_ms",
        "prost_analyzing_ms",
        "risk_budget",
        "constraint_violation",
    )
    rows: list[dict[str, str]] = []
    for metric in metrics:
        row = {
            "seed": str(summary.seed),
            "metric": metric,
            "PROST": prost[metric],
            "note": notes.get(metric, ""),
        }
        for label, metrics_map in darp_maps.items():
            row[label] = metrics_map.get(metric, "")
        rows.append(row)
    return rows


def _darp_headers(summaries: list[RunSummary]) -> list[str]:
    """Return DARP variant columns in first-seen order."""
    headers: list[str] = []
    for summary in summaries:
        for variant in summary.darp_variants:
            if variant.label not in headers:
                headers.append(variant.label)
    return headers


def _darp_metric_map(result: dict[str, Any]) -> dict[str, str]:
    """Format one DARP variant result into table metrics."""
    return {
        "total_reward": _format_optional_float(_optional_float(result.get("reward"))),
        "turns": str(len(result.get("actions", ()))) if result.get("actions") else "",
        "runtime_s": _format_optional_float(_optional_float(result.get("runtime_seconds"))),
        "rddl_load_ms": _format_optional_float(_optional_float(result.get("rddl_load_ms"))),
        "grounding_ms": _format_optional_float(_optional_float(result.get("grounding_ms"))),
        "and_or_interface_ms": _format_optional_float(_optional_float(result.get("and_or_interface_ms"))),
        "initial_belief_ms": _format_optional_float(_optional_float(result.get("initial_belief_ms"))),
        "planner_elapsed_ms": _format_optional_float(_optional_float(result.get("planner_elapsed_ms"))),
        "decision_ms": _format_optional_float(_optional_float(result.get("decision_ms"))),
        "frontier_expand_ms": _format_optional_float(_optional_float(result.get("frontier_expand_ms"))),
        "heuristic_eval_ms": _format_optional_float(_optional_float(result.get("heuristic_eval_ms"))),
        "ilp_encode_ms": _format_optional_float(_optional_float(result.get("ilp_encode_ms"))),
        "tree_ilp_build_ms": _format_optional_float(_optional_float(result.get("tree_ilp_build_ms"))),
        "gurobi_call_ms": _format_optional_float(_optional_float(result.get("gurobi_call_ms"))),
        "postprocess_ms": _format_optional_float(_optional_float(result.get("postprocess_ms"))),
        "actions": ",".join(str(action) for action in result.get("actions", ())),
        "states": "->".join(str(state) for state in result.get("state_trace", ())),
        "expanded_nodes": _format_optional_float(_optional_float(result.get("expanded_nodes"))),
        "performed_trials": "",
        "ilp_vars": _format_optional_float(_optional_float(result.get("ilp_variables"))),
        "ilp_constraints": _format_optional_float(_optional_float(result.get("ilp_constraints"))),
        "gurobi_ms": _format_optional_float(_optional_float(result.get("gurobi_ms"))),
        "prost_parsing_ms": "",
        "prost_simplifying_ms": "",
        "prost_analyzing_ms": "",
        "risk_budget": "",
        "constraint_violation": "",
    }


def _darp_run_row(
    spec: DARPProstExperimentSpec,
    seed: int,
    variant: DARPVariantSummary,
) -> dict[str, str]:
    """Return one long-form CSV row for a DARP variant."""
    result = variant.result
    return {
        "scenario": spec.name,
        "seed": str(seed),
        "system": "DARP",
        "variant": variant.label,
        "planner": str(result.get("planner", spec.default_planner)),
        "heuristic": variant.heuristic,
        "domain": str(spec.domain_path),
        "instance": str(spec.instance_path),
        "duration": str(spec.duration_path) if spec.duration_path is not None else "fixed-unit-default",
        "total_reward": _csv_number(_optional_float(result.get("reward"))),
        "turns": _csv_number(float(len(result.get("actions", ()))) if result.get("actions") else None),
        "runtime_s": _csv_number(_optional_float(result.get("runtime_seconds"))),
        "decision_ms": _csv_number(_optional_float(result.get("decision_ms"))),
        "planner_elapsed_ms": _csv_number(_optional_float(result.get("planner_elapsed_ms"))),
        "rddl_load_ms": _csv_number(_optional_float(result.get("rddl_load_ms"))),
        "grounding_ms": _csv_number(_optional_float(result.get("grounding_ms"))),
        "and_or_interface_ms": _csv_number(_optional_float(result.get("and_or_interface_ms"))),
        "initial_belief_ms": _csv_number(_optional_float(result.get("initial_belief_ms"))),
        "frontier_expand_ms": _csv_number(_optional_float(result.get("frontier_expand_ms"))),
        "heuristic_eval_ms": _csv_number(_optional_float(result.get("heuristic_eval_ms"))),
        "ilp_encode_ms": _csv_number(_optional_float(result.get("ilp_encode_ms"))),
        "tree_ilp_build_ms": _csv_number(_optional_float(result.get("tree_ilp_build_ms"))),
        "gurobi_call_ms": _csv_number(_optional_float(result.get("gurobi_call_ms"))),
        "postprocess_ms": _csv_number(_optional_float(result.get("postprocess_ms"))),
        "expanded_nodes": _csv_number(_optional_float(result.get("expanded_nodes"))),
        "performed_trials": "",
        "ilp_vars": _csv_number(_optional_float(result.get("ilp_variables"))),
        "ilp_constraints": _csv_number(_optional_float(result.get("ilp_constraints"))),
        "gurobi_ms": _csv_number(_optional_float(result.get("gurobi_ms"))),
        "prost_parsing_ms": "",
        "prost_instantiating_ms": "",
        "prost_simplifying_ms": "",
        "prost_analyzing_ms": "",
        "risk_budget": "",
        "constraint_violation": "",
        "returncode": _csv_number(_optional_float(result.get("returncode"))),
        "actions": ",".join(str(action) for action in result.get("actions", ())),
        "states": "->".join(str(state) for state in result.get("state_trace", ())),
    }


def _prost_run_row(spec: DARPProstExperimentSpec, summary: RunSummary) -> dict[str, str]:
    """Return one long-form CSV row for PROST."""
    return {
        "scenario": spec.name,
        "seed": str(summary.seed),
        "system": "PROST",
        "variant": "PROST",
        "planner": "prost",
        "heuristic": "",
        "domain": str(spec.domain_path),
        "instance": str(spec.instance_path),
        "duration": "not-used-by-prost",
        "total_reward": _csv_number(_optional_float(summary.prost_reward)),
        "turns": _csv_number(float(summary.prost_turns[-1]) if summary.prost_turns else None),
        "runtime_s": _csv_number(_optional_float(summary.prost_runtime_seconds)),
        "decision_ms": _csv_number(_optional_float(summary.prost_search_ms)),
        "planner_elapsed_ms": "",
        "rddl_load_ms": _csv_number(_prost_metric(summary, "parser_complete_ms")),
        "grounding_ms": _csv_number(_prost_metric(summary, "instantiating_ms")),
        "and_or_interface_ms": "",
        "initial_belief_ms": "",
        "frontier_expand_ms": "",
        "heuristic_eval_ms": "",
        "ilp_encode_ms": "",
        "tree_ilp_build_ms": "",
        "gurobi_call_ms": "",
        "postprocess_ms": "",
        "expanded_nodes": _csv_number(_prost_metric(summary, "created_search_nodes")),
        "performed_trials": _csv_number(_prost_metric(summary, "performed_trials")),
        "ilp_vars": "",
        "ilp_constraints": "",
        "gurobi_ms": "",
        "prost_parsing_ms": _csv_number(_prost_metric(summary, "parsing_ms")),
        "prost_instantiating_ms": _csv_number(_prost_metric(summary, "instantiating_ms")),
        "prost_simplifying_ms": _csv_number(_prost_metric(summary, "simplifying_ms")),
        "prost_analyzing_ms": _csv_number(_prost_metric(summary, "analyzing_ms")),
        "risk_budget": "",
        "constraint_violation": "",
        "returncode": _csv_number(float(summary.prost_returncode) if summary.prost_returncode is not None else None),
        "actions": ",".join(summary.prost_actions),
        "states": "->".join(summary.prost_state_trace),
    }


def summary_csv_columns() -> tuple[str, ...]:
    """Return stable columns for aggregate CSV files."""
    numeric_columns: list[str] = []
    for column in SUMMARY_NUMERIC_COLUMNS:
        numeric_columns.extend([f"{column}_mean", f"{column}_std"])
    return (*SUMMARY_GROUP_COLUMNS, "seeds", "successes", *numeric_columns)


def _prost_metric(summary: RunSummary, key: str) -> float | None:
    """Return one optional numeric PROST metric."""
    return _optional_float(summary.prost_metrics.get(key))


def _sum_darp_planner_elapsed_ms(steps: tuple[dict[str, Any], ...]) -> float | None:
    """Sum DARP per-step choose_action wall-clock elapsed times from JSON trace."""
    values = []
    for step in steps:
        decision = step.get("decision")
        if not isinstance(decision, dict):
            continue
        timing = decision.get("timing")
        if isinstance(timing, dict) and timing.get("planner_elapsed_ms") is not None:
            values.append(timing["planner_elapsed_ms"])
        elif decision.get("elapsed_ms") is not None:
            values.append(decision["elapsed_ms"])
    return sum(float(value) for value in values) if values else None


def _sum_darp_timing(steps: tuple[dict[str, Any], ...], key: str) -> float | None:
    """Sum one numeric DARP decision timing/diagnostic field."""
    values = []
    for step in steps:
        decision = step.get("decision")
        if not isinstance(decision, dict):
            continue
        timing = decision.get("timing")
        if isinstance(timing, dict) and timing.get(key) is not None:
            values.append(timing[key])
    return sum(float(value) for value in values) if values else None


def _state_trace_from_darp_steps(steps: tuple[dict[str, Any], ...]) -> tuple[str, ...]:
    """Return compact active-state labels from DARP trace steps."""
    if not steps:
        return ()
    trace = [_state_label(steps[0].get("state"))]
    trace.extend(_state_label(step.get("next_state")) for step in steps)
    return tuple(label for label in trace if label)


def _state_label(state: Any) -> str:
    """Return a readable label for a Boolean RDDL state mapping."""
    if not isinstance(state, dict):
        return ""
    active = [name for name, value in sorted(state.items()) if value is True]
    if not active:
        if any(str(name).startswith("robot-at___") for name in state):
            return "lost"
        return ""
    return "+".join(_short_state_name(name) for name in active)


def _short_state_name(name: str) -> str:
    """Shorten grounded labels while keeping generic names readable."""
    if name.startswith("at___"):
        return name.removeprefix("at___")
    return name


def _normalize_prost_action(action: str) -> str:
    """Normalize PROST action text to match DARP trace labels where possible."""
    cleaned = action.strip()
    if cleaned == "noop()":
        return "noop"
    return cleaned


def _prost_top_level_phase_seconds(text: str, label: str) -> float | None:
    """Parse one top-level PROST parser phase duration in seconds."""
    pattern = rf"^{re.escape(label)}\.\.\.$(?P<body>.*?)^\.\.\.finished \(([0-9.eE+-]+)s?\)\."
    matches = tuple(re.finditer(pattern, text, flags=re.M | re.S))
    if not matches:
        return None
    return float(matches[-1].group(2))


def _last_float(pattern: str, text: str) -> float | None:
    """Return the last float captured by a regex."""
    values = re.findall(pattern, text, flags=re.M)
    return float(values[-1]) if values else None


def _sum_float_matches(pattern: str, text: str) -> float | None:
    """Return the sum of all float captures for a regex."""
    values = [float(value) for value in re.findall(pattern, text, flags=re.M)]
    return sum(values) if values else None


def _seconds_to_ms(value: float | None) -> float | None:
    """Convert optional seconds to milliseconds."""
    return None if value is None else value * 1000.0


def _optional_float(value: Any) -> float | None:
    """Convert optional numeric values to float."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _print_table(rows: list[dict[str, str]], *, headers: list[str]) -> None:
    """Print a fixed-header metric table with blank cells for unavailable metrics."""
    clipped_rows = [{key: _clip(value) for key, value in row.items()} for row in rows]
    widths = {
        key: max(len(key), *(len(row.get(key, "")) for row in clipped_rows))
        for key in headers
    }
    print(" | ".join(key.ljust(widths[key]) for key in headers))
    print("-+-".join("-" * widths[key] for key in headers))
    for row in clipped_rows:
        print(" | ".join(row.get(key, "").ljust(widths[key]) for key in headers))


def _format_optional_float(value: float | None) -> str:
    """Format an optional float for a compact table."""
    return "" if value is None else f"{value:.3f}"


def _csv_number(value: float | None) -> str:
    """Format an optional number for machine-readable CSV output."""
    if value is None:
        return ""
    return f"{value:.12g}"


def _csv_float(value: str) -> float | None:
    """Parse an optional CSV number."""
    if value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _clip(value: str, *, limit: int = 88) -> str:
    """Clip long table cells while keeping the table readable."""
    return value if len(value) <= limit else value[: limit - 3] + "..."
