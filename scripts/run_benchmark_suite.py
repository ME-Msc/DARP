"""Run DARP-vs-PROST benchmark comparisons.

The runner is intentionally command-line first: one command can name a
benchmark, point to RDDL files, choose HILP settings, and write all artifacts
under `experiments/`.
"""

# TODO(benchmarks): Add domain-specific PROST state parsers as benchmark scenarios mature.

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Mapping

from darp_prost_compare import (
    DARPProstExperimentSpec,
    ProstStateTraceParser,
    REPO_ROOT,
    RUN_CSV_COLUMNS,
    run_experiment,
    summary_csv_columns,
    summary_rows_from_run_rows,
    write_csv_rows,
    write_latex_csv_table,
    write_latex_preview,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the benchmark-suite command-line parser."""
    parser = argparse.ArgumentParser(description="Run one or more DARP/PROST benchmark comparisons.")
    parser.add_argument(
        "experiments",
        nargs="*",
        help="experiment folder(s) containing config.json, or explicit config JSON file(s)",
    )
    parser.add_argument("--name", help="direct experiment name, e.g. navigation_2011_inst2_fixed_1")
    parser.add_argument("--domain", help="direct RDDL domain path")
    parser.add_argument("--instance", help="direct RDDL instance path")
    parser.add_argument("--duration", help="direct duration sidecar path, e.g. examples/durations/fixed_1.yaml")
    parser.add_argument("--prost-instance-name", help="direct PROST instance name; defaults to the instance filename stem")
    parser.add_argument("--prost-state-parser", help="named PROST state parser, e.g. tiny_grid")
    parser.add_argument("--seed", type=int, help="single random seed; shorthand for --seeds")
    parser.add_argument("--seeds", help="comma-separated seeds, e.g. 0,1,2")
    parser.add_argument("--output-dir", help="override output root directory")
    parser.add_argument("--timeout", type=float, help="override per-command timeout in seconds")
    parser.add_argument("--port", type=int, help="override rddlsim server port")
    parser.add_argument("--planner", help="override DARP planner name")
    parser.add_argument("--hilp-heuristics", help="override comma-separated DARP HILP heuristic modes")
    parser.add_argument("--heuristic-lookahead-depth", type=int, help="override DARP reachable-bellman heuristic depth")
    parser.add_argument("--expansion-rounds", type=int, help="override optional DARP HILP expansion budget")
    parser.add_argument("--frontier-width", type=int, help="override DARP HILP frontier width")
    parser.add_argument(
        "--no-visualize",
        action="store_true",
        help="skip replay.html generation; visualization is enabled by default",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the configured benchmark suite."""
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_seed_args(args, parser)
    _validate_direct_args(args, parser)
    config_paths = tuple(_config_path(Path(experiment).resolve()) for experiment in args.experiments)
    output_root = Path(args.output_dir or REPO_ROOT / "experiments").resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    configs = list(_read_config(path) for path in config_paths)
    direct_config = _direct_config_from_args(args)
    if direct_config is not None:
        configs.append(direct_config)
        _write_direct_config(direct_config, output_root)
    if not configs:
        parser.error("provide at least one experiment folder/config, or use --name --domain --instance")

    all_run_rows: list[dict[str, str]] = []
    for config in configs:
        for scenario_config in _scenarios(config):
            spec = _spec_from_config(scenario_config, output_root)
            scenario_args = _scenario_args(config, scenario_config, args, output_root / spec.name)
            print(f"\n== Running {spec.name} ==")
            run_experiment(spec, scenario_args)
            scenario_runs_csv = output_root / spec.name / "runs.csv"
            scenario_rows = _read_run_rows(scenario_runs_csv)
            all_run_rows.extend(scenario_rows)
            if not args.no_visualize and scenario_rows:
                from darp.visualization import build_replay_html

                replay_path = build_replay_html(
                    scenario_runs_csv,
                    output_root / spec.name / "replay.html",
                    title=spec.name,
                )
                print(f"Wrote replay visualizer: {replay_path}")

    runs_csv = output_root / "runs.csv"
    write_csv_rows(runs_csv, all_run_rows, RUN_CSV_COLUMNS)
    summary_csv = output_root / "summary.csv"
    write_csv_rows(summary_csv, summary_rows_from_run_rows(all_run_rows), _suite_summary_columns())
    latex_dir = output_root / "latex"
    latex_table = latex_dir / "summary_table.tex"
    write_latex_csv_table(latex_table, "../summary.csv")
    latex_preview = latex_dir / "preview_summary_table.tex"
    write_latex_preview(latex_preview)
    print("\n== Aggregate artifacts ==")
    print(f"Wrote aggregate run CSV: {runs_csv}")
    print(f"Wrote aggregate summary CSV: {summary_csv}")
    print(f"Wrote aggregate LaTeX table reader: {latex_table}")
    print(f"Wrote aggregate LaTeX preview: {latex_preview}")
    return 0


def _read_config(path: Path) -> Mapping[str, Any]:
    """Read one benchmark-suite JSON config."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("Benchmark suite config must be a JSON object.")
    return payload


def _config_path(path: Path) -> Path:
    """Return `config.json` when an experiment directory is provided."""
    if path.is_dir():
        return path / "config.json"
    return path


def _scenarios(config: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    """Return scenario configs after basic validation."""
    if "scenarios" not in config:
        return (config,)
    scenarios = config.get("scenarios")
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError("Benchmark suite config must contain a non-empty `scenarios` list.")
    if not all(isinstance(scenario, Mapping) for scenario in scenarios):
        raise ValueError("Every benchmark scenario must be a JSON object.")
    return tuple(scenarios)


def _direct_config_from_args(args: argparse.Namespace) -> Mapping[str, Any] | None:
    """Build a one-scenario config from direct CLI paths."""
    direct_values = (
        args.name,
        args.domain,
        args.instance,
        args.duration,
        args.prost_instance_name,
        args.prost_state_parser,
    )
    if not any(value is not None for value in direct_values):
        return None
    config: dict[str, Any] = {
        "name": args.name,
        "description": f"DARP/PROST comparison for {args.name}.",
        "domain": args.domain,
        "instance": args.instance,
        "duration": args.duration,
        "prost_instance_name": args.prost_instance_name or _instance_name_from_rddl_or_path(args.instance),
        "seeds": _seed_arg(args) or "0",
        "planner": args.planner or "hilp",
        "heuristic_lookahead_depth": args.heuristic_lookahead_depth
        if args.heuristic_lookahead_depth is not None
        else 4,
        "expansion_rounds": args.expansion_rounds,
        "frontier_width": args.frontier_width if args.frontier_width is not None else 1,
        "hilp_heuristics": args.hilp_heuristics or "reachable-bellman",
        "timeout": args.timeout if args.timeout is not None else 120.0,
    }
    if args.prost_state_parser:
        config["prost_state_parser"] = args.prost_state_parser
    return config


def _validate_direct_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Validate direct CLI benchmark mode before running experiments."""
    direct_values = (
        args.name,
        args.domain,
        args.instance,
        args.duration,
        args.prost_instance_name,
        args.prost_state_parser,
    )
    if not any(value is not None for value in direct_values):
        return
    missing = [option for option, value in (("--name", args.name), ("--domain", args.domain), ("--instance", args.instance)) if not value]
    if missing:
        parser.error(f"direct benchmark mode requires: {', '.join(missing)}")


def _validate_seed_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Validate seed arguments."""
    if args.seed is not None and args.seeds:
        parser.error("use either --seed or --seeds, not both")


def _write_direct_config(config: Mapping[str, Any], output_root: Path) -> Path:
    """Persist direct CLI experiment settings for reproducibility."""
    experiment_dir = output_root / str(config["name"])
    experiment_dir.mkdir(parents=True, exist_ok=True)
    path = experiment_dir / "config.json"
    path.write_text(json.dumps(config, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    print(f"Wrote experiment config: {path}")
    return path


def _spec_from_config(config: Mapping[str, Any], suite_output_dir: Path) -> DARPProstExperimentSpec:
    """Build a DARPProstExperimentSpec from one JSON scenario object."""
    name = str(_required(config, "name"))
    return DARPProstExperimentSpec(
        name=name,
        description=str(config.get("description", f"DARP/PROST comparison for {name}.")),
        domain_path=_repo_path(_required(config, "domain")),
        instance_path=_repo_path(_required(config, "instance")),
        duration_path=_optional_repo_path(config.get("duration")),
        prost_instance_name=str(config.get("prost_instance_name", _instance_name_from_rddl_or_path(config["instance"]))),
        default_output_dir=suite_output_dir / name,
        prost_state_trace_parser=_state_parser_from_config(config.get("prost_state_parser")),
        prost_state_note=str(config.get("prost_state_note", "PROST state trace parser is not configured for this scenario")),
        default_planner=str(config.get("planner", "hilp")),
        default_heuristic_lookahead_depth=int(config.get("heuristic_lookahead_depth", config.get("lookahead_depth", 4))),
        default_expansion_rounds=_optional_int(config.get("expansion_rounds", config.get("hilp_iterations"))),
        default_frontier_width=int(config.get("frontier_width", 1)),
        default_hilp_heuristics=_heuristic_tuple(config.get("hilp_heuristics", ("reachable-bellman",))),
        default_prost_config=str(config.get("prost_config", "[Prost -s {seed} -se [IPC2014]]")),
    )


def _scenario_args(
    suite_config: Mapping[str, Any],
    scenario_config: Mapping[str, Any],
    cli_args: argparse.Namespace,
    output_dir: Path,
) -> list[str]:
    """Build per-scenario argv for the reusable comparison runner."""
    defaults = suite_config.get("defaults", {})
    if defaults is None:
        defaults = {}
    if not isinstance(defaults, Mapping):
        raise ValueError("Benchmark suite `defaults` must be a JSON object when present.")
    args = [
        "--seeds",
        str(_seed_arg(cli_args) or scenario_config.get("seeds", defaults.get("seeds", "0"))),
        "--output-dir",
        str(output_dir),
        "--timeout",
        str(cli_args.timeout or scenario_config.get("timeout", defaults.get("timeout", 120.0))),
        "--planner",
        str(cli_args.planner or scenario_config.get("planner", defaults.get("planner", "hilp"))),
        "--heuristic-lookahead-depth",
        str(
            cli_args.heuristic_lookahead_depth
            if cli_args.heuristic_lookahead_depth is not None
            else scenario_config.get(
                "heuristic_lookahead_depth",
                defaults.get("heuristic_lookahead_depth", scenario_config.get("lookahead_depth", defaults.get("lookahead_depth", 4))),
            )
        ),
        "--frontier-width",
        str(
            cli_args.frontier_width
            if cli_args.frontier_width is not None
            else scenario_config.get("frontier_width", defaults.get("frontier_width", 1))
        ),
        "--hilp-heuristics",
        _heuristic_arg(
            cli_args.hilp_heuristics
            or scenario_config.get("hilp_heuristics", defaults.get("hilp_heuristics", "reachable-bellman"))
        ),
    ]
    _append_optional_arg(args, "--port", cli_args.port or scenario_config.get("port", defaults.get("port")))
    expansion_rounds = (
        cli_args.expansion_rounds
        if cli_args.expansion_rounds is not None
        else scenario_config.get(
            "expansion_rounds",
            defaults.get("expansion_rounds", scenario_config.get("hilp_iterations", defaults.get("hilp_iterations"))),
        )
    )
    _append_optional_arg(args, "--expansion-rounds", expansion_rounds)
    _append_optional_arg(args, "--prost-config", scenario_config.get("prost_config", defaults.get("prost_config")))
    return args


def _read_run_rows(path: Path) -> list[dict[str, str]]:
    """Read long-form run rows from one scenario artifact."""
    with path.open("r", encoding="utf-8", newline="") as file:
        return [dict(row) for row in csv.DictReader(file)]


def _suite_summary_columns() -> tuple[str, ...]:
    """Return columns from a generated summary row set."""
    return summary_csv_columns()


def _append_optional_arg(args: list[str], option: str, value: object | None) -> None:
    """Append one optional command-line argument."""
    if value is not None:
        args.extend([option, str(value)])


def _optional_int(value: object | None) -> int | None:
    """Return an optional integer from JSON config values."""
    if value is None:
        return None
    return int(value)


def _seed_arg(args: argparse.Namespace) -> str | None:
    """Return the effective seed string from --seed or --seeds."""
    if args.seed is not None:
        return str(args.seed)
    return args.seeds


def _state_parser_from_config(value: object | None) -> ProstStateTraceParser | None:
    """Return a named PROST state parser for scenarios that need one."""
    if value in {None, ""}:
        return None
    if value == "tiny_grid":
        from tiny_grid_fixed_1 import tiny_grid_prost_state_trace

        return tiny_grid_prost_state_trace
    raise ValueError(f"Unknown prost_state_parser: {value}")


def _heuristic_tuple(value: object) -> tuple[str, ...]:
    """Normalize heuristic config values to a tuple of names."""
    if isinstance(value, str):
        return tuple(piece.strip() for piece in value.split(",") if piece.strip())
    if isinstance(value, list | tuple):
        return tuple(str(piece).strip() for piece in value if str(piece).strip())
    raise ValueError("hilp_heuristics must be a comma-separated string or list.")


def _heuristic_arg(value: object) -> str:
    """Normalize heuristic config values for the scenario runner CLI."""
    return ",".join(_heuristic_tuple(value))


def _repo_path(value: object) -> Path:
    """Resolve a repo-relative or absolute path."""
    path = Path(str(value))
    return path if path.is_absolute() else REPO_ROOT / path


def _optional_repo_path(value: object | None) -> Path | None:
    """Resolve an optional repo-relative path."""
    return None if value in {None, ""} else _repo_path(value)


def _instance_name_from_path(value: object) -> str:
    """Infer an RDDL instance name from a path stem."""
    stem = Path(str(value)).stem
    return stem


def _instance_name_from_rddl_or_path(value: object) -> str:
    """Infer the RDDL instance name from file text, falling back to the path stem."""
    path = _repo_path(value)
    if path.exists():
        match = re.search(r"\binstance\s+([A-Za-z_][A-Za-z0-9_]*)\s*\{", path.read_text(encoding="utf-8"))
        if match:
            return match.group(1)
    return _instance_name_from_path(value)


def _required(config: Mapping[str, Any], key: str) -> Any:
    """Return a required config value."""
    if key not in config:
        raise ValueError(f"Scenario config is missing required key `{key}`.")
    return config[key]


if __name__ == "__main__":
    raise SystemExit(main())
