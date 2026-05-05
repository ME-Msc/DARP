import json

import pytest

from darp.__main__ import build_parser, main


def test_darp_help_exits_successfully(capsys):
    """Check `darp -h` help is available. / 检查 `darp -h` 帮助可用。"""
    parser = build_parser()

    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["-h"])

    assert exc_info.value.code == 0
    assert "--visualizer" in capsys.readouterr().out


def test_darp_visualizer_arguments_parse():
    """Check top-level visualizer arguments parse cleanly. / 检查顶层 visualizer 参数能正确解析。"""
    args = build_parser().parse_args(
        [
            "--visualizer",
            "--domain",
            "domain.rddl",
            "--instance",
            "instance.rddl",
            "--with-simulator",
            "darp",
            "--frontend",
            "darp",
            "--host",
            "127.0.0.1",
            "--port",
            "8080",
            "--no-open",
        ]
    )

    assert args.visualizer is True
    assert args.domain == "domain.rddl"
    assert args.instance == "instance.rddl"
    assert args.with_simulator == "darp"
    assert args.frontend == "darp"
    assert args.port == 8080
    assert args.no_open is True


def test_darp_solve_online_arguments_parse():
    """Check online solve arguments parse cleanly. / 检查在线 solve 参数能正确解析。"""
    args = build_parser().parse_args(
        [
            "solve",
            "--mode",
            "online",
            "--steps",
            "2",
            "--seed",
            "7",
            "--time-budget-ms",
            "5",
        ]
    )

    assert args.command == "solve"
    assert args.mode == "online"
    assert args.steps == 2
    assert args.seed == 7
    assert args.time_budget_ms == 5.0


def test_darp_solve_online_outputs_trace(capsys):
    """Check `darp solve` prints an online trace. / 检查 `darp solve` 会输出在线轨迹。"""
    exit_code = main(["solve", "--mode", "online", "--steps", "2", "--seed", "7"])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload["mode"] == "online"
    assert payload["planner"] == "finite-horizon-dp"
    assert len(payload["steps"]) == 2
    assert payload["steps"][0]["action"] == "safe_path"
