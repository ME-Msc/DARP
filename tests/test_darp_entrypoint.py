"""Tests for the top-level DARP entrypoint."""

import json

import pytest

from darp.__main__ import build_parser, main


def test_darp_help_exits_successfully(capsys):
    """Check `darp -h` help is available. / 检查 `darp -h` 帮助可用。"""
    parser = build_parser()

    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["-h"])

    assert exc_info.value.code == 0
    help_text = capsys.readouterr().out
    assert "--domain" in help_text
    assert "--instance" in help_text
    assert "--lookahead-depth" in help_text
    assert "--particles" in help_text


def test_darp_rddl_arguments_parse():
    """Check top-level RDDL trace arguments parse cleanly. / 检查顶层 RDDL trace 参数能正确解析。"""
    args = build_parser().parse_args(
        [
            "--domain",
            "domain.rddl",
            "--instance",
            "instance.rddl",
            "--seed",
            "3",
        ]
    )

    assert args.domain == "domain.rddl"
    assert args.instance == "instance.rddl"
    assert args.seed == 3
    assert args.lookahead_depth == 4
    assert args.particles == 32


def test_darp_requires_rddl_files():
    """Check DARP requires explicit RDDL domain and instance files. / 检查 DARP 需要显式 RDDL domain 和 instance。"""
    with pytest.raises(SystemExit) as exc_info:
        main([])

    assert exc_info.value.code == 2


def test_darp_unknown_argument_errors():
    """Check unknown CLI arguments fail clearly. / 检查未知 CLI 参数会明确报错。"""
    with pytest.raises(SystemExit) as exc_info:
        build_parser().parse_args(["--unknown"])

    assert exc_info.value.code == 2


def test_darp_runs_rddl_online_through_pyrddlgym(capsys):
    """Check RDDL inputs run through pyRDDLGym online execution. / 检查 RDDL 输入会通过 pyRDDLGym 在线执行。"""
    pytest.importorskip("pyRDDLGym")

    exit_code = main(
        [
            "--domain",
            "examples/rddl/tiny_grid_domain.rddl",
            "--instance",
            "examples/rddl/tiny_grid_instance.rddl",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert captured.out.startswith("DARP pyRDDLGym online trace")
    assert "Planner: pyrddlgym-rollout" in captured.out
    assert "action=move-east" in captured.out
    assert "next=at___c33" in captured.out


def test_darp_rddl_output_writes_json_trace(tmp_path, capsys):
    """Check RDDL online execution writes JSON when requested. / 检查 RDDL 在线执行按需写入 JSON。"""
    pytest.importorskip("pyRDDLGym")
    output = tmp_path / "rddl-trace.json"
    exit_code = main(
        [
            "--domain",
            "examples/rddl/tiny_grid_domain.rddl",
            "--instance",
            "examples/rddl/tiny_grid_instance.rddl",
            "--output",
            str(output),
        ]
    )
    captured = capsys.readouterr()
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert captured.out.startswith("DARP pyRDDLGym online trace")
    assert payload["planner"] == "pyrddlgym-rollout"
    assert payload["rddl"]["artifacts"]["env"] == "RDDLEnv"
    assert payload["steps"][3]["next_state"]["at___c33"] is True
