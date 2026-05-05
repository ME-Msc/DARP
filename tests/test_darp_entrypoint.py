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
    assert "--visualizer" in help_text
    assert "--simulator" in help_text
    assert "solve" not in help_text


def test_darp_default_trace_arguments_parse():
    """Check top-level trace arguments parse cleanly. / 检查顶层 trace 参数能正确解析。"""
    args = build_parser().parse_args(
        [
            "--domain",
            "domain.rddl",
            "--instance",
            "instance.rddl",
            "--simulator",
            "darp",
            "--frontend",
            "darp",
            "--seed",
            "3",
            "--host",
            "127.0.0.1",
            "--port",
            "8080",
            "--no-open",
        ]
    )

    assert args.visualizer is False
    assert args.mode == "online"
    assert args.domain == "domain.rddl"
    assert args.instance == "instance.rddl"
    assert args.simulator == "darp"
    assert args.frontend == "darp"
    assert args.seed == 3
    assert args.port == 8080
    assert args.no_open is True


def test_darp_legacy_with_simulator_alias_parse():
    """Check legacy simulator alias still maps to the runtime simulator. / 检查旧 simulator 别名仍会映射到运行时 simulator。"""
    args = build_parser().parse_args(
        [
            "--domain",
            "domain.rddl",
            "--instance",
            "instance.rddl",
            "--with-simulator",
            "rddlgym",
        ]
    )

    assert args.simulator == "rddlgym"


def test_darp_visualizer_invokes_visualizer(monkeypatch):
    """Check `--visualizer` starts the live UI. / 检查 `--visualizer` 会启动实时界面。"""
    called = {}

    def fake_serve_visualizer(**kwargs):
        """Capture visualizer arguments without opening a server. / 捕获 visualizer 参数而不启动服务。"""
        called.update(kwargs)
        return 0

    monkeypatch.setattr("darp.__main__.serve_visualizer", fake_serve_visualizer)
    exit_code = main(
        [
            "--visualizer",
            "--domain",
            "domain.rddl",
            "--instance",
            "instance.rddl",
            "--no-open",
        ]
    )

    assert exit_code == 0
    assert called["domain"] == "domain.rddl"
    assert called["instance"] == "instance.rddl"
    assert called["simulator"] == "darp"
    assert called["frontend"] == "darp"
    assert called["seed"] == 0
    assert called["open_browser"] is False


def test_darp_typo_visualizer_argument_errors():
    """Check misspelled visualizer arguments fail clearly. / 检查拼错的 visualizer 参数会明确报错。"""
    with pytest.raises(SystemExit) as exc_info:
        build_parser().parse_args(["--visulaizer"])

    assert exc_info.value.code == 2


def test_darp_top_level_outputs_trace(capsys):
    """Check top-level DARP prints a terminal trace without JSON. / 检查顶层 DARP 默认打印非 JSON 终端轨迹。"""
    exit_code = main(["--mode", "online", "--seed", "7"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert captured.out.startswith("DARP online trace")
    assert "action=safe_path" in captured.out
    assert "Total reward:" in captured.out
    with pytest.raises(json.JSONDecodeError):
        json.loads(captured.out)


def test_darp_output_writes_json_file(tmp_path, capsys):
    """Check JSON is written only when `--output` is provided. / 检查只有提供 `--output` 时才写 JSON。"""
    output = tmp_path / "trace.json"
    exit_code = main(["--mode", "online", "--seed", "7", "--output", str(output)])
    captured = capsys.readouterr()
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert captured.out.startswith("DARP online trace")
    assert payload["mode"] == "online"
    assert payload["planner"] == "finite-horizon-dp"
    assert len(payload["steps"]) == payload["max_depth"]
    assert payload["steps"][0]["action"] == "safe_path"
