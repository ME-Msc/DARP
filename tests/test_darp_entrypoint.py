import pytest

from darp.__main__ import build_parser


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
