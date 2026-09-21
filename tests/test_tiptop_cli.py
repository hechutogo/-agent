"""Parser-only CLI tests (no simulator or network)."""
import pytest

from tiptop_mac.cli import _parser


def test_run_parser_defaults():
    args = _parser().parse_args(["run", "把 A 叠到 B 上"])
    assert args.command == "run"
    assert args.instruction == "把 A 叠到 B 上"
    assert args.seed == 0


def test_run_parser_seed():
    args = _parser().parse_args(["run", "--seed", "7", "把 A 放进盒子"])
    assert args.seed == 7


def test_benchmark_parser_output():
    args = _parser().parse_args(["benchmark", "--output", "out"])
    assert args.command == "benchmark"
    assert str(args.output) == "out"


def test_command_is_required():
    with pytest.raises(SystemExit):
        _parser().parse_args([])
