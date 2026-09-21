"""Tests for the two-architecture benchmark harness."""
import json

from tiptop_mac.benchmark import (
    BENCHMARK_CASES, Benchmark, Case, CaseResult,
)


def test_cases_cover_required_seeds_and_relations():
    seeds = {(c.seed, c.relation) for c in BENCHMARK_CASES}
    assert {(0, "on"), (3, "on"), (7, "in")} <= seeds


def test_benchmark_writes_results_and_report(tmp_path):
    def good(case):
        return CaseResult("tiptop_mac", case.seed, "completed", True, False,
                          1.25, case.relation, "目标达成")

    def skipped(case):
        return CaseResult("pickparts", case.seed, "skipped", False, False,
                          0.0, "", "Missing cloud configuration")

    case = Case(0, "把 A 叠到 B 上", "A", "B", "on")
    output = Benchmark((case,),
                       runners={"tiptop_mac": good, "pickparts": skipped}
                       ).run(tmp_path)
    data = json.loads((output / "results.json").read_text())
    assert len(data["runs"]) == 2
    assert data["runs"][0]["success"] is True
    assert data["runs"][1]["status"] == "skipped"
    report = (output / "report.md").read_text()
    assert "tiptop_mac" in report and "pickparts" in report
    assert "skipped" in report


def test_benchmark_records_aborted_runs(tmp_path):
    def aborted(case):
        return CaseResult("tiptop_mac", case.seed, "completed", False, True,
                          .5, "", "轨迹执行中止")

    case = BENCHMARK_CASES[0]
    output = Benchmark((case,), runners={"tiptop_mac": aborted}).run(tmp_path)
    data = json.loads((output / "results.json").read_text())
    assert data["runs"][0]["aborted"] is True
