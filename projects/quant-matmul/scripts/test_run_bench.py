import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from run_bench import (
    check_changed_rounds,
    check_time_ns,
    cpu_slug,
    create_run_dir,
    parse_cpu_speed_limit,
    parse_low_power_mode,
    round_speeds,
    slow_rounds,
    source_revision,
    stay_awake,
    summarize,
    throttled_rounds,
)


@pytest.mark.parametrize(
    ("model", "slug"),
    [
        ("Intel(R) Core(TM) i9-8950HK CPU @ 2.90GHz", "i9-8950hk"),
        ("Apple M2 Pro", "apple-m2-pro"),
        ("AMD Ryzen 9 7950X 16-Core Processor", "amd-ryzen-9-7950x-16-core"),
        ("", "unknown-cpu"),
    ],
)
def test_cpu_slug(model: str, slug: str) -> None:
    assert cpu_slug(model) == slug


def test_create_run_dir_never_reuses_a_folder(tmp_path: Path) -> None:
    first = create_run_dir(tmp_path, "2026-11-07", "i9-8950hk")
    (first / "round_1.json").write_text("{}")
    second = create_run_dir(tmp_path, "2026-11-07", "i9-8950hk")
    third = create_run_dir(tmp_path, "2026-11-07", "i9-8950hk")
    assert first.name == "2026-11-07_i9-8950hk"
    assert second.name == "2026-11-07_i9-8950hk_2"
    assert third.name == "2026-11-07_i9-8950hk_3"
    assert (first / "round_1.json").read_text() == "{}"


def test_parse_cpu_speed_limit() -> None:
    text = "Note: No thermal warning level has been recorded\n\tCPU_Speed_Limit \t= 87\n"
    assert parse_cpu_speed_limit(text) == 87
    assert parse_cpu_speed_limit("Note: No thermal warning level has been recorded") is None


def test_parse_low_power_mode() -> None:
    assert parse_low_power_mode(" lowpowermode         0\n") is False
    assert parse_low_power_mode(" lowpowermode         1\n") is True
    assert parse_low_power_mode("sleep 1") is None


def test_throttled_rounds_reports_any_limit_below_100() -> None:
    def state(limit: int | None) -> dict[str, Any]:
        return {"cpu_speed_limit": limit, "raw": ""}

    rounds = [
        {"round": 1, "thermal_before": state(100), "thermal_after": state(100)},
        {"round": 2, "thermal_before": state(100), "thermal_after": state(87)},
        {"round": 3, "thermal_before": state(None), "thermal_after": None},
    ]
    assert throttled_rounds(rounds) == [2]


def row(
    name: str, real_time: float, unit: str = "ms", extra: dict[str, object] | None = None
) -> dict[str, object]:
    return {
        "name": name,
        "run_name": name,
        "run_type": "iteration",
        "real_time": real_time,
        "time_unit": unit,
    } | (extra or {})


def test_summarize_takes_medians_across_rounds_and_skips_aggregates() -> None:
    rounds: list[dict[str, Any]] = [
        {
            "benchmarks": [
                row("gemv", 1.0, extra={"FLOP/s": 10.0}),
                row("gemv", 3.0, extra={"FLOP/s": 30.0}),
                {
                    "name": "gemv_median",
                    "run_name": "gemv",
                    "run_type": "aggregate",
                    "real_time": 99.0,
                    "time_unit": "ms",
                },
            ]
        },
        {
            "benchmarks": [
                row("gemv", 2.0, extra={"FLOP/s": 20.0}),
                row("bandwidth", 500.0, unit="us", extra={"bytes_per_second": 4e9}),
                row("broken", 1.0, extra={"error_occurred": True}),
                row("no_avx2", 1.0, extra={"skipped": True}),
            ]
        },
    ]
    summary = summarize(rounds)
    assert summary["gemv"] == {"time_ns": 2e6, "FLOP/s": 20.0, "samples": 3.0}
    assert summary["bandwidth"] == {"time_ns": 5e5, "bytes_per_second": 4e9, "samples": 1.0}
    assert "broken" not in summary
    assert "no_avx2" not in summary


def test_check_time_ns_takes_the_median_of_the_named_benchmark() -> None:
    output: dict[str, Any] = {
        "benchmarks": [row("check", 2.0), row("check", 1.0), row("check", 3.0), row("other", 9.0)]
    }
    assert check_time_ns(output, "check") == 2e6
    assert check_time_ns(output, "missing") is None


def checks(*times: tuple[float | None, float | None]) -> list[dict[str, Any]]:
    return [
        {"round": n, "check_before_ns": before, "check_after_ns": after}
        for n, (before, after) in enumerate(times, start=1)
    ]


def test_check_changed_rounds_compares_every_check_with_the_first() -> None:
    assert check_changed_rounds(checks((100.0, 104.0), (96.0, 100.0)), 0.05) == []
    assert check_changed_rounds(checks((100.0, 101.0), (100.0, 130.0)), 0.05) == [2]
    assert check_changed_rounds(checks((100.0, 100.0), (190.0, 190.0), (190.0, 190.0)), 0.05) == [
        2,
        3,
    ]
    assert check_changed_rounds(checks((100.0, 100.0), (70.0, 100.0)), 0.05) == [2]


def test_check_changed_rounds_skips_checks_that_did_not_run() -> None:
    assert check_changed_rounds(checks((None, None), (None, None)), 0.05) == []
    assert check_changed_rounds(checks((None, 100.0), (100.0, 200.0)), 0.05) == [2]


def round_output(**times_ms: float) -> dict[str, Any]:
    return {"benchmarks": [row(name, t) for name, t in times_ms.items()]}


def test_round_speeds_take_the_median_ratio_to_round_1() -> None:
    speeds = round_speeds(
        [
            round_output(a=10.0, b=20.0, c=40.0),
            # One benchmark much slower and one slightly faster: the median
            # follows the typical benchmark, not the outlier.
            round_output(a=10.0, b=60.0, c=39.0),
            round_output(a=20.0, b=40.0, c=80.0),
        ]
    )
    assert speeds == [1.0, 1.0, 2.0]


def test_round_speeds_use_only_benchmarks_in_both_rounds() -> None:
    speeds = round_speeds(
        [round_output(a=10.0), round_output(a=11.0, new=99.0), round_output(b=1.0)]
    )
    assert speeds == [1.0, pytest.approx(1.1), None]
    assert round_speeds([]) == []


def test_slow_rounds_flags_rounds_beyond_the_tolerance_either_way() -> None:
    # The invalid 2026-11-12 run's rounds: its speed check flagged 2 and 3.
    assert slow_rounds([1.0, 0.991, 0.996], 0.05) == []
    # The screen saver run: rounds 2 and 3 about twice as slow.
    assert slow_rounds([1.0, 1.9, 2.1], 0.05) == [2, 3]
    assert slow_rounds([1.0, 0.94, 1.04], 0.05) == [2]
    assert slow_rounds([1.0, None], 0.05) == []


def git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


def test_source_revision_ignores_benchmark_results(tmp_path: Path) -> None:
    git(tmp_path, "init")
    (tmp_path / "kernel.cpp").write_text("one\n")
    git(tmp_path, "add", "kernel.cpp")
    identity = ["-c", "user.name=test", "-c", "user.email=test@example.invalid"]
    git(tmp_path, *identity, "-c", "commit.gpgsign=false", "commit", "-m", "start")
    results = tmp_path / "project" / "results"
    (results / "2026-11-10_cpu").mkdir(parents=True)
    (results / "2026-11-10_cpu" / "round_1.json").write_text("{}")

    clean = source_revision(tmp_path, results)
    assert clean["uncommitted_changes"] is False
    assert len(clean["revision"]) >= 40

    (tmp_path / "kernel.cpp").write_text("two\n")
    assert source_revision(tmp_path, results)["uncommitted_changes"] is True


def caffeinate_children() -> int:
    found = subprocess.run(["pgrep", "-P", str(os.getpid()), "caffeinate"], capture_output=True)
    return len(found.stdout.split())


@pytest.mark.skipif(sys.platform != "darwin", reason="caffeinate is macOS only")
def test_stay_awake_holds_caffeinate_for_the_whole_block() -> None:
    assert caffeinate_children() == 0
    with stay_awake():
        assert caffeinate_children() == 1
    assert caffeinate_children() == 0
