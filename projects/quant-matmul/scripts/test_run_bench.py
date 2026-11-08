from pathlib import Path
from typing import Any

import pytest
from run_bench import (
    cpu_slug,
    create_run_dir,
    parse_cpu_speed_limit,
    parse_low_power_mode,
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
