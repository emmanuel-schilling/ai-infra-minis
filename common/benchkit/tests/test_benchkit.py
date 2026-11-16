import os
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from benchkit import (
    cpu_slug,
    create_run_dir,
    mac_cpu_features,
    machine_details,
    parse_cmake_cache,
    parse_cpu_speed_limit,
    parse_low_power_mode,
    stay_awake,
    throttled_rounds,
    utc_date,
)


def test_utc_date_names_a_day() -> None:
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", utc_date())


def test_parse_cmake_cache() -> None:
    text = "# comment\n//help\nGGML_METAL:BOOL=OFF\nCMAKE_C_FLAGS:STRING=-g\n\nEMPTY:STRING=\n"
    assert parse_cmake_cache(text) == {"GGML_METAL": "OFF", "CMAKE_C_FLAGS": "-g", "EMPTY": ""}


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
    first = create_run_dir(tmp_path, "2026-11-07_i9-8950hk")
    (first / "round_1.json").write_text("{}")
    second = create_run_dir(tmp_path, "2026-11-07_i9-8950hk")
    third = create_run_dir(tmp_path, "2026-11-07_i9-8950hk")
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


def test_machine_details_always_name_the_cpu() -> None:
    details = machine_details()
    assert details["cpu"]
    assert details["logical_cpus"] == os.cpu_count()


@pytest.mark.skipif(sys.platform != "darwin", reason="sysctl is macOS only")
def test_mac_cpu_features_are_a_sorted_list_even_without_the_keys() -> None:
    # Intel Macs list their features; Apple silicon has no such keys and gets [].
    features = mac_cpu_features()
    assert features == sorted(set(features))
    if platform.machine() == "x86_64":
        assert "FMA" in features


def caffeinate_children() -> int:
    found = subprocess.run(["pgrep", "-P", str(os.getpid()), "caffeinate"], capture_output=True)
    return len(found.stdout.split())


@pytest.mark.skipif(sys.platform != "darwin", reason="caffeinate is macOS only")
def test_stay_awake_holds_caffeinate_for_the_whole_block() -> None:
    assert caffeinate_children() == 0
    with stay_awake():
        assert caffeinate_children() == 1
    assert caffeinate_children() == 0
