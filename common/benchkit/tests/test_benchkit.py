import os
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import benchkit
import pytest
from benchkit import (
    ScreenSaverRunning,
    cpu_slug,
    create_run_dir,
    mac_cpu_features,
    machine_details,
    parse_cmake_cache,
    parse_cpu_speed_limit,
    parse_low_power_mode,
    pgrep_found,
    screen_saver_rounds,
    screen_saver_running,
    stay_awake,
    throttled_rounds,
    utc_date,
)


def test_pgrep_found() -> None:
    assert pgrep_found(0) is True
    assert pgrep_found(1) is False
    with pytest.raises(RuntimeError, match="exit status 2"):
        pgrep_found(2)


def test_screen_saver_running_is_a_bool_on_macos() -> None:
    # Tests run with the screen saver off: a running one would have stopped them.
    expected = False if sys.platform == "darwin" else None
    assert screen_saver_running() is expected


def test_screen_saver_problem(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(benchkit, "screen_saver_running", lambda: True)
    assert "dismiss it" in (benchkit.screen_saver_problem() or "")
    for state in (False, None):
        monkeypatch.setattr(benchkit, "screen_saver_running", lambda s=state: s)
        assert benchkit.screen_saver_problem() is None


def test_screen_saver_rounds() -> None:
    def state(running: bool | None) -> dict[str, Any]:
        return {"cpu_speed_limit": 100, "screen_saver_running": running, "raw": ""}

    rounds = [
        {"round": 1, "thermal_before": state(False), "thermal_after": state(False)},
        {"round": 2, "thermal_before": state(False), "thermal_after": state(True)},
        {"round": 3, "thermal_before": None, "thermal_after": {"cpu_speed_limit": 100}},
    ]
    assert screen_saver_rounds(rounds) == [2]


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


@pytest.mark.skipif(sys.platform != "darwin", reason="caffeinate is macOS only")
def test_stay_awake_refuses_a_running_screen_saver(monkeypatch: pytest.MonkeyPatch) -> None:
    # Stands in for a running screen saver: starting the real one locks the screen.
    monkeypatch.setattr(benchkit, "screen_saver_running", lambda: True)
    ran = False
    with pytest.raises(ScreenSaverRunning, match="dismiss it"):
        with stay_awake():
            ran = True
    assert not ran
    assert caffeinate_children() == 0
