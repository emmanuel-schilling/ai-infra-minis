"""Helpers shared by the projects' benchmark scripts.

Machine details, thermal state, keeping the machine awake, and dated results
folders that are never reused, so every project records its runs the same way.
"""

import contextlib
import datetime
import itertools
import os
import platform
import re
import statistics
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

IGNORED_CPU_WORDS = {"intel", "core", "cpu", "processor"}


def output_of(command: list[str]) -> str:
    return subprocess.run(command, check=True, capture_output=True, text=True).stdout.strip()


def cpu_slug(model: str) -> str:
    """A short folder-name form of a CPU model, e.g. 'i9-8950hk' or 'apple-m2-pro'."""
    text = re.sub(r"\((R|TM)\)", "", model, flags=re.IGNORECASE).split("@")[0]
    words = [w for w in re.split(r"[^A-Za-z0-9-]+", text) if w]
    kept = [w.lower() for w in words if w.lower() not in IGNORED_CPU_WORDS]
    return "-".join(kept) or "unknown-cpu"


def utc_date() -> str:
    """Today's date in UTC, as results folders are named: 2026-11-15."""
    return datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d")


def parse_cmake_cache(text: str) -> dict[str, str]:
    """NAME:TYPE=VALUE lines of a CMakeCache.txt, as {NAME: VALUE}."""
    entries = {}
    for line in text.splitlines():
        if not line or line.startswith(("#", "//")) or "=" not in line:
            continue
        key, _, value = line.partition("=")
        entries[key.split(":", 1)[0]] = value
    return entries


def create_run_dir(results: Path, name: str) -> Path:
    """Create and return a new folder results/<name>, never reusing an existing one.

    If the name is taken, the folder gets a suffix: <name>_2, <name>_3 and so on.
    """
    results.mkdir(parents=True, exist_ok=True)
    for n in itertools.count(1):
        path = results / (name if n == 1 else f"{name}_{n}")
        try:
            path.mkdir()
        except FileExistsError:
            continue
        return path
    raise AssertionError("unreachable")


def parse_cpu_speed_limit(pmset_therm: str) -> int | None:
    """The CPU_Speed_Limit percentage in `pmset -g therm` output, if reported."""
    match = re.search(r"CPU_Speed_Limit\s*=\s*(\d+)", pmset_therm)
    return int(match.group(1)) if match else None


def parse_low_power_mode(pmset_settings: str) -> bool | None:
    """Whether `pmset -g` output reports Low Power Mode on, if it reports it."""
    match = re.search(r"lowpowermode\s+(\d+)", pmset_settings)
    return match.group(1) != "0" if match else None


def cpu_model() -> str:
    # PLATFORM: macOS, Linux, Windows — each reports the CPU model its own way;
    # other systems fall back to platform.processor().
    if sys.platform == "darwin":
        return output_of(["sysctl", "-n", "machdep.cpu.brand_string"])
    if sys.platform.startswith("linux"):
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    if sys.platform == "win32":
        return output_of(
            ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"]
        )
    return platform.processor() or "unknown"


def mac_cpu_features() -> list[str]:
    """The CPU feature flags macOS reports on Intel Macs, such as AVX2 and FMA."""
    # PLATFORM: macOS on Intel — sysctl's machdep.cpu feature keys; Apple
    # silicon has none (it reports hw.optional.* instead), so they come back
    # empty there. Linux would read the flags line of /proc/cpuinfo.
    names: list[str] = []
    for key in ("machdep.cpu.features", "machdep.cpu.leaf7_features"):
        result = subprocess.run(["sysctl", "-n", key], capture_output=True, text=True)
        if result.returncode == 0:
            names += result.stdout.split()
    return sorted(set(names))


def machine_details() -> dict[str, Any]:
    details: dict[str, Any] = {
        "cpu": cpu_model(),
        "logical_cpus": os.cpu_count(),
        "os": platform.platform(),
        "python": platform.python_version(),
    }
    # PLATFORM: macOS — physical cores, memory, CPU features and Low Power
    # Mode come from sysctl and pmset; Linux would need /proc/cpuinfo,
    # /proc/meminfo and lscpu, Windows CIM.
    if sys.platform == "darwin":
        details["physical_cpus"] = int(output_of(["sysctl", "-n", "hw.physicalcpu"]))
        details["memory_bytes"] = int(output_of(["sysctl", "-n", "hw.memsize"]))
        details["cpu_features"] = mac_cpu_features()
        details["low_power_mode"] = parse_low_power_mode(output_of(["pmset", "-g"]))
    return details


# The process a running screen saver shows up as (checked on macOS 15.8 with
# the built-in screen saver). Third-party savers run in a legacyScreenSaver
# process under it, so this one name covers both.
SCREEN_SAVER_PROCESS = "ScreenSaverEngine"


class ScreenSaverRunning(RuntimeError):
    """The screen saver was running when a benchmark run was about to start."""


def pgrep_found(returncode: int) -> bool:
    """pgrep's exit status as found (0) or not found (1); anything else is an error."""
    if returncode not in (0, 1):
        raise RuntimeError(f"pgrep failed with exit status {returncode}")
    return returncode == 0


def screen_saver_running() -> bool | None:
    """Whether the screen saver is running now; None where this can't be checked."""
    # PLATFORM: macOS — the screen saver is a process, ScreenSaverEngine. Linux
    # has none to check: it would depend on the desktop environment, if any.
    if sys.platform != "darwin":
        return None
    found = subprocess.run(["pgrep", "-x", SCREEN_SAVER_PROCESS], capture_output=True)
    return pgrep_found(found.returncode)


SCREEN_SAVER_MESSAGE = (
    "the screen saver is running; dismiss it (move the mouse or press a key) "
    "and start the run again"
)


def screen_saver_problem() -> str | None:
    """Why a run must not start now, if the screen saver is running; else None.

    For scripts to check before they create anything. stay_awake checks again
    once caffeinate holds the display awake.
    """
    return SCREEN_SAVER_MESSAGE if screen_saver_running() else None


def thermal_state() -> dict[str, Any] | None:
    """The CPU speed limit and whether the screen saver is running, at this moment."""
    # PLATFORM: macOS — pmset reports the CPU speed limit under thermal
    # pressure; Linux would read /sys/devices/system/cpu/*/cpufreq instead.
    if sys.platform != "darwin":
        return None
    text = output_of(["pmset", "-g", "therm"])
    return {
        "cpu_speed_limit": parse_cpu_speed_limit(text),
        "screen_saver_running": screen_saver_running(),
        "raw": text,
    }


def median_ratios_to_first(rounds: list[dict[str, float]]) -> list[float]:
    """Each round's median, over the measurements it shares with round 1, of value / round 1's.

    Each round maps a measurement's name to its value, such as a speed or a
    time. A round sharing nothing with round 1 gets 1.0.
    """
    if not rounds:
        return []
    first = rounds[0]
    ratios = []
    for current in rounds:
        common = [k for k in current if k in first and first[k] > 0]
        ratios.append(statistics.median(current[k] / first[k] for k in common) if common else 1.0)
    return ratios


def rounds_beyond(ratios: list[float], tolerance: float) -> list[int]:
    """1-based numbers of the rounds whose ratio to round 1 is off by more than tolerance."""
    return [i + 1 for i, ratio in enumerate(ratios) if abs(ratio - 1) > tolerance]


def throttled_rounds(rounds: list[dict[str, Any]]) -> list[int]:
    """Rounds where macOS reported a CPU speed limit below 100% before or after.

    Takes one record per round, or several per round (one per test, say), each
    with its round number and the thermal state before and after.
    """
    limited = set()
    for r in rounds:
        for state in (r["thermal_before"], r["thermal_after"]):
            if state and state["cpu_speed_limit"] is not None and state["cpu_speed_limit"] < 100:
                limited.add(r["round"])
    return sorted(limited)


def started_throttled(record: dict[str, Any]) -> bool:
    """Whether macOS reported a CPU speed limit below 100% before this test or round began."""
    state = record["thermal_before"]
    return bool(state and state["cpu_speed_limit"] is not None and state["cpu_speed_limit"] < 100)


def started_throttled_rounds(rounds: list[dict[str, Any]]) -> list[int]:
    """Rounds with a test (or the round itself) that started under a CPU speed limit.

    Unlike throttled_rounds, a limit that only set in during a test doesn't
    count: under sustained all-core load a laptop can reach its power and
    thermal limits within seconds, which is its real behaviour, not a bad
    starting condition.
    """
    return sorted({r["round"] for r in rounds if started_throttled(r)})


def screen_saver_rounds(rounds: list[dict[str, Any]]) -> list[int]:
    """Rounds where the screen saver was found running before or after."""
    found = set()
    for r in rounds:
        for state in (r["thermal_before"], r["thermal_after"]):
            if state and state.get("screen_saver_running"):
                found.add(r["round"])
    return sorted(found)


@contextlib.contextmanager
def stay_awake() -> Iterator[None]:
    """Keep the display and system awake for the whole block, cooldowns included.

    Raises ScreenSaverRunning, before the block runs, if the screen saver is
    already running: keeping the display awake stops it from starting, not a
    running one, which slows the CPU (by 1.3-3.4x in Quantized matmul's
    2026-11-10 run). The check comes after caffeinate starts, so the screen
    saver can't start between the check and the run.
    """
    # PLATFORM: macOS — caffeinate keeps the display and system awake, so
    # neither the screen saver nor sleep starts during the run; -w ends it if
    # this script dies. Linux could use systemd-inhibit, and Windows
    # SetThreadExecutionState.
    if sys.platform != "darwin":
        yield
        return
    awake = subprocess.Popen(["caffeinate", "-d", "-i", "-w", str(os.getpid())])
    try:
        if screen_saver_running():
            raise ScreenSaverRunning(SCREEN_SAVER_MESSAGE)
        yield
    finally:
        awake.terminate()
        awake.wait()
