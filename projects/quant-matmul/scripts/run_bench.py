"""Run the Quantized matmul benchmarks and keep the raw results.

Builds the release preset, then runs qmm_bench for several rounds with a
cooldown between them, and writes everything to a new folder
results/<date>_<cpu>/ in this project:

- metadata.json: hardware, OS, compiler, flags, the source revision, the
  command, and the thermal state before and after each round
- round_<n>.json: Google Benchmark's raw output for each round
- summary.json: the median of every benchmark across all rounds

Earlier runs are never overwritten. The BLAS baselines run single-threaded,
like the kernels they are compared with: Accelerate through
VECLIB_MAXIMUM_THREADS=1, and OpenBLAS is built without threads (with
OPENBLAS_NUM_THREADS=1 set as well).

Usage, from anywhere in the repository:
    uv run python projects/quant-matmul/scripts/run_bench.py --help
"""

import argparse
import datetime
import itertools
import json
import os
import platform
import re
import shlex
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

PROJECT = Path(__file__).resolve().parents[1]
REPO = PROJECT.parents[1]
BUILD = REPO / "build-release"
RESULTS = PROJECT / "results"

NS_PER_UNIT = {"ns": 1.0, "us": 1e3, "ms": 1e6, "s": 1e9}
IGNORED_CPU_WORDS = {"intel", "core", "cpu", "processor"}


def cpu_slug(model: str) -> str:
    """A short folder-name form of a CPU model, e.g. 'i9-8950hk' or 'apple-m2-pro'."""
    text = re.sub(r"\((R|TM)\)", "", model, flags=re.IGNORECASE).split("@")[0]
    words = [w for w in re.split(r"[^A-Za-z0-9-]+", text) if w]
    kept = [w.lower() for w in words if w.lower() not in IGNORED_CPU_WORDS]
    return "-".join(kept) or "unknown-cpu"


def create_run_dir(results: Path, date: str, slug: str) -> Path:
    """Create and return a new folder for this run, never reusing an existing one."""
    results.mkdir(parents=True, exist_ok=True)
    for n in itertools.count(1):
        path = results / (f"{date}_{slug}" if n == 1 else f"{date}_{slug}_{n}")
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


def summarize(rounds: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Medians per benchmark over every repetition of every round.

    Aggregate rows (mean, median, ...) and skipped or failed benchmarks are
    left out, so each sample is one measured repetition.
    """
    samples: dict[str, dict[str, list[float]]] = {}
    for round_output in rounds:
        for row in round_output["benchmarks"]:
            if row.get("run_type") != "iteration" or row.get("error_occurred"):
                continue
            if row.get("skipped"):
                continue
            entry = samples.setdefault(row.get("run_name", row["name"]), {})
            entry.setdefault("time_ns", []).append(row["real_time"] * NS_PER_UNIT[row["time_unit"]])
            for counter in ("FLOP/s", "bytes_per_second"):
                if counter in row:
                    entry.setdefault(counter, []).append(row[counter])
    return {
        name: {key: statistics.median(values) for key, values in entry.items()}
        | {"samples": float(len(entry["time_ns"]))}
        for name, entry in samples.items()
    }


def output_of(command: list[str]) -> str:
    return subprocess.run(command, check=True, capture_output=True, text=True).stdout.strip()


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


def machine_details() -> dict[str, Any]:
    details: dict[str, Any] = {
        "cpu": cpu_model(),
        "logical_cpus": os.cpu_count(),
        "os": platform.platform(),
        "python": platform.python_version(),
    }
    # PLATFORM: macOS — physical cores, memory and Low Power Mode come from
    # sysctl and pmset; Linux would need /proc/meminfo and lscpu, Windows CIM.
    if sys.platform == "darwin":
        details["physical_cpus"] = int(output_of(["sysctl", "-n", "hw.physicalcpu"]))
        details["memory_bytes"] = int(output_of(["sysctl", "-n", "hw.memsize"]))
        details["low_power_mode"] = parse_low_power_mode(output_of(["pmset", "-g"]))
    return details


def thermal_state() -> dict[str, Any] | None:
    # PLATFORM: macOS — pmset reports the CPU speed limit under thermal
    # pressure; Linux would read /sys/devices/system/cpu/*/cpufreq instead.
    if sys.platform != "darwin":
        return None
    text = output_of(["pmset", "-g", "therm"])
    return {"cpu_speed_limit": parse_cpu_speed_limit(text), "raw": text}


def build_details() -> dict[str, Any]:
    """The compiler and the exact compile commands of the release build."""
    wanted = {
        "CMAKE_CXX_COMPILER",
        "CMAKE_BUILD_TYPE",
        "CMAKE_CXX_FLAGS",
        "CMAKE_CXX_FLAGS_RELEASE",
    }
    settings = {}
    for line in (BUILD / "CMakeCache.txt").read_text().splitlines():
        name, _, value = line.partition("=")
        if name.split(":", 1)[0] in wanted:
            settings[name.split(":", 1)[0]] = value
    compiler: dict[str, str] = {}
    for info in BUILD.glob("CMakeFiles/*/CMakeCXXCompiler.cmake"):
        for key in ("CMAKE_CXX_COMPILER_ID", "CMAKE_CXX_COMPILER_VERSION"):
            match = re.search(rf'set\({key} "([^"]*)"\)', info.read_text())
            if match:
                compiler[key] = match.group(1)
    commands = json.loads((BUILD / "compile_commands.json").read_text())
    ours = [c for c in commands if Path(c["file"]).is_relative_to(PROJECT)]
    return {"cache": settings, "compiler": compiler, "compile_commands": ours}


def source_revision() -> dict[str, Any]:
    head = output_of(["git", "-C", str(REPO), "rev-parse", "HEAD"])
    dirty = output_of(["git", "-C", str(REPO), "status", "--porcelain"]) != ""
    return {"revision": head, "uncommitted_changes": dirty}


def bench_command(exe: Path, out: Path, repetitions: int, bench_filter: str) -> list[str]:
    command = [
        str(exe),
        f"--benchmark_repetitions={repetitions}",
        "--benchmark_enable_random_interleaving=true",
        f"--benchmark_filter={bench_filter}",
        f"--benchmark_out={out}",
        "--benchmark_out_format=json",
    ]
    # PLATFORM: macOS — caffeinate keeps the display and system awake, so
    # neither the screen saver nor sleep interrupts the run; Linux could use
    # systemd-inhibit, and Windows SetThreadExecutionState.
    if sys.platform == "darwin":
        return ["caffeinate", "-d", "-i", *command]
    return command


def throttled_rounds(rounds: list[dict[str, Any]]) -> list[int]:
    """Rounds where macOS reported a CPU speed limit below 100% before or after."""
    limited = set()
    for r in rounds:
        for state in (r["thermal_before"], r["thermal_after"]):
            if state and state["cpu_speed_limit"] is not None and state["cpu_speed_limit"] < 100:
                limited.add(r["round"])
    return sorted(limited)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Quantized matmul benchmarks.")
    parser.add_argument("--rounds", type=int, default=3, help="separate runs (default 3)")
    parser.add_argument("--repetitions", type=int, default=5, help="repetitions per round")
    parser.add_argument("--cooldown", type=float, default=60.0, help="seconds between rounds")
    parser.add_argument("--filter", default=".", help="regular expression of benchmarks to run")
    args = parser.parse_args()

    if "VCPKG_ROOT" not in os.environ:
        print("run_bench: set VCPKG_ROOT to your vcpkg checkout", file=sys.stderr)
        return 2
    subprocess.run(["cmake", "--preset", "release"], cwd=REPO, check=True)
    subprocess.run(
        ["cmake", "--build", "--preset", "release", "--target", "qmm_bench"], cwd=REPO, check=True
    )
    exe_name = "qmm_bench.exe" if os.name == "nt" else "qmm_bench"
    exe = BUILD / "projects" / "quant-matmul" / exe_name

    machine = machine_details()
    if machine.get("low_power_mode"):
        print("run_bench: WARNING: Low Power Mode is on, which lowers CPU speed", file=sys.stderr)
    started = datetime.datetime.now(datetime.UTC)
    run_dir = create_run_dir(RESULTS, started.date().isoformat(), cpu_slug(machine["cpu"]))
    single_thread = {"VECLIB_MAXIMUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    env = os.environ | single_thread

    rounds: list[dict[str, Any]] = []
    outputs: list[dict[str, Any]] = []
    for n in range(1, args.rounds + 1):
        out = run_dir / f"round_{n}.json"
        command = bench_command(exe, out, args.repetitions, args.filter)
        before = thermal_state()
        print(f"run_bench: round {n} of {args.rounds}: {shlex.join(command)}", flush=True)
        subprocess.run(command, check=True, env=env)
        rounds.append(
            {
                "round": n,
                "command": command,
                "thermal_before": before,
                "thermal_after": thermal_state(),
            }
        )
        outputs.append(json.loads(out.read_text()))
        if n < args.rounds:
            time.sleep(args.cooldown)

    throttled = throttled_rounds(rounds)
    metadata = {
        "started_utc": started.isoformat(),
        "finished_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "script_command": [sys.executable, *sys.argv],
        "settings": vars(args),
        "environment": single_thread,
        "machine": machine,
        "build": build_details(),
        "source": source_revision(),
        "rounds": rounds,
        "throttled_rounds": throttled,
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    summary = summarize(outputs)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print(f"\nrun_bench: results in {run_dir}")
    if throttled:
        print(f"run_bench: WARNING: the CPU speed was limited in rounds {throttled}")
    print(f"{'benchmark':<52} {'median ms':>10} {'GFLOP/s':>9} {'GB/s':>8}")
    for name, values in summary.items():
        gflops = values.get("FLOP/s", 0.0) / 1e9
        gbps = values.get("bytes_per_second", 0.0) / 1e9
        print(f"{name:<52} {values['time_ns'] / 1e6:>10.3f} {gflops:>9.2f} {gbps:>8.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
