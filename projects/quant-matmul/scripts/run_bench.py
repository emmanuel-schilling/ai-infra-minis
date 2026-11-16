"""Run the Quantized matmul benchmarks and keep the raw results.

Builds the release preset, then runs qmm_bench for several rounds with a
cooldown between them, and writes everything to a new folder
results/<date>_<cpu>/ in this project:

- metadata.json: hardware, OS, compiler, flags, the source revision, the
  command, the thermal state and speed check before and after each round, and
  each round's time relative to round 1
- round_<n>.json: Google Benchmark's raw output for each round
- round_<n>_check_{before,after}.json: the speed check's raw output
- summary.json: the median of every benchmark across all rounds

Each round is compared with round 1 through its own results: the median, over
all benchmarks, of the ratio of their times. A round more than 5% slower or
faster than round 1 is listed in slow_rounds: something else changed the
CPU's speed, such as a screen saver or a background task, even when the
operating system reports no throttling.

The speed check times one short benchmark before and after every round. It is
recorded as a diagnostic, for instance of what happened during a cooldown, but
doesn't exclude rounds: it follows the clock speed of the few seconds it runs.

Earlier runs are never overwritten. The BLAS baselines run single-threaded,
like the kernels they are compared with: Accelerate through
VECLIB_MAXIMUM_THREADS=1, and OpenBLAS is built without threads (with
OPENBLAS_NUM_THREADS=1 set as well).

Usage, from anywhere in the repository:
    uv run python projects/quant-matmul/scripts/run_bench.py --help
"""

import argparse
import datetime
import json
import os
import re
import shlex
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from benchkit import (
    cpu_slug,
    create_run_dir,
    machine_details,
    output_of,
    stay_awake,
    thermal_state,
    throttled_rounds,
)

PROJECT = Path(__file__).resolve().parents[1]
REPO = PROJECT.parents[1]
BUILD = REPO / "build-release"
RESULTS = PROJECT / "results"

NS_PER_UNIT = {"ns": 1.0, "us": 1e3, "ms": 1e6, "s": 1e9}

# The speed check: short, built on every platform, and bound by the latency of
# fp32 additions, so its time follows the CPU's clock speed.
SPEED_CHECK = "fp32/scalar/attn_out/tokens:128"
SPEED_CHECK_TOLERANCE = 0.05

# A round whose results differ from round 1's by more than this is excluded.
ROUND_TOLERANCE = 0.05


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


def check_time_ns(output: dict[str, Any], name: str) -> float | None:
    """The median time of benchmark `name` in one Google Benchmark output, if it ran."""
    return summarize([output]).get(name, {}).get("time_ns")


def round_speeds(outputs: list[dict[str, Any]]) -> list[float | None]:
    """Each round's time relative to round 1: the median over benchmarks of the time ratio.

    Only benchmarks measured in both rounds count. None for a round that
    shares no benchmark with round 1.
    """
    if not outputs:
        return []
    medians = [summarize([output]) for output in outputs]
    first = medians[0]
    speeds: list[float | None] = []
    for current in medians:
        ratios = [
            current[name]["time_ns"] / first[name]["time_ns"]
            for name in current
            if name in first and first[name]["time_ns"] > 0
        ]
        speeds.append(statistics.median(ratios) if ratios else None)
    return speeds


def slow_rounds(speeds: list[float | None], tolerance: float) -> list[int]:
    """Rounds whose results differ from round 1's by more than tolerance.

    Every round runs every benchmark for minutes, so a change that lasts
    through a round, such as a screen saver that starts between rounds, moves
    the median of all the ratios, while a brief one doesn't.
    """
    return [
        n
        for n, speed in enumerate(speeds, start=1)
        if speed is not None and abs(speed - 1.0) > tolerance
    ]


def check_changed_rounds(rounds: list[dict[str, Any]], tolerance: float) -> list[int]:
    """Rounds whose speed check before or after differs from the run's first by more than tolerance.

    A diagnostic only: a check lasts a few seconds and follows the CPU's clock
    speed at that moment, which changes with what ran just before it, so it
    can differ from the first while the rounds' results agree.
    """
    times = [(r["round"], t) for r in rounds for t in (r["check_before_ns"], r["check_after_ns"])]
    known = [(n, t) for n, t in times if t is not None]
    if not known:
        return []
    reference = known[0][1]
    return sorted({n for n, t in known if abs(t / reference - 1.0) > tolerance})


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


def source_revision(repo: Path, results: Path) -> dict[str, Any]:
    """The checked-out revision, and whether anything outside `results` differs from it.

    Benchmark results are left out, so earlier runs that aren't committed yet
    don't mark the source as changed.
    """
    head = output_of(["git", "-C", str(repo), "rev-parse", "HEAD"])
    excluded = f":(exclude){results.relative_to(repo).as_posix()}"
    status = output_of(["git", "-C", str(repo), "status", "--porcelain", "--", ".", excluded])
    return {"revision": head, "uncommitted_changes": status != ""}


def bench_command(exe: Path, out: Path, repetitions: int, bench_filter: str) -> list[str]:
    return [
        str(exe),
        f"--benchmark_repetitions={repetitions}",
        "--benchmark_enable_random_interleaving=true",
        f"--benchmark_filter={bench_filter}",
        f"--benchmark_out={out}",
        "--benchmark_out_format=json",
    ]


def speed_check(exe: Path, out: Path, env: dict[str, str]) -> float | None:
    """Run the speed check, keep its raw output in `out`, and return its median time."""
    command = bench_command(exe, out, 3, f"^{re.escape(SPEED_CHECK)}$")
    subprocess.run(command, check=True, env=env, stdout=subprocess.DEVNULL)
    return check_time_ns(json.loads(out.read_text()), SPEED_CHECK)


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
    source = source_revision(REPO, RESULTS)
    started = datetime.datetime.now(datetime.UTC)
    run_dir = create_run_dir(RESULTS, f"{started.date().isoformat()}_{cpu_slug(machine['cpu'])}")
    single_thread = {"VECLIB_MAXIMUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    env = os.environ | single_thread

    rounds: list[dict[str, Any]] = []
    outputs: list[dict[str, Any]] = []
    with stay_awake():
        for n in range(1, args.rounds + 1):
            out = run_dir / f"round_{n}.json"
            command = bench_command(exe, out, args.repetitions, args.filter)
            before = thermal_state()
            check_before = speed_check(exe, run_dir / f"round_{n}_check_before.json", env)
            print(f"run_bench: round {n} of {args.rounds}: {shlex.join(command)}", flush=True)
            subprocess.run(command, check=True, env=env)
            check_after = speed_check(exe, run_dir / f"round_{n}_check_after.json", env)
            rounds.append(
                {
                    "round": n,
                    "command": command,
                    "thermal_before": before,
                    "thermal_after": thermal_state(),
                    "check_before_ns": check_before,
                    "check_after_ns": check_after,
                }
            )
            outputs.append(json.loads(out.read_text()))
            if n < args.rounds:
                time.sleep(args.cooldown)

    throttled = throttled_rounds(rounds)
    speeds = round_speeds(outputs)
    for entry, speed in zip(rounds, speeds, strict=True):
        entry["time_vs_round_1"] = speed
    slow = slow_rounds(speeds, ROUND_TOLERANCE)
    check_changed = check_changed_rounds(rounds, SPEED_CHECK_TOLERANCE)
    metadata = {
        "started_utc": started.isoformat(),
        "finished_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "script_command": [sys.executable, *sys.argv],
        "settings": vars(args),
        "environment": single_thread,
        "machine": machine,
        "build": build_details(),
        "source": source,
        "speed_check": {
            "benchmark": SPEED_CHECK,
            "tolerance": SPEED_CHECK_TOLERANCE,
            "changed_rounds": check_changed,
        },
        "round_tolerance": ROUND_TOLERANCE,
        "rounds": rounds,
        "throttled_rounds": throttled,
        "slow_rounds": slow,
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    summary = summarize(outputs)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print(f"\nrun_bench: results in {run_dir}")
    if throttled:
        print(f"run_bench: WARNING: the CPU speed was limited in rounds {throttled}")
    if slow:
        change = f"{ROUND_TOLERANCE:.0%}"
        print(f"run_bench: WARNING: rounds {slow} differ from round 1 by more than {change}")
    print(f"run_bench: time relative to round 1, per round: {speeds}")
    if check_changed:
        change = f"{SPEED_CHECK_TOLERANCE:.0%}"
        print(f"run_bench: note: the speed check changed by more than {change} in {check_changed}")
    if all(r["check_before_ns"] is None for r in rounds):
        print(f"run_bench: WARNING: the speed check {SPEED_CHECK} didn't run")
    print(f"{'benchmark':<52} {'median ms':>10} {'GFLOP/s':>9} {'GB/s':>8}")
    for name, values in summary.items():
        gflops = values.get("FLOP/s", 0.0) / 1e9
        gbps = values.get("bytes_per_second", 0.0) / 1e9
        print(f"{name:<52} {values['time_ns'] / 1e6:>10.3f} {gflops:>9.2f} {gbps:>8.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
