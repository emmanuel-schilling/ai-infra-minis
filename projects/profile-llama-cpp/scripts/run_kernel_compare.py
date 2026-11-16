"""Time ggml's matrix multiply on GPT-2's shapes, against Quantized matmul's kernels.

Runs llama.cpp's test-backend-ops in perf mode on the CPU backend, with test
cases written to a file: a matrix multiply of each of GPT-2's weight shapes
(n x k) by 1, 128 and 1,024 tokens, as Quantized matmul benchmarks them (the
output layer only up to 128), with Q8_0 weights and with fp32 weights. With
f32 activations, ggml quantizes them to Q8_0 blocks itself for a Q8_0 weight,
as Quantized matmul's Q8 kernel does.

Single-threaded by default, like Quantized matmul's kernels:
test-backend-ops runs half of TEST_ARG_HARDWARE_CONCURRENCY threads, so it is
set to twice --threads.

Writes a new folder results/<date>_<cpu>_kernels[_<label>]/ in this project:
- cases.txt: the test cases
- round_<n>.txt: test-backend-ops's raw output for each round
- summary.json: the median time per case over the rounds, and its GFLOP/s
- comparison.md: against the latest Quantized matmul run's q8/avx2_v2 and
  fp32/avx2_v4, as time ratios (above 1: ggml is slower)
- metadata.json: hardware, build, revisions, command and settings, the
  thermal state before and after each round, and the round checks: each
  round's median, over cases, of its time relative to round 1's (a round
  more than 5% away is listed in slow_rounds), and the rounds where macOS
  reported a CPU speed limit or the screen saver ran

Usage, from anywhere in the repository, with LLAMA_CPP_DIR set:
    uv run python projects/profile-llama-cpp/scripts/run_kernel_compare.py --help
"""

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchkit import (
    cpu_model,
    cpu_slug,
    create_run_dir,
    machine_details,
    median_ratios_to_first,
    rounds_beyond,
    screen_saver_problem,
    screen_saver_rounds,
    stay_awake,
    thermal_state,
    throttled_rounds,
    utc_date,
)
from runkit import QUANT_MATMUL, RESULTS, build_details, llama_source, revisions
from setup_llama_cpp import executable

SHAPES_HEADER = QUANT_MATMUL / "support" / "gpt2_shapes.hpp"
TOKENS = [1, 128, 1024]
LARGEST_LM_HEAD_TOKENS = 128
ROUND_TOLERANCE = 0.05
# ggml's weight types compared, and the Quantized matmul kernel each is compared with.
WEIGHT_TYPES = {"q8_0": "q8/avx2_v2", "f32": "fp32/avx2_v4"}

ANSI = re.compile(r"\x1b\[[0-9;]*m")
RESULT = re.compile(
    r"MUL_MAT\(.*sources=(?P<type>\w+)\[(?P<k>\d+),(?P<n>\d+),1,1\],f32\[\d+,(?P<m>\d+),1,1\]\)"
    r".*?(?P<runs>\d+) runs -\s+(?P<us>[\d.]+) us/run"
)


@dataclass(frozen=True)
class Case:
    weight_type: str
    shape: str
    n: int
    k: int
    m: int

    @property
    def key(self) -> str:
        return f"{self.weight_type}/{self.shape}/tokens:{self.m}"

    @property
    def flops(self) -> int:
        return 2 * self.m * self.n * self.k


def gpt2_shapes(header: str) -> list[tuple[str, int, int]]:
    """(name, n, k) of each shape in Quantized matmul's gpt2_shapes.hpp."""
    return [(s, int(n), int(k)) for s, n, k in re.findall(r'\{"(\w+)",\s*(\d+),\s*(\d+)\}', header)]


def enum_values(header: str, enum: str) -> dict[str, int]:
    """The members of a C enum and their values, from a header's text."""
    match = re.search(rf"enum\s+{enum}\s*\{{(.*?)\}};", header, re.S)
    if not match:
        raise ValueError(f"enum {enum} not found")
    body = re.sub(r"//[^\n]*|/\*.*?\*/", "", match.group(1), flags=re.S)
    values: dict[str, int] = {}
    value = -1
    for item in body.split(","):
        name, _, expression = item.strip().partition("=")
        name = name.strip()
        if not name:
            continue
        value = int(expression, 0) if expression.strip() else value + 1
        values[name] = value
    return values


def cases(shapes: list[tuple[str, int, int]]) -> list[Case]:
    return [
        Case(weight, shape, n, k, m)
        for weight in WEIGHT_TYPES
        for shape, n, k in shapes
        for m in TOKENS
        if not (shape == "lm_head" and m > LARGEST_LM_HEAD_TOKENS)
    ]


def case_line(case: Case, op: int, types: dict[str, int]) -> str:
    """One test case in test-backend-ops's --test-file format.

    The fields: op, output type, output dimensions, op parameter count and
    values, source count, then per source its type, dimensions and strides
    (0: contiguous), and a name ("-": none). ggml's weight is k x n
    (ne = [k, n]), its activations k x m, and the output n x m.
    """
    weight = types[f"GGML_TYPE_{case.weight_type.upper()}"]
    f32 = types["GGML_TYPE_F32"]
    fields = [op, f32, case.n, case.m, 1, 1, 0, 2]
    fields += [weight, case.k, case.n, 1, 1, 0, 0, 0, 0]
    fields += [f32, case.k, case.m, 1, 1, 0, 0, 0, 0]
    return " ".join(map(str, fields)) + " -"


def parse_output(text: str) -> dict[str, float]:
    """Microseconds per run of each MUL_MAT case, keyed like Case.key (by shape dimensions)."""
    times = {}
    for line in ANSI.sub("", text).splitlines():
        match = RESULT.search(line)
        if match:
            dims = f"{match['type']}/{match['n']}x{match['k']}/tokens:{match['m']}"
            times[dims] = float(match["us"])
    return times


def dims_key(case: Case) -> str:
    return f"{case.weight_type}/{case.n}x{case.k}/tokens:{case.m}"


def latest_quant_matmul_run(results: Path) -> Path | None:
    """The latest benchmark run of Quantized matmul (not an accuracy run)."""
    runs = sorted(
        p for p in results.glob("*") if (p / "summary.json").exists() and "_accuracy" not in p.name
    )
    return runs[-1] if runs else None


def compare(
    all_cases: list[Case], medians: dict[str, float], qmm: dict[str, Any]
) -> list[dict[str, Any]]:
    out = []
    for case in all_cases:
        us = medians.get(dims_key(case))
        if us is None:
            continue
        reference = qmm.get(f"{WEIGHT_TYPES[case.weight_type]}/{case.shape}/tokens:{case.m}")
        qmm_us = reference["time_ns"] / 1e3 if reference else None
        out.append(
            {
                "case": case.key,
                "ggml_us": us,
                "ggml_gflops": case.flops / us / 1e3,
                "qmm_kernel": WEIGHT_TYPES[case.weight_type],
                "qmm_us": qmm_us,
                "time_ratio": us / qmm_us if qmm_us else None,
            }
        )
    return out


def markdown(rows: list[dict[str, Any]], qmm_run: str) -> str:
    lines = [
        "# ggml's matrix multiply against Quantized matmul's kernels",
        "",
        f"Quantized matmul run: `{qmm_run}`. Time ratio above 1: ggml is slower.",
        "",
        "| Case | ggml (µs) | ggml GFLOP/s | Kernel | Kernel (µs) | Time ratio |",
        "|---|---:|---:|---|---:|---:|",
    ]
    for r in rows:
        qmm = f"{r['qmm_us']:.1f}" if r["qmm_us"] else "—"
        ratio = f"{r['time_ratio']:.2f}" if r["time_ratio"] else "—"
        lines.append(
            f"| {r['case']} | {r['ggml_us']:.1f} | {r['ggml_gflops']:.1f} | {r['qmm_kernel']} "
            f"| {qmm} | {ratio} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Time ggml's MUL_MAT on GPT-2's shapes.")
    parser.add_argument("--build", default="build-cpu", help="llama.cpp build folder")
    parser.add_argument("--threads", type=int, default=1, help="threads ggml uses (default 1)")
    parser.add_argument("--rounds", type=int, default=3, help="separate runs (default 3)")
    parser.add_argument("--cooldown", type=float, default=30.0, help="seconds between rounds")
    parser.add_argument("--label", default="", help="suffix for the results folder")
    args = parser.parse_args()
    if problem := screen_saver_problem():
        print(f"run_kernel_compare: {problem}", file=sys.stderr)
        return 2

    try:
        source = llama_source(os.environ)
    except ValueError as e:
        print(f"run_kernel_compare: {e}", file=sys.stderr)
        return 2
    build_dir = source / args.build
    exe = executable(build_dir, "test-backend-ops", sys.platform)
    if not exe.exists():
        print(f"run_kernel_compare: {exe} is missing; run setup_llama_cpp.py", file=sys.stderr)
        return 2
    header = (source / "ggml" / "include" / "ggml.h").read_text()
    enums = enum_values(header, "ggml_op") | enum_values(header, "ggml_type")
    all_cases = cases(gpt2_shapes(SHAPES_HEADER.read_text()))

    suffix = "_kernels" + (f"_{args.label}" if args.label else "")
    run_dir = create_run_dir(RESULTS, f"{utc_date()}_{cpu_slug(cpu_model())}{suffix}")
    print(f"run_kernel_compare: writing to {run_dir}", file=sys.stderr)
    case_file = run_dir / "cases.txt"
    lines = [case_line(c, enums["GGML_OP_MUL_MAT"], enums) for c in all_cases]
    case_file.write_text("\n".join(lines) + "\n")
    command = [str(exe), "perf", "-b", "CPU", "--test-file", str(case_file)]
    env = {**os.environ, "TEST_ARG_HARDWARE_CONCURRENCY": str(2 * args.threads)}

    times: dict[str, list[float]] = {}
    round_times: list[dict[str, float]] = []
    rounds: list[dict[str, Any]] = []
    with stay_awake():
        for n in range(1, args.rounds + 1):
            if n > 1:
                time.sleep(args.cooldown)
            print(f"run_kernel_compare: round {n}/{args.rounds}", file=sys.stderr)
            record: dict[str, Any] = {"round": n, "thermal_before": thermal_state()}
            result = subprocess.run(command, check=True, capture_output=True, text=True, env=env)
            record["thermal_after"] = thermal_state()
            rounds.append(record)
            (run_dir / f"round_{n}.txt").write_text(result.stdout)
            parsed = parse_output(result.stdout)
            round_times.append(parsed)
            for key, us in parsed.items():
                times.setdefault(key, []).append(us)
    ratios = median_ratios_to_first(round_times)
    for record, ratio in zip(rounds, ratios, strict=True):
        record["time_ratio_to_round_1"] = ratio
    missing = [c.key for c in all_cases if dims_key(c) not in times]
    if missing:
        print(f"run_kernel_compare: no result for {missing}", file=sys.stderr)

    medians = {key: statistics.median(values) for key, values in times.items()}
    qmm_run = latest_quant_matmul_run(QUANT_MATMUL / "results")
    qmm = json.loads((qmm_run / "summary.json").read_text()) if qmm_run else {}
    rows = compare(all_cases, medians, qmm)
    summary = {"rows": rows, "round_times_us": times}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (run_dir / "comparison.md").write_text(markdown(rows, qmm_run.name if qmm_run else "none"))
    metadata = {
        "machine": machine_details(),
        "build": build_details(build_dir),
        "revisions": revisions(source),
        "command": command,
        "environment": {"TEST_ARG_HARDWARE_CONCURRENCY": env["TEST_ARG_HARDWARE_CONCURRENCY"]},
        "quant_matmul_run": str(qmm_run) if qmm_run else None,
        "settings": vars(args),
        "missing_cases": missing,
        "rounds": rounds,
        "round_tolerance": ROUND_TOLERANCE,
        "slow_rounds": rounds_beyond(ratios, ROUND_TOLERANCE),
        "throttled_rounds": throttled_rounds(rounds),
        "screen_saver_rounds": screen_saver_rounds(rounds),
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"run_kernel_compare: done; comparison in {run_dir / 'comparison.md'}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
