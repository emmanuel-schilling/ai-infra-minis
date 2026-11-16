"""Time llama.cpp's prompt processing and token generation, and the machine's ceilings.

For each round, with a cooldown between rounds, runs llama-bench once per
model: a prompt of --prompt tokens (prompt processing) and --gen generated
tokens (token generation) as separate tests, at each thread count, with
nothing offloaded to a GPU. Before and after the rounds, it measures the
roofline's ceilings with Quantized matmul's qmm_bench: read bandwidth and
peak fp32 FMA throughput on 1, 6 and 12 threads. Measuring them twice catches
a reading lowered by something else running at that moment, which a single
measurement would hide.

Writes a new folder results/<date>_<cpu>[_<label>]/ in this project:
- ceilings_before.json, ceilings_after.json: Google Benchmark's raw output
- round_<n>_<model>.json: llama-bench's raw output
- summary.json: for each model, test and thread count, the median tokens per
  second over every sample of every round; and each model's weight sizes
  from its GGUF header, for the roofline
- metadata.json: hardware, OS, CPU features, the llama.cpp build's options,
  both repositories' revisions, the commands, the thermal state around each
  round, and how each round compares with round 1

Each round is compared with round 1 through its own results, as in Quantized
matmul: the median, over all tests, of the ratio of their speeds. A round more
than 5% away is listed in slow_rounds.

Steps of the plan it serves, with their options:
- baseline (the defaults): both models, 1, 6 and 12 threads
- Accelerate: --build build-accelerate --threads 6 --label accelerate (macOS)
- thread scaling: --threads 1-12 --label scaling (llama-bench reads ranges)

Usage, from anywhere in the repository, with LLAMA_CPP_DIR and MODELS_DIR set:
    uv run python projects/profile-llama-cpp/scripts/run_baseline.py --help
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from benchkit import (
    cpu_model,
    cpu_slug,
    create_run_dir,
    machine_details,
    screen_saver_problem,
    screen_saver_rounds,
    stay_awake,
    thermal_state,
    throttled_rounds,
    utc_date,
)
from gguf_tensors import read_tensors, weight_summary
from runkit import (
    REPO,
    RESULTS,
    build_details,
    llama_source,
    model_for,
    models_dir,
    revisions,
    split_list,
)
from setup_llama_cpp import executable

QMM_BENCH = REPO / "build-release" / "projects" / "quant-matmul" / "qmm_bench"
ROUND_TOLERANCE = 0.05


def llama_bench_command(
    exe: Path, model: Path, prompt: int, gen: int, threads: str, repetitions: int
) -> list[str]:
    return [
        str(exe),
        "-m", str(model),
        "-p", str(prompt),
        "-n", str(gen),
        "-t", threads,
        "-r", str(repetitions),
        "-ngl", "0",
        "-o", "json",
    ]  # fmt: skip


def ceilings_command(exe: Path, out: Path, repetitions: int) -> list[str]:
    return [
        str(exe),
        "--benchmark_filter=^roofline/",
        f"--benchmark_repetitions={repetitions}",
        f"--benchmark_out={out}",
        "--benchmark_out_format=json",
    ]


def bench_key(model: str, test: dict[str, Any]) -> str:
    """'q8_0/pp512/threads:6' for prompt processing, 'q8_0/tg128/threads:6' for generation."""
    if test["n_prompt"] > 0 and test["n_gen"] == 0:
        phase = f"pp{test['n_prompt']}"
    elif test["n_gen"] > 0 and test["n_prompt"] == 0:
        phase = f"tg{test['n_gen']}"
    else:
        phase = f"pp{test['n_prompt']}+tg{test['n_gen']}"
    return f"{model}/{phase}/threads:{test['n_threads']}"


def summarize(rounds: list[dict[str, list[dict[str, Any]]]]) -> dict[str, dict[str, Any]]:
    """Median tokens per second per test over all rounds' samples.

    Each round maps a model key to llama-bench's list of tests.
    """
    samples: dict[str, list[float]] = {}
    for tests_by_model in rounds:
        for model, tests in tests_by_model.items():
            for test in tests:
                samples.setdefault(bench_key(model, test), []).extend(test["samples_ts"])
    return {
        key: {"tokens_per_second": statistics.median(values), "samples": len(values)}
        for key, values in sorted(samples.items())
    }


def round_ratios(rounds: list[dict[str, list[dict[str, Any]]]]) -> list[float]:
    """Each round's median, over tests, of its speed relative to round 1's."""

    def speeds(tests_by_model: dict[str, list[dict[str, Any]]]) -> dict[str, float]:
        return {
            bench_key(model, t): t["avg_ts"]
            for model, tests in tests_by_model.items()
            for t in tests
        }

    first = speeds(rounds[0])
    ratios = []
    for r in rounds:
        current = speeds(r)
        common = [k for k in current if k in first and first[k] > 0]
        ratios.append(statistics.median(current[k] / first[k] for k in common) if common else 1.0)
    return ratios


def slow_rounds(ratios: list[float], tolerance: float) -> list[int]:
    """1-based numbers of the rounds more than `tolerance` away from round 1."""
    return [i + 1 for i, ratio in enumerate(ratios) if abs(ratio - 1) > tolerance]


def main() -> int:
    parser = argparse.ArgumentParser(description="Time llama.cpp and the machine's ceilings.")
    parser.add_argument("--build", default="build-cpu", help="llama.cpp build folder")
    parser.add_argument("--models", default="q8_0,q4_k_m", help="comma-separated: q8_0, q4_k_m")
    parser.add_argument("--threads", default="1,6,12", help="llama-bench's -t (list or range)")
    parser.add_argument("--prompt", type=int, default=512, help="prompt tokens")
    parser.add_argument("--gen", type=int, default=128, help="generated tokens")
    parser.add_argument("--rounds", type=int, default=3, help="separate runs (default 3)")
    parser.add_argument("--repetitions", type=int, default=5, help="llama-bench's -r")
    parser.add_argument("--cooldown", type=float, default=60.0, help="seconds between rounds")
    parser.add_argument("--label", default="", help="suffix for the results folder")
    parser.add_argument("--skip-ceilings", action="store_true", help="don't run qmm_bench")
    args = parser.parse_args()
    if problem := screen_saver_problem():
        print(f"run_baseline: {problem}", file=sys.stderr)
        return 2

    try:
        source = llama_source(os.environ)
        folder = models_dir(os.environ)
        models = {key: folder / model_for(key).file for key in split_list(args.models)}
    except ValueError as e:
        print(f"run_baseline: {e}", file=sys.stderr)
        return 2
    build_dir = source / args.build
    bench = executable(build_dir, "llama-bench", sys.platform)
    for path in [bench, *models.values()]:
        if not path.exists():
            print(f"run_baseline: {path} is missing; run setup_llama_cpp.py", file=sys.stderr)
            return 2

    name = f"{utc_date()}_{cpu_slug(cpu_model())}" + (f"_{args.label}" if args.label else "")
    run_dir = create_run_dir(RESULTS, name)
    print(f"run_baseline: writing to {run_dir}", file=sys.stderr)
    commands: dict[str, list[str]] = {}
    rounds: list[dict[str, Any]] = []
    outputs: list[dict[str, list[dict[str, Any]]]] = []

    with stay_awake():
        if not args.skip_ceilings:
            subprocess.run(["cmake", "--preset", "release"], cwd=REPO, check=True)
            build = ["cmake", "--build", "--preset", "release", "--target", "qmm_bench"]
            subprocess.run(build, cwd=REPO, check=True)
            time.sleep(args.cooldown)
            command = ceilings_command(QMM_BENCH, run_dir / "ceilings_before.json", 5)
            commands["ceilings"] = command
            subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
            time.sleep(args.cooldown)
        for n in range(1, args.rounds + 1):
            record: dict[str, Any] = {"round": n, "thermal_before": thermal_state()}
            by_model = {}
            for key, path in models.items():
                command = llama_bench_command(
                    bench, path, args.prompt, args.gen, args.threads, args.repetitions
                )
                commands[key] = command
                print(f"run_baseline: round {n}/{args.rounds}, {key}", file=sys.stderr)
                result = subprocess.run(command, check=True, capture_output=True, text=True)
                (run_dir / f"round_{n}_{key}.json").write_text(result.stdout)
                by_model[key] = json.loads(result.stdout)
            record["thermal_after"] = thermal_state()
            rounds.append(record)
            outputs.append(by_model)
            if n < args.rounds or not args.skip_ceilings:
                time.sleep(args.cooldown)
        if not args.skip_ceilings:
            after = ceilings_command(QMM_BENCH, run_dir / "ceilings_after.json", 5)
            subprocess.run(after, check=True, stdout=subprocess.DEVNULL)

    ratios = round_ratios(outputs)
    for record, ratio in zip(rounds, ratios, strict=True):
        record["ratio_to_round_1"] = ratio
    weights = {key: weight_summary(read_tensors(path)) for key, path in models.items()}
    summary = {"tests": summarize(outputs), "models": weights}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    metadata = {
        "machine": machine_details(),
        "build": build_details(build_dir),
        "revisions": revisions(source),
        "models": {key: str(path) for key, path in models.items()},
        "commands": commands,
        "settings": vars(args),
        "rounds": rounds,
        "slow_rounds": slow_rounds(ratios, ROUND_TOLERANCE),
        "throttled_rounds": throttled_rounds(rounds),
        "screen_saver_rounds": screen_saver_rounds(rounds),
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"run_baseline: done; round ratios {[round(r, 4) for r in ratios]}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
