"""Time llama.cpp's prompt processing and token generation, and the machine's ceilings.

Each test is one llama-bench run of one model, one phase and one thread
count, with nothing offloaded to a GPU: a prompt of --prompt tokens (prompt
processing, pp) or --gen generated tokens (token generation, tg). Every round
runs every test once, in a different order each round (a shuffle seeded from
--seed and the round number), after a --cooldown pause before each test. So
every test starts from a similar temperature, and an effect of the order
would show as spread between rounds instead of a constant bias.

Before and after the rounds, it measures the roofline's ceilings with
Quantized matmul's qmm_bench: read bandwidth and peak fp32 FMA throughput on
1, 6 and 12 threads. Measuring them twice catches a reading lowered by
something else running at that moment, which a single measurement would hide.

Writes a new folder results/<date>_<cpu>[_<label>]/ in this project:
- ceilings_before.json, ceilings_after.json: Google Benchmark's raw output
- round_<n>_<model>_<phase>_threads<t>.json: llama-bench's raw output
- summary.json: for each test, the median tokens per second over every
  sample of every round; and each model's weight sizes from its GGUF header,
  for the roofline
- metadata.json: hardware, OS, CPU features, the llama.cpp build's options,
  both repositories' revisions, the commands, each round's test order, the
  thermal state before and after every test, and the checks below

The checks, as in Quantized matmul: each round is compared with round 1
through its own results, the median over all tests of the ratio of their
speeds, and a round more than 5% away is listed in slow_rounds. Also listed:
rounds with a test that started under a macOS CPU speed limit
(started_throttled_rounds, which invalidate a run), rounds where a limit was
reported at all, typically one that set in during a test (throttled_rounds,
recorded and reported), and rounds where the screen saver ran.

Steps of the plan it serves, with their options:
- baseline (the defaults): both models, 1, 6 and 12 threads
- Accelerate: --build build-accelerate --threads 6 --label accelerate (macOS)
- thread scaling: --threads 1,2,3,4,6,8,10,12 --label scaling

Usage, from anywhere in the repository, with LLAMA_CPP_DIR and MODELS_DIR set:
    uv run python projects/profile-llama-cpp/scripts/run_baseline.py --help
"""

import argparse
import json
import os
import random
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
    started_throttled,
    started_throttled_rounds,
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
PHASES = ("pp", "tg")


@dataclass(frozen=True)
class BenchCase:
    model: str
    phase: str  # "pp" or "tg"
    threads: int

    @property
    def stem(self) -> str:
        return f"{self.model}_{self.phase}_threads{self.threads}"


def parse_threads(text: str) -> list[int]:
    """Thread counts from a comma-separated list whose items may be ranges: '1,4-6,12'."""
    counts: list[int] = []
    for item in split_list(text):
        first, _, last = item.partition("-")
        low, high = int(first), int(last or first)
        if low < 1 or high < low:
            raise ValueError(f"bad thread count or range {item!r}")
        counts.extend(range(low, high + 1))
    return sorted(set(counts))


def plan_tests(models: list[str], threads: list[int]) -> list[BenchCase]:
    return [BenchCase(m, p, t) for m in models for p in PHASES for t in threads]


def round_order(tests: list[BenchCase], seed: int, round_number: int) -> list[BenchCase]:
    """The tests in this round's order: a shuffle that depends only on the seed and the round."""
    order = list(tests)
    random.Random(f"{seed}/{round_number}").shuffle(order)
    return order


def llama_bench_command(
    exe: Path, model: Path, test: BenchCase, prompt: int, gen: int, repetitions: int
) -> list[str]:
    tokens = (prompt, 0) if test.phase == "pp" else (0, gen)
    return [
        str(exe),
        "-m", str(model),
        "-p", str(tokens[0]),
        "-n", str(tokens[1]),
        "-t", str(test.threads),
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

    Each round maps a model key to the llama-bench tests it ran for that model.
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
    speeds = [
        {bench_key(model, t): t["avg_ts"] for model, tests in r.items() for t in tests}
        for r in rounds
    ]
    return median_ratios_to_first(speeds)


def main() -> int:
    parser = argparse.ArgumentParser(description="Time llama.cpp and the machine's ceilings.")
    parser.add_argument("--build", default="build-cpu", help="llama.cpp build folder")
    parser.add_argument("--models", default="q8_0,q4_k_m", help="comma-separated: q8_0, q4_k_m")
    parser.add_argument("--threads", default="1,6,12", help="thread counts: a list, ranges allowed")
    parser.add_argument("--prompt", type=int, default=512, help="prompt tokens")
    parser.add_argument("--gen", type=int, default=128, help="generated tokens")
    parser.add_argument("--rounds", type=int, default=3, help="separate runs (default 3)")
    parser.add_argument("--repetitions", type=int, default=5, help="llama-bench's -r")
    parser.add_argument("--cooldown", type=float, default=30.0, help="seconds before each test")
    parser.add_argument("--seed", type=int, default=1, help="seed for each round's test order")
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
        tests = plan_tests(list(models), parse_threads(args.threads))
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
    records: list[dict[str, Any]] = []
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
        for n in range(1, args.rounds + 1):
            by_model: dict[str, list[dict[str, Any]]] = {key: [] for key in models}
            for position, test in enumerate(round_order(tests, args.seed, n), start=1):
                time.sleep(args.cooldown)
                command = llama_bench_command(
                    bench, models[test.model], test, args.prompt, args.gen, args.repetitions
                )
                commands[test.stem] = command
                print(f"run_baseline: round {n}/{args.rounds}, {test.stem}", file=sys.stderr)
                record: dict[str, Any] = {
                    "round": n,
                    "position": position,
                    "test": test.stem,
                    "thermal_before": thermal_state(),
                }
                result = subprocess.run(command, check=True, capture_output=True, text=True)
                record["thermal_after"] = thermal_state()
                (run_dir / f"round_{n}_{test.stem}.json").write_text(result.stdout)
                by_model[test.model].extend(json.loads(result.stdout))
                records.append(record)
            outputs.append(by_model)
        if not args.skip_ceilings:
            time.sleep(args.cooldown)
            after = ceilings_command(QMM_BENCH, run_dir / "ceilings_after.json", 5)
            subprocess.run(after, check=True, stdout=subprocess.DEVNULL)

    ratios = round_ratios(outputs)
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
        "tests": records,
        "round_ratios_to_round_1": ratios,
        "slow_rounds": rounds_beyond(ratios, ROUND_TOLERANCE),
        "started_throttled_rounds": started_throttled_rounds(records),
        "started_throttled_tests": [
            f"round {r['round']}: {r['test']}" for r in records if started_throttled(r)
        ],
        "throttled_rounds": throttled_rounds(records),
        "throttled_tests": [
            f"round {r['round']}: {r['test']}" for r in records if throttled_rounds([r])
        ],
        "screen_saver_rounds": screen_saver_rounds(records),
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"run_baseline: done; round ratios {[round(r, 4) for r in ratios]}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
