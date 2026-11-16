"""Profile llama.cpp's prompt processing and token generation under xctrace.

For each phase, records llama-bench under an Instruments template, with
nothing offloaded to a GPU:
- pp: a prompt of --prompt tokens and no generation
- tg: --gen generated tokens and no prompt

With the Time Profiler template (the default), it then exports the samples
and writes, per phase, where ggml's graph computation spent its CPU time: by
ggml operation, by leaf function (self time, including spinning at the
barrier between operations), by function anywhere on the stack, and by
thread. Other templates, such as CPU Counters, are only recorded, to open in
Instruments.

Writes a new folder results/<date>_<cpu>_profile[_<label>]/ in this project:
- <model>_<phase>.trace: the recording, to open in Instruments (git-ignored:
  traces run to hundreds of MB)
- <model>_<phase>.log: llama-bench's output during the recording
- <model>_<phase>.json and .md: the summary
- metadata.json: hardware, build, revisions and commands

Usage, from anywhere in the repository, with LLAMA_CPP_DIR and MODELS_DIR set:
    uv run python projects/profile-llama-cpp/scripts/run_profile.py --help
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import time_profile
from benchkit import cpu_model, cpu_slug, create_run_dir, machine_details, stay_awake, utc_date
from runkit import (
    RESULTS,
    build_details,
    llama_source,
    model_for,
    models_dir,
    revisions,
    split_list,
)
from setup_llama_cpp import executable

TIME_PROFILER = "Time Profiler"
TIME_PROFILE_XPATH = '/trace-toc/run[@number="1"]/data/table[@schema="time-profile"]'


def phase_tokens(phase: str, prompt: int, gen: int) -> tuple[int, int]:
    """llama-bench's (-p, -n) for one phase."""
    if phase == "pp":
        return prompt, 0
    if phase == "tg":
        return 0, gen
    raise ValueError(f"unknown phase {phase!r}; choose pp or tg")


def record_command(
    template: str, trace: Path, bench: Path, model: Path, tokens: tuple[int, int], threads: int,
    repetitions: int,
) -> list[str]:  # fmt: skip
    prompt, gen = tokens
    return [
        "xcrun", "xctrace", "record",
        "--template", template,
        "--output", str(trace),
        "--no-prompt",
        "--launch", "--",
        str(bench),
        "-m", str(model),
        "-p", str(prompt),
        "-n", str(gen),
        "-t", str(threads),
        "-r", str(repetitions),
        "-ngl", "0",
        "-o", "json",
    ]  # fmt: skip


def export_command(trace: Path) -> list[str]:
    return ["xcrun", "xctrace", "export", "--input", str(trace), "--xpath", TIME_PROFILE_XPATH]


def main() -> int:
    parser = argparse.ArgumentParser(description="Profile llama.cpp under xctrace.")
    parser.add_argument("--build", default="build-cpu", help="llama.cpp build folder")
    parser.add_argument("--model", default="q8_0", help="q8_0 or q4_k_m")
    parser.add_argument("--phases", default="pp,tg", help="comma-separated: pp, tg")
    parser.add_argument("--threads", type=int, default=6, help="llama-bench's -t")
    parser.add_argument("--prompt", type=int, default=512, help="prompt tokens")
    parser.add_argument("--gen", type=int, default=128, help="generated tokens")
    parser.add_argument("--repetitions", type=int, default=3, help="llama-bench's -r")
    parser.add_argument("--template", default=TIME_PROFILER, help="Instruments template")
    parser.add_argument("--top", type=int, default=30, help="functions listed per table")
    parser.add_argument("--label", default="", help="suffix for the results folder")
    args = parser.parse_args()

    # PLATFORM: macos — xctrace ships with Xcode; Linux would use perf, in a
    # script of its own (future work: step 9 of the plan in STATUS.md).
    if sys.platform != "darwin":
        print("run_profile: xctrace needs macOS; Linux profiling is future work", file=sys.stderr)
        return 2
    try:
        source = llama_source(os.environ)
        model = models_dir(os.environ) / model_for(args.model).file
        phases = {p: phase_tokens(p, args.prompt, args.gen) for p in split_list(args.phases)}
    except ValueError as e:
        print(f"run_profile: {e}", file=sys.stderr)
        return 2
    build_dir = source / args.build
    bench = executable(build_dir, "llama-bench", sys.platform)
    for path in (bench, model):
        if not path.exists():
            print(f"run_profile: {path} is missing; run setup_llama_cpp.py", file=sys.stderr)
            return 2

    suffix = "_profile" + (f"_{args.label}" if args.label else "")
    run_dir = create_run_dir(RESULTS, f"{utc_date()}_{cpu_slug(cpu_model())}{suffix}")
    print(f"run_profile: writing to {run_dir}", file=sys.stderr)
    commands: dict[str, list[str]] = {}
    with stay_awake():
        for phase, tokens in phases.items():
            stem = f"{args.model}_{phase}"
            trace = run_dir / f"{stem}.trace"
            command = record_command(
                args.template, trace, bench, model, tokens, args.threads, args.repetitions
            )
            commands[phase] = command
            print(f"run_profile: recording {stem}", file=sys.stderr)
            with (run_dir / f"{stem}.log").open("w") as log:
                subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
            if args.template != TIME_PROFILER:
                continue
            with subprocess.Popen(export_command(trace), stdout=subprocess.PIPE) as export:
                assert export.stdout is not None
                profile = time_profile.parse(export.stdout)
            if export.returncode != 0:
                raise subprocess.CalledProcessError(export.returncode, export.args)
            data = time_profile.summary(profile, args.top)
            (run_dir / f"{stem}.json").write_text(json.dumps(data, indent=2) + "\n")
            title = f"{args.model} {phase}, {args.threads} threads"
            (run_dir / f"{stem}.md").write_text(time_profile.markdown(title, data))

    metadata = {
        "machine": machine_details(),
        "build": build_details(build_dir),
        "revisions": revisions(source),
        "model": str(model),
        "commands": commands,
        "settings": vars(args),
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"run_profile: done; summaries in {run_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
