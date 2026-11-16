"""Place a baseline run on the roofline, and show how it scales with threads.

Reads a run_baseline.py folder and writes report.md into it:
- token generation, against the read-bandwidth ceiling: each generated token
  reads every weight once, except the token embedding, of which it reads one
  row, so bytes per token = weight bytes - embedding bytes;
- prompt processing, against the fp32 FMA peak: 2 FLOPs per weight per
  token, again without the embedding, which is looked up. For Q8_0 also
  against Quantized matmul's Q8 estimate: 4/6 of the fp32 peak, since AVX2's
  int8 path takes 6 operations on the FMA ports per 32 int8 products where
  fp32 takes 4 FMAs.
- thread scaling: each test's speed relative to its fewest-thread run.

Left out of both models, and so a few percent of optimism in the shares:
attention's FLOPs and its reads of the KV cache, which at these lengths are
small next to the weights', and the activations' bytes.

Each test is compared with the ceiling measured on the same thread count
when there is one, and always with the machine's: the highest measured. The
ceilings are measured before and after the rounds, and each is the higher of
the two: a ceiling is what the machine can reach, and a lower reading means
something else was running. Readings more than 10% apart are flagged.

Usage, from anywhere in the repository:
    uv run python projects/profile-llama-cpp/scripts/report_baseline.py results/<run>
"""

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

Q8_CEILING_FACTOR = 4 / 6
CEILING_TOLERANCE = 0.10
KEY = re.compile(r"^(?P<model>[^/]+)/(?P<phase>pp|tg)(?P<tokens>\d+)/threads:(?P<threads>\d+)$")


@dataclass(frozen=True)
class Ceilings:
    flops: dict[int, float]  # threads -> FLOP/s
    bandwidth: dict[int, float]  # threads -> bytes/s


def parse_ceilings(output: dict[str, Any]) -> Ceilings:
    """The medians of Google Benchmark's repetitions of the roofline benchmarks."""
    flops: dict[int, float] = {}
    bandwidth: dict[int, float] = {}
    for b in output["benchmarks"]:
        if b.get("aggregate_name") != "median":
            continue
        threads = int(b.get("threads", 1))
        if b["run_name"].startswith("roofline/peak_fma"):
            flops[threads] = b["items_per_second"]
        elif b["run_name"].startswith("roofline/read_bandwidth"):
            bandwidth[threads] = b["bytes_per_second"]
    return Ceilings(flops, bandwidth)


def best(readings: list[Ceilings]) -> tuple[Ceilings, list[str]]:
    """The highest reading of each ceiling, and the ones whose readings disagree."""
    merged: dict[str, dict[int, float]] = {"flops": {}, "bandwidth": {}}
    disagree = []
    for kind, merged_kind in merged.items():
        threads = {t for r in readings for t in getattr(r, kind)}
        for t in sorted(threads):
            values = [getattr(r, kind)[t] for r in readings if t in getattr(r, kind)]
            merged_kind[t] = max(values)
            if min(values) < max(values) * (1 - CEILING_TOLERANCE):
                disagree.append(f"{kind} on {t} threads: {', '.join(f'{v:.3g}' for v in values)}")
    return Ceilings(merged["flops"], merged["bandwidth"]), disagree


@dataclass(frozen=True)
class Row:
    model: str
    phase: str  # "pp" or "tg"
    tokens: int
    threads: int
    tokens_per_second: float
    achieved: float  # bytes/s for tg, FLOP/s for pp
    same_threads: float | None  # share of the ceiling measured on as many threads
    machine: float  # share of the highest ceiling measured
    q8_estimate: float | None  # pp on Q8_0: share of the Q8 estimate


def rows(summary: dict[str, Any], ceilings: Ceilings) -> list[Row]:
    out = []
    for key, test in summary["tests"].items():
        match = KEY.match(key)
        if not match:
            continue  # combined pp+tg tests aren't placed
        model, phase = match["model"], match["phase"]
        threads, speed = int(match["threads"]), test["tokens_per_second"]
        weights = summary["models"][model]
        if phase == "tg":
            achieved = (weights["bytes"] - weights["embedding_bytes"]) * speed
            table = ceilings.bandwidth
        else:
            achieved = 2 * (weights["values"] - weights["embedding_values"]) * speed
            table = ceilings.flops
        same = achieved / table[threads] if threads in table else None
        machine = achieved / max(table.values())
        q8 = None
        if phase == "pp" and model == "q8_0":
            q8 = achieved / (max(table.values()) * Q8_CEILING_FACTOR)
        out.append(
            Row(model, phase, int(match["tokens"]), threads, speed, achieved, same, machine, q8)
        )
    return sorted(out, key=lambda r: (r.model, r.phase, r.threads))


def scaling(table: list[Row]) -> dict[tuple[str, str], list[tuple[int, float, float]]]:
    """Per model and phase: (threads, speedup over the fewest threads, speedup per thread)."""
    groups: dict[tuple[str, str], list[Row]] = {}
    for r in table:
        groups.setdefault((r.model, r.phase), []).append(r)
    result = {}
    for key, group in groups.items():
        base = min(group, key=lambda r: r.threads)
        result[key] = [
            (
                r.threads,
                r.tokens_per_second / base.tokens_per_second,
                r.tokens_per_second / base.tokens_per_second / (r.threads / base.threads),
            )
            for r in group
        ]
    return result


def percent(value: float | None) -> str:
    return "—" if value is None else f"{value:.0%}"


def markdown(table: list[Row], ceilings: Ceilings) -> str:
    lines = ["# Baseline on the roofline", "", "## Ceilings measured", ""]
    lines += ["| Threads | Read bandwidth (GB/s) | fp32 FMA peak (GFLOP/s) |", "|---:|---:|---:|"]
    for t in sorted(set(ceilings.flops) | set(ceilings.bandwidth)):
        bw, fl = ceilings.bandwidth.get(t), ceilings.flops.get(t)
        bw_text = f"{bw / 1e9:.1f}" if bw else "—"
        fl_text = f"{fl / 1e9:.0f}" if fl else "—"
        lines.append(f"| {t} | {bw_text} | {fl_text} |")
    lines += [
        "",
        "## Token generation against bandwidth",
        "",
        "| Model | Threads | Tokens/s | Weight reads (GB/s) | Same threads | Machine |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in (r for r in table if r.phase == "tg"):
        lines.append(
            f"| {r.model} | {r.threads} | {r.tokens_per_second:.1f} | {r.achieved / 1e9:.1f} "
            f"| {percent(r.same_threads)} | {percent(r.machine)} |"
        )
    lines += [
        "",
        "## Prompt processing against compute",
        "",
        "| Model | Threads | Tokens/s | GFLOP/s | Same threads | Machine | Q8 estimate |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in (r for r in table if r.phase == "pp"):
        lines.append(
            f"| {r.model} | {r.threads} | {r.tokens_per_second:.1f} | {r.achieved / 1e9:.0f} "
            f"| {percent(r.same_threads)} | {percent(r.machine)} | {percent(r.q8_estimate)} |"
        )
    lines += ["", "## Thread scaling", "", "| Model | Phase | Threads | Speedup | Per thread |"]
    lines.append("|---|---|---:|---:|---:|")
    for (model, phase), entries in sorted(scaling(table).items()):
        for threads, speedup, per_thread in entries:
            lines.append(f"| {model} | {phase} | {threads} | {speedup:.2f}× | {per_thread:.0%} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Place a baseline run on the roofline.")
    parser.add_argument("run", type=Path, help="a run_baseline.py results folder")
    parser.add_argument(
        "--ceilings", type=Path, help="another run's folder, for a run that skipped the ceilings"
    )
    args = parser.parse_args()
    source = args.ceilings or args.run
    files = sorted(source.glob("ceilings_*.json"))
    if not files:
        print(f"report_baseline: no ceilings in {source}; pass --ceilings", file=sys.stderr)
        return 2
    ceilings, disagree = best([parse_ceilings(json.loads(f.read_text())) for f in files])
    summary = json.loads((args.run / "summary.json").read_text())
    report = markdown(rows(summary, ceilings), ceilings)
    report += f"\nCeilings: the higher of {', '.join(f'`{f.name}`' for f in files)}"
    report += f" in `{source}`.\n" if args.ceilings else ".\n"
    if disagree:
        report += "\nReadings more than 10% apart: " + "; ".join(disagree) + ".\n"
    (args.run / "report.md").write_text(report)
    print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
