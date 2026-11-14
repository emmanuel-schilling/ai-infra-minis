"""Measure every kernel's accuracy and keep the results.

Builds the release preset, runs qmm_accuracy on uniform random weights and,
when scripts/fetch_gpt2.py has saved them, on GPT-2's own weights, and writes
a new folder results/<date>_<cpu>_accuracy/ in this project:

- accuracy.json: qmm_accuracy's output, one row per kernel and matrix
- metadata.json: hardware, OS, compiler, flags, the source revision, the
  command, and the weights' revision
- summary.md: tables for RESULTS.md

The errors don't depend on timing, so one pass is enough. They can depend on
the CPU and the BLAS library, which choose their own summation order.

Usage, from anywhere in the repository:
    uv run python projects/quant-matmul/scripts/run_accuracy.py [--tokens N]
"""

import argparse
import datetime
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from fetch_gpt2 import DEFAULT_DIR as WEIGHTS_DIR
from run_bench import (
    BUILD,
    REPO,
    RESULTS,
    build_details,
    cpu_slug,
    create_run_dir,
    machine_details,
    source_revision,
)

# The kernels in the tables; matmul_avx2 and matmul_q8_avx2 only call steps
# already listed, so their rows would repeat them.
TABLE_KERNELS = [
    "fp32/scalar",
    "fp32/avx2_v1",
    "fp32/avx2_v2",
    "fp32/avx2_v3",
    "fp32/avx2_v4",
    "fp32/accelerate",
    "fp32/openblas",
    "q8/scalar",
    "q8/avx2_v1",
    "q8/avx2_v2",
    "q8/neon",
]
SHAPES = ["attn_qkv", "attn_out", "mlp_up", "mlp_down", "lm_head"]
DATA_TITLES = {
    "uniform": "uniform random W",
    "gpt2": "GPT-2's W, worst of its 12 layers (lm_head: the embedding)",
}


def worst(rows: list[dict[str, Any]], data: str, shape: str, kernel: str, key: str) -> float | None:
    """The largest `key` over every matrix of one data set and shape, for one kernel."""
    values = [
        r[key] for r in rows if r["data"] == data and r["shape"] == shape and r["kernel"] == kernel
    ]
    return max(values) if values else None


def summary_tables(rows: list[dict[str, Any]]) -> str:
    """One table per data set and error, kernels as columns and shapes as rows."""
    kernels = [k for k in TABLE_KERNELS if any(r["kernel"] == k for r in rows)]
    out: list[str] = []
    for data, title in DATA_TITLES.items():
        if not any(r["data"] == data for r in rows):
            continue
        for key, label in (("relative_error", "relative error"), ("max_abs_error", "max |error|")):
            out += [f"{label.capitalize()}, {title}:", ""]
            out.append("| Shape | " + " | ".join(kernels) + " |")
            out.append("|---" * (len(kernels) + 1) + "|")
            for shape in SHAPES:
                cells = [worst(rows, data, shape, k, key) for k in kernels]
                if all(c is None for c in cells):
                    continue
                text = ["" if c is None else f"{c:.1e}" for c in cells]
                out.append(f"| {shape} | " + " | ".join(text) + " |")
            out.append("")
    return "\n".join(out)


def weights_comments(weights: Path) -> list[str]:
    index = weights / "index.txt"
    if not index.exists():
        return []
    return [line[2:] for line in index.read_text().splitlines() if line.startswith("# ")]


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure the Quantized matmul kernels' accuracy.")
    parser.add_argument("--tokens", type=int, default=128, help="rows of A (default 128)")
    parser.add_argument("--weights", type=Path, default=WEIGHTS_DIR, help="fetch_gpt2.py's folder")
    args = parser.parse_args()

    if "VCPKG_ROOT" not in os.environ:
        print("run_accuracy: set VCPKG_ROOT to your vcpkg checkout", file=sys.stderr)
        return 2
    subprocess.run(["cmake", "--preset", "release"], cwd=REPO, check=True)
    subprocess.run(
        ["cmake", "--build", "--preset", "release", "--target", "qmm_accuracy"],
        cwd=REPO,
        check=True,
    )
    exe_name = "qmm_accuracy.exe" if os.name == "nt" else "qmm_accuracy"
    exe = BUILD / "projects" / "quant-matmul" / exe_name

    machine = machine_details()
    source = source_revision(REPO, RESULTS)
    started = datetime.datetime.now(datetime.UTC)
    run_dir = create_run_dir(
        RESULTS, started.date().isoformat(), cpu_slug(machine["cpu"]) + "_accuracy"
    )
    command = [str(exe), "--tokens", str(args.tokens), "--out", str(run_dir / "accuracy.json")]
    has_weights = (args.weights / "index.txt").exists()
    if has_weights:
        command += ["--weights", str(args.weights)]
    else:
        print(f"run_accuracy: no GPT-2 weights in {args.weights}; uniform only", file=sys.stderr)
    subprocess.run(command, check=True)

    metadata = {
        "started_utc": started.isoformat(),
        "finished_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "script_command": sys.argv,
        "command": command,
        "machine": machine,
        "build": build_details(),
        "source": source,
        "weights": weights_comments(args.weights) if has_weights else None,
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    rows = json.loads((run_dir / "accuracy.json").read_text())
    (run_dir / "summary.md").write_text(summary_tables(rows))
    print(f"run_accuracy: results in {run_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
