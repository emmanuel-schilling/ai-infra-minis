"""Place each benchmark of a run on the measured roofline.

Reads a run's summary.json (written by run_bench.py) and, for every kernel
benchmark, computes:

- arithmetic intensity: FLOPs per byte of A, W and C, each read or written
  once (the bytes qmm_bench counts; a Q8 W is 36 bytes per 32 weights)
- the roofline's ceiling there: min(peak FLOP/s, intensity x read bandwidth),
  from the run's own roofline benchmarks
- the share of that ceiling the kernel reaches

GEMM's W stays in cache between iterations (see RESULTS.md), so its traffic
is a lower bound and its intensity an upper one; that doesn't matter at
GEMM's intensities, which are far right of the ridge point.

The Q8 kernels' compute ceiling is lower than the FMA peak: each 32 int8
products take 6 operations on the two FMA ports where fp32 takes 4 FMAs, so
Q8 is drawn against 4/6 of the peak as well. That ceiling is an estimate from
Intel's instruction tables, not a measurement, and is labeled as one.

Writes roofline.md (every benchmark) and roofline.svg (the headline kernels)
into the run folder, and prints a per-kernel summary for RESULTS.md.

Usage, from anywhere in the repository:
    uv run python projects/quant-matmul/scripts/roofline.py results/<run>
"""

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PEAK = "roofline/peak_fma"
BANDWIDTH = "roofline/read_bandwidth"
Q8_CEILING_FACTOR = 4 / 6

NAME = re.compile(r"^(?P<kernel>[^/]+/[^/]+)/(?P<shape>[^/]+)/tokens:(?P<tokens>\d+)$")

# The kernels on the chart, in categorical palette order (slots 1 to 3), and
# the scalar references in muted gray. Colors from the dataviz reference
# palette, validated all-pairs on the light surface.
HEADLINE = {
    "fp32/avx2_v4": "#2a78d6",
    "q8/avx2_v2": "#eb6834",
    "fp32/accelerate": "#1baf7a",
}
MUTED = {"fp32/scalar": "#9b9a96", "q8/scalar": "#6b6a66"}
MARKERS = {1: "o", 128: "s", 1024: "^"}
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"


@dataclass(frozen=True)
class Point:
    kernel: str
    shape: str
    tokens: int
    flops: float  # FLOP/s achieved
    intensity: float  # FLOP per byte
    ceiling: float  # FLOP/s the roofline allows at this intensity
    peak: float  # the roofline's compute ceiling

    @property
    def efficiency(self) -> float:
        return self.flops / self.ceiling

    @property
    def memory_bound(self) -> bool:
        return self.ceiling < self.peak


def ceiling(intensity: float, peak: float, bandwidth: float) -> float:
    """The roofline: bandwidth-bound below the ridge point, compute-bound above it."""
    return min(peak, intensity * bandwidth)


def points(summary: dict[str, Any]) -> tuple[list[Point], float, float]:
    """Every kernel benchmark as a point, with the run's peak FLOP/s and bandwidth."""
    peak = summary[PEAK]["FLOP/s"]
    bandwidth = summary[BANDWIDTH]["bytes_per_second"]
    result = []
    for name, stats in sorted(summary.items()):
        match = NAME.match(name)
        if not match or "FLOP/s" not in stats or "bytes_per_second" not in stats:
            continue
        intensity = stats["FLOP/s"] / stats["bytes_per_second"]
        result.append(
            Point(
                kernel=match["kernel"],
                shape=match["shape"],
                tokens=int(match["tokens"]),
                flops=stats["FLOP/s"],
                intensity=intensity,
                ceiling=ceiling(intensity, peak, bandwidth),
                peak=peak,
            )
        )
    return result, peak, bandwidth


def table(pts: list[Point], peak: float, bandwidth: float) -> str:
    lines = [
        f"Peak {peak / 1e9:.1f} GFLOP/s, read bandwidth {bandwidth / 2**30:.2f} GiB/s, "
        f"ridge point {peak / bandwidth:.2f} FLOP/byte.",
        "",
        "| Benchmark | FLOP/byte | GFLOP/s | Ceiling GFLOP/s | Bound | Share of ceiling |",
        "|---|---|---|---|---|---|",
    ]
    for p in pts:
        bound = "memory" if p.memory_bound else "compute"
        lines.append(
            f"| {p.kernel}/{p.shape}/tokens:{p.tokens} | {p.intensity:.2f} | "
            f"{p.flops / 1e9:.1f} | {p.ceiling / 1e9:.1f} | {bound} | {p.efficiency:.0%} |"
        )
    return "\n".join(lines) + "\n"


def summary_by_kernel(pts: list[Point], peak: float) -> str:
    """Per kernel, the range of its share of the ceiling, for GEMV and for GEMM."""
    lines = [
        "| Kernel | GEMV (1 token) | GEMM (128 and 1,024 tokens) | GEMM, of the Q8 estimate |",
        "|---|---|---|---|",
    ]
    for kernel in sorted({p.kernel for p in pts}):
        gemv = [p.efficiency for p in pts if p.kernel == kernel and p.tokens == 1]
        gemm = [p for p in pts if p.kernel == kernel and p.tokens > 1]
        q8 = ""
        if kernel.startswith("q8/") and gemm:
            shares = [p.flops / min(p.ceiling, peak * Q8_CEILING_FACTOR) for p in gemm]
            q8 = span_of(shares)
        lines.append(
            f"| {kernel} | {span_of(gemv)} | {span_of([p.efficiency for p in gemm])} | {q8} |"
        )
    return "\n".join(lines) + "\n"


def span_of(values: list[float]) -> str:
    if not values:
        return ""
    low, high = min(values), max(values)
    return f"{low:.0%}" if round(low, 2) == round(high, 2) else f"{low:.0%}–{high:.0%}"


def chart(pts: list[Point], peak: float, bandwidth: float, cpu: str, out: Path) -> None:
    import matplotlib

    matplotlib.use("svg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 10,
            "text.color": TEXT_PRIMARY,
            "axes.labelcolor": TEXT_SECONDARY,
            "xtick.color": TEXT_SECONDARY,
            "ytick.color": TEXT_SECONDARY,
            "svg.fonttype": "none",
        }
    )
    fig, ax = plt.subplots(figsize=(8, 5.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_xscale("log")
    ax.set_yscale("log")

    shown = [p for p in pts if p.kernel in HEADLINE or p.kernel in MUTED]
    xs = [p.intensity for p in shown]
    x_low, x_high = min(xs) / 5, max(xs) * 2
    ridge = peak / bandwidth

    ax.plot(
        [x_low, ridge, x_high],
        [x_low * bandwidth / 1e9, peak / 1e9, peak / 1e9],
        color=TEXT_PRIMARY,
        linewidth=2,
        zorder=2,
    )
    q8_peak = peak * Q8_CEILING_FACTOR
    q8_ridge = q8_peak / bandwidth
    ax.plot(
        [q8_ridge, x_high],
        [q8_peak / 1e9, q8_peak / 1e9],
        color=TEXT_SECONDARY,
        linewidth=1.5,
        linestyle=(0, (4, 3)),
        zorder=2,
    )
    ax.annotate(
        f"FMA peak {peak / 1e9:.0f} GFLOP/s (measured)",
        (x_high, peak / 1e9),
        xytext=(-4, 5),
        textcoords="offset points",
        ha="right",
        color=TEXT_PRIMARY,
    )
    # Left of where its line starts, above the sloped roof.
    ax.annotate(
        f"Q8 ceiling ≈ {q8_peak / 1e9:.0f} GFLOP/s\n(estimate)",
        (q8_ridge, q8_peak / 1e9),
        xytext=(-8, 0),
        textcoords="offset points",
        ha="right",
        va="center",
        color=TEXT_SECONDARY,
    )
    # Below the sloped roof, where no benchmark sits.
    label_x = min(p.intensity for p in shown) * 1.4
    ax.annotate(
        f"read bandwidth {bandwidth / 2**30:.1f} GiB/s (measured)",
        (label_x, label_x * bandwidth / 1e9 / 3),
        ha="left",
        va="center",
        color=TEXT_PRIMARY,
    )

    colors = HEADLINE | MUTED
    for kernel, color in colors.items():
        for tokens, marker in MARKERS.items():
            group = [p for p in shown if p.kernel == kernel and p.tokens == tokens]
            if not group:
                continue
            ax.scatter(
                [p.intensity for p in group],
                [p.flops / 1e9 for p in group],
                s=42,
                marker=marker,
                color=color,
                edgecolors=SURFACE,
                linewidths=1.5,  # the surface ring that separates overlapping marks
                zorder=3 if kernel in HEADLINE else 2.5,
            )

    ax.set_xlim(x_low, x_high)
    plain = FuncFormatter(lambda value, _: f"{value:g}")
    ax.xaxis.set_major_formatter(plain)
    ax.yaxis.set_major_formatter(plain)
    ax.set_xlabel("Arithmetic intensity (FLOP per byte of A, W and C)")
    ax.set_ylabel("GFLOP/s")
    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)

    kernel_handles = [
        Line2D([], [], marker="o", linestyle="", color=c, markersize=8, label=k)
        for k, c in colors.items()
    ]
    token_handles = [
        Line2D(
            [],
            [],
            marker=m,
            linestyle="",
            color=TEXT_SECONDARY,
            markersize=8,
            label=f"{t:,} token{'s' if t > 1 else ''}",
        )
        for t, m in MARKERS.items()
    ]
    # Both legends in the empty corner above the sloped roof.
    first = ax.legend(handles=kernel_handles, loc="upper left", frameon=False)
    ax.add_artist(first)
    ax.legend(handles=token_handles, loc="upper left", frameon=False, bbox_to_anchor=(0.0, 0.7))
    ax.set_title(
        f"Single-core roofline on GPT-2's weight shapes\n{cpu}",
        loc="left",
        color=TEXT_PRIMARY,
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Place a run's benchmarks on its roofline.")
    parser.add_argument("run", type=Path, help="a results/<run> folder")
    args = parser.parse_args()

    summary = json.loads((args.run / "summary.json").read_text())
    pts, peak, bandwidth = points(summary)
    (args.run / "roofline.md").write_text(table(pts, peak, bandwidth))
    metadata = args.run / "metadata.json"
    cpu = json.loads(metadata.read_text())["machine"]["cpu"] if metadata.exists() else ""
    chart(pts, peak, bandwidth, cpu, args.run / "roofline.svg")
    print(summary_by_kernel(pts, peak), end="")
    print(f"roofline: roofline.md and roofline.svg in {args.run}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
