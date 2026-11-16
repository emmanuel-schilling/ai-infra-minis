"""Aggregate an xctrace Time Profiler export into where the time went.

`xctrace export --xpath ...table[@schema="time-profile"]` writes one <row>
per sample: its thread, its weight (the sampled time, in nanoseconds) and its
backtrace, leaf frame first. Any element that repeats is written in full once
with an id attribute, and afterwards as an empty element whose ref attribute
names that id.

Only samples whose backtrace runs through ggml's graph computation count
toward the shares, so model loading and startup are left out. For each, the
time is attributed three ways:
- self time, to the leaf function: where the CPU actually was, including
  spinning in ggml_barrier;
- total time, to every function on the stack, counted once per sample;
- to the ggml operation, the outermost ggml_compute_forward_<op> frame.
"""

import re
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass, field
from typing import IO, Any

COMPUTE = "ggml_graph_compute"
OP = re.compile(r"ggml_compute_forward_([a-z0-9_]+)")


@dataclass
class Profile:
    compute_ns: int = 0
    other_ns: int = 0
    self_ns: Counter[str] = field(default_factory=Counter)
    total_ns: Counter[str] = field(default_factory=Counter)
    op_ns: Counter[str] = field(default_factory=Counter)
    thread_ns: Counter[str] = field(default_factory=Counter)

    def add(self, thread: str, weight: int, frames: list[str]) -> None:
        if not any(name.startswith(COMPUTE) for name in frames):
            self.other_ns += weight
            return
        self.compute_ns += weight
        self.thread_ns[thread] += weight
        self.self_ns[short_name(frames[0])] += weight
        for name in {short_name(f) for f in frames}:
            self.total_ns[name] += weight
        self.op_ns[operation(frames)] += weight


def short_name(name: str) -> str:
    """A function name without its trailing parameter list."""
    if not name.endswith(")"):
        return name
    depth = 0
    for i in range(len(name) - 1, -1, -1):
        if name[i] == ")":
            depth += 1
        elif name[i] == "(":
            depth -= 1
            if depth == 0:
                return name[:i] if i > 0 else name
    return name


def operation(frames: list[str]) -> str:
    """The ggml operation a sample belongs to.

    That is the outermost ggml_compute_forward_<op> frame, the one ggml's
    dispatcher called; inner ones are helpers such as mul_mat_one_chunk.
    Where the compiler inlined the operation's own function, a helper is the
    outermost frame and names the operation instead (mul_mat_one_chunk for
    mul_mat). Samples outside any operation are threads waiting at the barrier
    between operations, or the graph scheduler itself.
    """
    for name in reversed(frames):
        match = OP.search(name)
        if match:
            return match.group(1)
    return "(between operations)"


def parse(stream: IO[bytes] | IO[str]) -> Profile:
    """Aggregate a time-profile export, read incrementally."""
    profile = Profile()
    known: dict[str, Any] = {}  # id -> the value it stands for

    def value(element: ET.Element, extract: Any) -> Any:
        ref = element.get("ref")
        if ref is not None:
            return known[ref]
        result = extract(element)
        if element.get("id") is not None:
            known[element.get("id", "")] = result
        return result

    def frame_name(frame: ET.Element) -> str:
        return frame.get("name") or frame.get("addr") or "?"

    def backtrace(element: ET.Element) -> list[str]:
        return [value(f, frame_name) for f in element.findall("frame")]

    for _, row in ET.iterparse(stream, events=("end",)):
        if row.tag != "row":
            continue
        thread_el, weight_el, trace_el = (
            row.find("thread"),
            row.find("weight"),
            row.find("backtrace"),
        )
        if thread_el is not None and weight_el is not None and trace_el is not None:
            thread = value(thread_el, lambda e: e.get("fmt", ""))
            weight = value(weight_el, lambda e: int(e.text or 0))
            frames = value(trace_el, backtrace)
            if frames:
                profile.add(thread, weight, frames)
        row.clear()
    return profile


def table(counts: Counter[str], whole: int, top: int) -> list[tuple[str, float, float]]:
    """The top entries as (name, milliseconds, share of whole)."""
    return [(name, ns / 1e6, ns / whole) for name, ns in counts.most_common(top)] if whole else []


def summary(profile: Profile, top: int) -> dict[str, Any]:
    whole = profile.compute_ns
    rows = {
        "self": table(profile.self_ns, whole, top),
        "total": table(profile.total_ns, whole, top),
        "operations": table(profile.op_ns, whole, len(profile.op_ns)),
        "threads": table(profile.thread_ns, whole, len(profile.thread_ns)),
    }
    return {
        "compute_ms": whole / 1e6,
        "outside_compute_ms": profile.other_ns / 1e6,
        **{k: [{"name": n, "ms": ms, "share": s} for n, ms, s in v] for k, v in rows.items()},
    }


def markdown(title: str, data: dict[str, Any]) -> str:
    lines = [
        f"# {title}",
        "",
        f"Samples in ggml's graph computation: {data['compute_ms']:.0f} ms of CPU time "
        f"(outside it, left out: {data['outside_compute_ms']:.0f} ms).",
    ]
    headings = {
        "operations": "By ggml operation",
        "self": "Self time (leaf function)",
        "total": "Total time (function anywhere on the stack)",
        "threads": "By thread",
    }
    for key, heading in headings.items():
        lines += ["", f"## {heading}", "", "| Share | ms | Name |", "|---:|---:|---|"]
        for row in data[key]:
            name = row["name"].replace("|", "\\|")
            lines.append(f"| {row['share']:.1%} | {row['ms']:.0f} | `{name}` |")
    return "\n".join(lines) + "\n"
