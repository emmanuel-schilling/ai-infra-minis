"""Aggregate an xctrace CPU Counters export into counts by ggml operation and function.

`xctrace export --xpath ...table[@schema="counters-profile"]` writes one <row>
per sample, from every process on the machine: its time, its thread, its
backtrace (leaf frame first), its weight (the sampling interval, in
nanoseconds) and its counter values, in the order of the events that the
trace's table of contents lists for the table (pmc_events reads them). The
export shares time_profile's id and ref scheme.

Only the given process's samples are counted, and they are labelled as in
time_profile: by ggml operation, or as the barrier, a worker's spin, idle
between graphs or the scheduler. Samples idle between graphs, and those
outside the graph computation, are reported apart.

Each sample's counts cover the interval since the thread's previous sample,
while its backtrace shows only where the thread was at the end of it. A
function or operation shorter than the interval therefore shares its
interval's counts with whatever else ran in it, so only work that runs for
many intervals at a time, such as prompt processing's matrix multiplications
at 1 ms, gets counts of its own; token generation's operations, at well
under 1 ms, are only meaningful as phase totals.
"""

import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass, field
from typing import IO, Any

from time_profile import COMPUTE, IDLE, ExportReader, Frame, operation, short_name

SCHEMA = "counters-profile"
CYCLES = "CORE_ACTIVE_CYCLE"
CACHE_LINE_BYTES = 64
# Events that count one cache line moved, so their counts convert to bandwidth.
CACHE_LINE_EVENTS = ("L2_RQSTS.MISS", "LONGEST_LAT_CACHE.MISS")

Counts = list[int]


def pmc_events(toc: str) -> list[str]:
    """The counters' event names, in the order of each sample's values.

    They come from the counters-profile table's pmc-events attribute in the
    trace's table of contents (`xctrace export --toc`), a list of quoted names.
    """
    for table in ET.fromstring(toc).iter("table"):
        if table.get("schema") == SCHEMA:
            return table.get("pmc-events", "").split('"')[1::2]
    raise ValueError(f"the trace has no {SCHEMA} table")


def add_counts(total: Counts, counts: Counts) -> None:
    for i, count in enumerate(counts):
        total[i] += count


@dataclass
class CounterProfile:
    events: list[str]
    compute: Counts = field(default_factory=list)
    compute_ns: int = 0
    idle_ns: int = 0
    other_ns: int = 0
    first_ns: int | None = None  # the first and last sample times in the graph computation
    last_ns: int | None = None
    self_counts: defaultdict[str, Counts] = field(default_factory=lambda: defaultdict(list))
    op_counts: defaultdict[str, Counts] = field(default_factory=lambda: defaultdict(list))
    thread_counts: defaultdict[str, Counts] = field(default_factory=lambda: defaultdict(list))

    def __post_init__(self) -> None:
        self.compute = self._zeros()
        for table in (self.self_counts, self.op_counts, self.thread_counts):
            table.default_factory = self._zeros

    def _zeros(self) -> Counts:
        return [0] * len(self.events)

    def add(
        self, time_ns: int, thread: str, weight: int, frames: list[Frame], counts: Counts
    ) -> None:
        names = [short_name(f.name) for f in frames]
        if not any(name.startswith(COMPUTE) for name in names):
            self.other_ns += weight
            return
        op = operation(names)
        if op == IDLE:
            self.idle_ns += weight
            return
        self.compute_ns += weight
        self.first_ns = time_ns if self.first_ns is None else min(self.first_ns, time_ns)
        self.last_ns = time_ns if self.last_ns is None else max(self.last_ns, time_ns)
        add_counts(self.compute, counts)
        add_counts(self.self_counts[names[0]], counts)
        add_counts(self.op_counts[op], counts)
        add_counts(self.thread_counts[thread], counts)


def in_process(thread: str, process: str) -> bool:
    """Whether a thread's label, like 'Main Thread 0x1 (llama-bench, pid: 7)', is the process's."""
    return f"({process}, pid: " in thread


def parse(stream: IO[bytes] | IO[str], events: list[str], process: str) -> CounterProfile:
    """Aggregate a counters-profile export's samples of one process, read incrementally."""
    profile = CounterProfile(events)
    reader = ExportReader()
    for _, row in ET.iterparse(stream, events=("end",)):
        if row.tag != "row":
            continue
        # Read every column that's present, even of rows left out, for later refs.
        elements = [row.find(tag) for tag in ("sample-time", "thread", "weight", "backtrace")]
        time_el, thread_el, weight_el, trace_el = elements
        counts_el = row.find("pmc-events")
        time_ns = reader.integer(time_el) if time_el is not None else 0
        thread = reader.label(thread_el) if thread_el is not None else ""
        weight = reader.integer(weight_el) if weight_el is not None else 0
        frames = reader.frames(trace_el) if trace_el is not None else []
        counts = reader.value(counts_el, sample_counts) if counts_el is not None else None
        if counts is not None and in_process(thread, process):
            if len(counts) != len(events):
                raise ValueError(f"a sample has {len(counts)} counts for {len(events)} events")
            if frames:
                profile.add(time_ns, thread, weight, frames, counts)
            else:
                profile.other_ns += weight
        row.clear()
    return profile


def sample_counts(element: ET.Element) -> Counts:
    return [int(value) for value in (element.text or "").split()]


def ratios(events: list[str], counts: Counts) -> dict[str, float]:
    """Each event's count per cycle."""
    cycles = counts[events.index(CYCLES)]
    return {
        e: (c / cycles if cycles else 0.0)
        for e, c in zip(events, counts, strict=True)
        if e != CYCLES
    }


def rows(events: list[str], table: dict[str, Counts], whole: int, top: int) -> list[dict[str, Any]]:
    """The entries with the most cycles, with their share of all cycles and their ratios."""
    cycle = events.index(CYCLES)
    ranked = sorted(table.items(), key=lambda item: -item[1][cycle])[:top]
    return [
        {
            "name": name,
            "cycles": counts[cycle],
            "share": counts[cycle] / whole if whole else 0.0,
            "per_cycle": ratios(events, counts),
        }
        for name, counts in ranked
    ]


def summary(profile: CounterProfile, top: int) -> dict[str, Any]:
    events = profile.events
    if CYCLES not in events:
        raise ValueError(f"the counters need {CYCLES} for their ratios; the trace has {events}")
    cycles = profile.compute[events.index(CYCLES)]
    first, last = profile.first_ns, profile.last_ns
    wall_ns = last - first if first is not None and last is not None else 0
    totals = dict(zip(events, profile.compute, strict=True))
    bandwidth = {
        e: totals[e] * CACHE_LINE_BYTES / wall_ns  # bytes per ns is GB/s
        for e in CACHE_LINE_EVENTS
        if e in totals and wall_ns
    }
    return {
        "events": events,
        "compute_thread_ms": profile.compute_ns / 1e6,
        "compute_wall_ms": wall_ns / 1e6,
        "idle_between_graphs_ms": profile.idle_ns / 1e6,
        "outside_compute_ms": profile.other_ns / 1e6,
        "ghz": cycles / profile.compute_ns if profile.compute_ns else 0.0,
        "totals": totals,
        "per_cycle": ratios(events, profile.compute),
        "gb_per_second": bandwidth,
        "operations": rows(events, profile.op_counts, cycles, len(profile.op_counts)),
        "self": rows(events, profile.self_counts, cycles, top),
        "threads": rows(events, profile.thread_counts, cycles, len(profile.thread_counts)),
    }


def markdown(title: str, data: dict[str, Any]) -> str:
    others = [e for e in data["events"] if e != CYCLES]
    lines = [
        f"# {title}",
        "",
        f"Samples in ggml's graphs: {data['compute_thread_ms']:.0f} ms of thread time over "
        f"{data['compute_wall_ms']:.0f} ms, at {data['ghz']:.2f} GHz on average; their cycles "
        f"are the whole for the shares below. Worker threads idle between graphs: "
        f"{data['idle_between_graphs_ms']:.0f} ms. Outside the graph computation, left out: "
        f"{data['outside_compute_ms']:.0f} ms.",
        "",
        "Each sample's counts cover the interval since its thread's previous sample, so a "
        "function or operation shorter than that shares its interval's counts with whatever "
        "else ran in it; see counters_profile.py.",
        "",
        "## Totals",
        "",
        "| Event | Count | Per cycle | GB/s (×64 B) |",
        "|---|---:|---:|---:|",
    ]
    for event, count in data["totals"].items():
        per_cycle = f"{data['per_cycle'][event]:.3f}" if event in data["per_cycle"] else ""
        bandwidth = data["gb_per_second"].get(event)
        gbs = f"{bandwidth:.1f}" if bandwidth is not None else ""
        lines.append(f"| `{event}` | {count:,} | {per_cycle} | {gbs} |")
    headings = {
        "operations": "By ggml operation",
        "self": "By leaf function",
        "threads": "By thread",
    }
    header = " | ".join(f"`{e}` per cycle" for e in others)
    for key, heading in headings.items():
        lines += ["", f"## {heading}", "", f"| Share of cycles | {header} | Name |"]
        lines.append("|---:|" + "---:|" * len(others) + "---|")
        for row in data[key]:
            name = row["name"].replace("|", "\\|")
            values = " | ".join(f"{row['per_cycle'][e]:.3f}" for e in others)
            lines.append(f"| {row['share']:.1%} | {values} | `{name}` |")
    return "\n".join(lines) + "\n"
