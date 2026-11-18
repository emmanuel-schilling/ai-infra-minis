"""Aggregate an xctrace Time Profiler export into where the time went.

`xctrace export --xpath ...table[@schema="time-profile"]` writes one <row>
per sample: its thread, its weight (the sampled time, in nanoseconds) and its
backtrace, leaf frame first. Any element that repeats is written in full once
with an id attribute, and afterwards as an empty element whose ref attribute
names that id.

The compiler inlines most of ggml's thread-pool functions, so a sample's
stack shows only part of what the thread was doing. Each sample whose
backtrace runs through ggml's graph computation is labelled by what its stack
proves:
- an operation: a ggml_compute_forward_<op> frame;
- at a barrier: spinning (ggml_thread_cpu_relax) under ggml_barrier, under
  the main thread's ggml_graph_compute, or under ggml_compute_forward, all
  places where only a barrier spins;
- spinning in a worker's loop: spinning directly under the worker loop
  (ggml_graph_compute_secondary_thread). The barrier between operations and
  the polling for the next graph are both inlined there, so the stack can't
  tell them apart; the return address within the loop can, given the
  binary's debug information (for instance with atos -i);
- idle between graphs: a worker checking for or sleeping until its next
  graph (ggml_graph_compute_thread_ready, or a condition-variable wait);
- the scheduler: anything else in the graph computation.
The idle time is reported apart; the shares are of everything else. Model
loading and startup, outside the graph computation, are left out. Time is
attributed three ways:
- self time, to the leaf function: where the CPU actually was, including
  spinning in ggml_barrier;
- total time, to every function on the stack, counted once per sample;
- to the ggml operation, the outermost ggml_compute_forward_<op> frame.
"""

import re
import xml.etree.ElementTree as ET
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import IO, Any, NamedTuple

COMPUTE = "ggml_graph_compute"
MAIN_COMPUTE = "ggml_graph_compute"
WORKER_LOOP = "ggml_graph_compute_secondary_thread"
DISPATCH = "ggml_compute_forward"
BARRIER = "ggml_barrier"
SPIN = "ggml_thread_cpu_relax"
WAITING_FOR_WORK = {"ggml_graph_compute_thread_ready", "__psynch_cvwait", "_pthread_cond_wait"}
OP = re.compile(r"ggml_compute_forward_([a-z0-9_]+)")
# An operation on an extra buffer type, such as weights repacked at load time,
# which ggml's dispatcher hands to that buffer type instead of to its own
# ggml_compute_forward_<op>: the buffer type's forward_<op> method.
EXTRA_DISPATCH = "ggml_cpu_extra_compute_forward"
EXTRA_OP = re.compile(r"::forward_([a-z0-9_]+)$")
AT_BARRIER = "(barrier)"
WORKER_SPIN = "(spinning in a worker's loop: barrier or next graph)"
SCHEDULER = "(scheduler)"
IDLE = "(idle between graphs)"


class Frame(NamedTuple):
    name: str
    addr: str
    binary: str  # the binary's path, or "" when the export names none
    load_addr: str


# Where a worker's spin returns to: (binary, load address, return address).
SpinSite = tuple[str, str, str]


@dataclass
class Profile:
    compute_ns: int = 0
    idle_ns: int = 0
    other_ns: int = 0
    self_ns: Counter[str] = field(default_factory=Counter)
    total_ns: Counter[str] = field(default_factory=Counter)
    op_ns: Counter[str] = field(default_factory=Counter)
    thread_ns: Counter[str] = field(default_factory=Counter)
    spin_sites: Counter[SpinSite] = field(default_factory=Counter)

    def add(self, thread: str, weight: int, frames: list[Frame]) -> None:
        names = [short_name(f.name) for f in frames]
        if not any(name.startswith(COMPUTE) for name in names):
            self.other_ns += weight
            return
        op = operation(names)
        if op == IDLE:
            self.idle_ns += weight
            return
        self.compute_ns += weight
        self.thread_ns[thread] += weight
        self.self_ns[names[0]] += weight
        # Each function once, in stack order, so tied totals keep the same order every run.
        for name in dict.fromkeys(names):
            self.total_ns[name] += weight
        self.op_ns[op] += weight
        if op == WORKER_SPIN:
            caller = frames[1]
            self.spin_sites[(caller.binary, caller.load_addr, caller.addr)] += weight


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
    mul_mat). An operation on an extra buffer type is named by the type's
    forward_<op> method under the extra-buffer dispatcher, marked "(repacked)"
    for weights repacked at load time or "(extra buffer type)" for other
    types. Only that method proves it: before every operation the dispatcher
    asks each extra buffer type, the repack one included, whether it takes
    it. Outside any operation, the labels the module's docstring lists.
    """
    names = [short_name(f) for f in frames]
    present = set(names)
    if EXTRA_DISPATCH in present:
        for name in reversed(names[: names.index(EXTRA_DISPATCH)]):
            match = EXTRA_OP.search(name)
            if match:
                kind = "repacked" if "::repack::" in name else "extra buffer type"
                return f"{match.group(1)} ({kind})"
    for name in reversed(frames):
        match = OP.search(name)
        if match:
            return match.group(1)
    if present & WAITING_FOR_WORK:
        return IDLE
    if BARRIER in present:
        return AT_BARRIER
    if names and names[0] == SPIN:
        # The spinning thread's caller: the first frame that isn't the spin itself.
        caller = names[1] if len(names) > 1 else ""
        if caller in (DISPATCH, MAIN_COMPUTE):
            return AT_BARRIER
        if caller == WORKER_LOOP:
            return WORKER_SPIN
    return SCHEDULER


class ExportReader:
    """Reads the values of an export's rows, resolving id and ref.

    Ids are shared by the whole export, so every row's values must be read,
    including rows that are then left out: a later row may refer to them.
    """

    def __init__(self) -> None:
        self.known: dict[str, Any] = {}  # id -> the value it stands for

    def value(self, element: ET.Element, extract: Callable[[ET.Element], Any]) -> Any:
        ref = element.get("ref")
        if ref is not None:
            return self.known[ref]
        result = extract(element)
        if element.get("id") is not None:
            self.known[element.get("id", "")] = result
        return result

    def label(self, element: ET.Element) -> str:
        """The element's formatted value, such as a thread's name."""
        return self.value(element, lambda e: e.get("fmt", ""))

    def integer(self, element: ET.Element) -> int:
        return self.value(element, lambda e: int(e.text or 0))

    def frames(self, element: ET.Element) -> list[Frame]:
        """A backtrace's frames, leaf first."""
        return self.value(element, self._backtrace)

    def _backtrace(self, element: ET.Element) -> list[Frame]:
        return [self.value(f, self._frame) for f in element.findall("frame")]

    def _frame(self, element: ET.Element) -> Frame:
        found = element.find("binary")
        path, load = self.value(found, self._binary) if found is not None else ("", "")
        addr = element.get("addr", "")
        return Frame(element.get("name") or addr or "?", addr, path, load)

    @staticmethod
    def _binary(element: ET.Element) -> tuple[str, str]:
        return element.get("path", ""), element.get("load-addr", "")


def parse(stream: IO[bytes] | IO[str]) -> Profile:
    """Aggregate a time-profile export, read incrementally."""
    profile = Profile()
    reader = ExportReader()
    for _, row in ET.iterparse(stream, events=("end",)):
        if row.tag != "row":
            continue
        thread_el, weight_el, trace_el = (
            row.find("thread"),
            row.find("weight"),
            row.find("backtrace"),
        )
        if thread_el is not None and weight_el is not None and trace_el is not None:
            thread = reader.label(thread_el)
            weight = reader.integer(weight_el)
            frames = reader.frames(trace_el)
            if frames:
                profile.add(thread, weight, frames)
        row.clear()
    return profile


def table(counts: Counter[str], whole: int, top: int) -> list[tuple[str, float, float]]:
    """The top entries as (name, milliseconds, share of whole)."""
    return [(name, ns / 1e6, ns / whole) for name, ns in counts.most_common(top)] if whole else []


# Maps one binary's return addresses to the source line each is in, given the
# binary's path, its load address and the addresses.
Resolver = Callable[[str, str, list[str]], dict[str, str]]


def spin_lines(profile: Profile, resolve: Resolver | None) -> Counter[str]:
    """The worker-loop spin time by the source line it returns to, where resolvable.

    Sites that can't be resolved are listed by binary and address instead.
    """
    by_binary: dict[tuple[str, str], list[str]] = {}
    for path, load, addr in profile.spin_sites:
        by_binary.setdefault((path, load), []).append(addr)
    lines: dict[SpinSite, str] = {}
    for (path, load), addrs in by_binary.items():
        resolved = resolve(path, load, addrs) if resolve and path and load else {}
        for addr in addrs:
            lines[(path, load, addr)] = resolved.get(addr) or f"{path or '?'} {addr}"
    out: Counter[str] = Counter()
    for site, ns in profile.spin_sites.items():
        out[lines[site]] += ns
    return out


def summary(profile: Profile, top: int, resolve: Resolver | None = None) -> dict[str, Any]:
    whole = profile.compute_ns
    spins = spin_lines(profile, resolve)
    rows = {
        "self": table(profile.self_ns, whole, top),
        "total": table(profile.total_ns, whole, top),
        "operations": table(profile.op_ns, whole, len(profile.op_ns)),
        "threads": table(profile.thread_ns, whole, len(profile.thread_ns)),
        "worker_spin_lines": table(spins, whole, len(spins)),
    }
    return {
        "compute_ms": whole / 1e6,
        "idle_between_graphs_ms": profile.idle_ns / 1e6,
        "outside_compute_ms": profile.other_ns / 1e6,
        **{k: [{"name": n, "ms": ms, "share": s} for n, ms, s in v] for k, v in rows.items()},
    }


def markdown(title: str, data: dict[str, Any]) -> str:
    lines = [
        f"# {title}",
        "",
        f"CPU time in ggml's graphs: {data['compute_ms']:.0f} ms, the whole for the shares "
        f"below. Worker threads idle between graphs: {data['idle_between_graphs_ms']:.0f} ms. "
        f"Outside the graph computation, left out: {data['outside_compute_ms']:.0f} ms.",
    ]
    headings = {
        "operations": "By ggml operation",
        "self": "Self time (leaf function)",
        "total": "Total time (function anywhere on the stack)",
        "threads": "By thread",
        "worker_spin_lines": "Spinning in a worker's loop, by the source line it returns to",
    }
    for key, heading in headings.items():
        lines += ["", f"## {heading}", "", "| Share | ms | Name |", "|---:|---:|---|"]
        for row in data[key]:
            name = row["name"].replace("|", "\\|")
            lines.append(f"| {row['share']:.1%} | {row['ms']:.0f} | `{name}` |")
    return "\n".join(lines) + "\n"
