"""Summarize a profile run's traces: where the time went, or the CPU counters.

For each recording that run_profile.py made in the given run folder, exports
the trace's data and writes <model>_<phase>.json and .md next to it:
- a Time Profiler recording: CPU time by ggml operation, leaf function,
  function anywhere on the stack and thread (time_profile.py);
- a CPU Counters recording: llama-bench's counts and their ratios to cycles,
  overall and by ggml operation, leaf function and thread
  (counters_profile.py).
Summaries that already exist are left as they are.

run_profile.py summarizes each recording as it makes it; this script does it
for traces recorded before, for instance before a summary existed for their
template.

Usage, from anywhere in the repository:
    uv run python projects/profile-llama-cpp/scripts/summarize_profile.py <run folder>
"""

import argparse
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections.abc import Callable
from pathlib import Path
from typing import IO, Any

import counters_profile
import time_profile

TIME_PROFILE = "time-profile"
BENCH_PROCESS = "llama-bench"


def table_xpath(schema: str) -> str:
    return f'/trace-toc/run[@number="1"]/data/table[@schema="{schema}"]'


def export_command(trace: Path, schema: str | None = None) -> list[str]:
    """xctrace exporting one table of the trace, or its table of contents without a schema."""
    command = ["xcrun", "xctrace", "export", "--input", str(trace)]
    return [*command, "--xpath", table_xpath(schema)] if schema else [*command, "--toc"]


def table_schemas(toc: str) -> set[str]:
    """The schemas of the tables in a trace's table of contents."""
    return {t.get("schema", "") for t in ET.fromstring(toc).iter("table")}


def parse_atos(output: str, addrs: list[str]) -> dict[str, str]:
    """atos -i's output, one group of lines per address, as {address: innermost line}.

    With -i, atos prints an address's inlined frames innermost first, one per
    line, and a blank line after each address. Only the innermost is kept, as
    'function (file:line)'; addresses atos couldn't resolve to a line are left out.
    """
    groups = [g.strip().splitlines() for g in output.strip().split("\n\n")]
    resolved = {}
    for addr, group in zip(addrs, groups, strict=False):
        if group and "(in " in group[0] and ":" in group[0].rsplit("(", 1)[-1]:
            function = group[0].split(" (in ", 1)[0]
            location = group[0].rsplit("(", 1)[-1].rstrip(")")
            resolved[addr] = f"{function} ({location})"
    return resolved


def atos_resolver(binary: str, load_addr: str, addrs: list[str]) -> dict[str, str]:
    """Each return address's source line, from the binary's debug information."""
    # PLATFORM: macos — atos ships with Xcode's command-line tools; Linux
    # would use addr2line -i on the same addresses.
    command = ["xcrun", "atos", "-o", binary, "-l", load_addr, "-i", *addrs]
    result = subprocess.run(command, capture_output=True, text=True)
    return parse_atos(result.stdout, addrs) if result.returncode == 0 else {}


def exported[T](trace: Path, schema: str, parse: Callable[[IO[bytes]], T]) -> T:
    """One table of the trace, parsed as xctrace exports it."""
    with subprocess.Popen(export_command(trace, schema), stdout=subprocess.PIPE) as export:
        assert export.stdout is not None
        result = parse(export.stdout)
    if export.returncode != 0:
        raise subprocess.CalledProcessError(export.returncode, export.args)
    return result


def summarize(trace: Path, title: str, top: int) -> str | None:
    """Write the trace's summary next to it; the schema summarized, or None if neither is there."""
    # PLATFORM: macos — xctrace ships with Xcode; Linux profiles would come
    # from perf, in a script of its own (future work).
    toc = subprocess.run(export_command(trace), check=True, capture_output=True, text=True).stdout
    schemas = table_schemas(toc)
    data: dict[str, Any]
    if TIME_PROFILE in schemas:
        schema = TIME_PROFILE
        profile = exported(trace, schema, time_profile.parse)
        data = time_profile.summary(profile, top, atos_resolver)
        text = time_profile.markdown(title, data)
    elif counters_profile.SCHEMA in schemas:
        schema = counters_profile.SCHEMA
        events = counters_profile.pmc_events(toc)

        def parse(stream: IO[bytes]) -> counters_profile.CounterProfile:
            return counters_profile.parse(stream, events, BENCH_PROCESS)

        data = counters_profile.summary(exported(trace, schema, parse), top)
        text = counters_profile.markdown(title, data)
    else:
        return None
    trace.with_suffix(".json").write_text(json.dumps(data, indent=2) + "\n")
    trace.with_suffix(".md").write_text(text)
    return schema


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize a profile run's traces.")
    parser.add_argument("run_dir", type=Path, help="a results folder that run_profile.py wrote")
    parser.add_argument("--top", type=int, default=30, help="functions listed per table")
    args = parser.parse_args()
    if sys.platform != "darwin":
        print("summarize_profile: xctrace needs macOS", file=sys.stderr)
        return 2
    metadata = json.loads((args.run_dir / "metadata.json").read_text())
    model, threads = metadata["settings"]["model"], metadata["settings"]["threads"]
    for recording in metadata["recordings"]:
        trace = args.run_dir / f"{model}_{recording['phase']}.trace"
        if trace.with_suffix(".json").exists():
            print(
                f"summarize_profile: {trace.with_suffix('.json')} exists; left as is",
                file=sys.stderr,
            )
            continue
        if not trace.exists():
            print(f"summarize_profile: {trace} is missing", file=sys.stderr)
            return 2
        title = f"{model} {recording['phase']}, {threads} threads"
        schema = summarize(trace, title, args.top)
        print(
            f"summarize_profile: {trace.name}: {schema or 'nothing to summarize'}", file=sys.stderr
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
