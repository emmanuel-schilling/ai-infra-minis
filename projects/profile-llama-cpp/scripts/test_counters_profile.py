import io

import pytest
from counters_profile import in_process, markdown, parse, pmc_events, summary
from time_profile import AT_BARRIER

EVENTS = ["UOPS_DISPATCHED_PORT.PORT_0", "LONGEST_LAT_CACHE.MISS", "CORE_ACTIVE_CYCLE"]

TOC = """<?xml version="1.0"?>
<trace-toc><run number="1"><data>
<table schema="kdebug-counters-with-time-sample" pmc-events="&quot;X&quot;"/>
<table trigger="time" schema="counters-profile" sample-rate-micro-seconds="1000"
 pmc-events="&quot;UOPS_DISPATCHED_PORT.PORT_0&quot; &quot;LONGEST_LAT_CACHE.MISS&quot;
 &quot;CORE_ACTIVE_CYCLE&quot;"/>
</data></run></trace-toc>
"""

# Samples in xctrace's shape, from every process, each thread's process
# nested in it as real exports write it:
# 1. a kernel idle thread, whose counts (id 5) a llama-bench row reuses;
# 2. worker A in mul_mat, with counts by ref to the kernel row's;
# 3. worker A spinning at a barrier;
# 4. worker B, repeating row 2's backtrace by ref;
# 5. worker B idle between graphs;
# 6. the main thread loading the model, outside the graph computation;
# 7. llama-bench without a backtrace, also outside;
# 8. a blocked llama-bench thread, without counts, left out.
EXPORT = """<?xml version="1.0"?>
<trace-query-result><node><schema name="counters-profile"/>
<row><sample-time id="1">500000</sample-time>
<thread id="2" fmt="idle #2 0xb7 (kernel, pid: 0)"><tid id="3">183</tid>
<process id="4" fmt="kernel (0)"><pid id="40">0</pid></process></thread>
<process ref="4"/><sentinel/>
<weight id="6" fmt="1.00 ms">1000000</weight><pmc-events id="5">10 1 100</pmc-events></row>
<row><sample-time id="7">1000000</sample-time>
<thread id="8" fmt="worker A 0x1 (llama-bench, pid: 7)"><tid id="9">1</tid>
<process id="10" fmt="llama-bench (7)"><pid id="41">7</pid></process></thread>
<process ref="10"/>
<backtrace id="11">
<frame id="12" name="void tinyBLAS::gemm4xN&lt;2&gt;(long long, long long)" addr="0x1"/>
<frame id="13" name="ggml_compute_forward_mul_mat(ggml_compute_params const*, ggml_tensor*)"
 addr="0x2"/>
<frame id="14" name="ggml_graph_compute_secondary_thread(void*)" addr="0x3"/>
</backtrace><weight ref="6"/><pmc-events ref="5"/></row>
<row><sample-time id="15">2000000</sample-time><thread ref="8"/><process ref="10"/>
<backtrace id="16"><frame id="17" name="ggml_thread_cpu_relax" addr="0x4"/>
<frame id="18" name="ggml_barrier" addr="0x5"/><frame ref="14"/></backtrace>
<weight ref="6"/><pmc-events id="19">2 0 100</pmc-events></row>
<row><sample-time id="20">3000000</sample-time>
<thread id="21" fmt="worker B 0x2 (llama-bench, pid: 7)"><tid id="22">2</tid>
<process ref="10"/></thread><process ref="10"/>
<backtrace ref="11"/><weight ref="6"/><pmc-events id="23">30 3 100</pmc-events></row>
<row><sample-time id="24">4000000</sample-time><thread ref="21"/><process ref="10"/>
<backtrace id="25"><frame id="26" name="ggml_graph_compute_thread_ready" addr="0x6"/>
<frame ref="14"/></backtrace><weight ref="6"/><pmc-events id="27">1 0 50</pmc-events></row>
<row><sample-time id="28">5000000</sample-time>
<thread id="29" fmt="Main Thread 0x3 (llama-bench, pid: 7)"><tid id="30">3</tid>
<process ref="10"/></thread><process ref="10"/>
<backtrace id="31"><frame id="32" name="llama_model_load" addr="0x7"/></backtrace>
<weight ref="6"/><pmc-events id="33">5 5 5</pmc-events></row>
<row><sample-time id="34">6000000</sample-time><thread ref="29"/><process ref="10"/>
<sentinel/><weight ref="6"/><pmc-events id="35">7 7 7</pmc-events></row>
<row><sample-time id="36">7000000</sample-time><thread ref="29"/><process ref="10"/>
<sentinel/><backtrace ref="31"/><weight ref="6"/><sentinel/></row>
</node></trace-query-result>
"""


def test_pmc_events_reads_the_counter_tables_events_in_order() -> None:
    assert pmc_events(TOC) == EVENTS
    with pytest.raises(ValueError, match="no counters-profile table"):
        pmc_events("<trace-toc/>")


def test_in_process_matches_the_process_in_a_threads_label() -> None:
    assert in_process("Main Thread 0x1 (llama-bench, pid: 7)", "llama-bench")
    assert not in_process("idle #2 0xb7 (kernel, pid: 0)", "llama-bench")
    assert not in_process("t (llama-bench-2, pid: 8)", "llama-bench")


def test_parse_counts_the_processs_samples_in_the_graph_computation() -> None:
    profile = parse(io.StringIO(EXPORT), EVENTS, "llama-bench")
    assert profile.compute == [42, 4, 300]
    assert profile.compute_ns == 3_000_000
    assert profile.idle_ns == 1_000_000
    assert profile.other_ns == 2_000_000
    assert (profile.first_ns, profile.last_ns) == (1_000_000, 3_000_000)
    assert profile.op_counts["mul_mat"] == [40, 4, 200]
    assert profile.op_counts[AT_BARRIER] == [2, 0, 100]
    assert profile.self_counts["ggml_thread_cpu_relax"] == [2, 0, 100]
    assert profile.thread_counts["worker B 0x2 (llama-bench, pid: 7)"] == [30, 3, 100]


def test_parse_rejects_samples_whose_counts_dont_match_the_events() -> None:
    with pytest.raises(ValueError, match="3 counts for 2 events"):
        parse(io.StringIO(EXPORT), EVENTS[1:], "llama-bench")


def test_summary_gives_ratios_to_cycles_and_bandwidth() -> None:
    data = summary(parse(io.StringIO(EXPORT), EVENTS, "llama-bench"), top=1)
    assert data["compute_thread_ms"] == 3.0
    assert data["compute_wall_ms"] == 2.0
    assert data["ghz"] == pytest.approx(300 / 3_000_000)
    assert data["per_cycle"] == pytest.approx(
        {"UOPS_DISPATCHED_PORT.PORT_0": 0.14, "LONGEST_LAT_CACHE.MISS": 4 / 300}
    )
    # 4 cache lines of 64 bytes over 2 ms of wall time
    assert data["gb_per_second"] == pytest.approx({"LONGEST_LAT_CACHE.MISS": 4 * 64 / 2e6})
    mul_mat, barrier = data["operations"]
    assert (mul_mat["name"], mul_mat["share"]) == ("mul_mat", pytest.approx(2 / 3))
    assert mul_mat["per_cycle"]["UOPS_DISPATCHED_PORT.PORT_0"] == pytest.approx(0.2)
    assert barrier["name"] == AT_BARRIER
    assert len(data["self"]) == 1  # top=1


def test_summary_needs_the_cycle_counter() -> None:
    with pytest.raises(ValueError, match="CORE_ACTIVE_CYCLE"):
        summary(parse(io.StringIO(EXPORT), ["A", "B", "C"], "llama-bench"), top=5)


def test_markdown_lists_totals_and_ratios() -> None:
    data = summary(parse(io.StringIO(EXPORT), EVENTS, "llama-bench"), top=5)
    text = markdown("q8_0 pp, 6 threads", data)
    assert text.startswith("# q8_0 pp, 6 threads\n")
    assert "| `UOPS_DISPATCHED_PORT.PORT_0` | 42 | 0.140 |  |" in text
    assert "| 66.7% | 0.200 | 0.020 | `mul_mat` |" in text
