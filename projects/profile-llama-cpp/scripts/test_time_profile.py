import io

from time_profile import (
    AT_BARRIER,
    IDLE,
    SCHEDULER,
    WORKER_SPIN,
    markdown,
    operation,
    parse,
    short_name,
    summary,
)

# Samples in xctrace's shape, with stacks as real traces show them (the
# compiler inlines ggml_graph_compute_thread, so worker stacks end in the
# worker loop): the second reuses the first's thread, weight and worker-loop
# frame by ref; the third repeats the first backtrace by ref; the fourth is a
# worker checking for its next graph; the fifth, during model loading, is
# outside the graph computation.
EXPORT = """<?xml version="1.0"?>
<trace-query-result><node><schema name="time-profile"/>
<row><sample-time id="1">1</sample-time>
<thread id="2" fmt="worker A"><tid id="3">7</tid></thread>
<weight id="4" fmt="1.00 ms">1000000</weight>
<backtrace id="5">
<frame id="6" name="void tinyBLAS::gemm4xN&lt;2&gt;(long long, long long)" addr="0x1"/>
<frame id="7" name="ggml_compute_forward_mul_mat_one_chunk(ggml_compute_params const*)" addr="0x2"/>
<frame id="8" name="ggml_compute_forward_mul_mat(ggml_compute_params const*, ggml_tensor*)"
 addr="0x3"/>
<frame id="9" name="ggml_graph_compute_secondary_thread(void*)" addr="0x4"/>
</backtrace></row>
<row><sample-time id="10">2</sample-time><thread ref="2"/><weight ref="4"/>
<backtrace id="11"><frame id="12" name="ggml_thread_cpu_relax" addr="0x5"/>
<frame id="13" name="ggml_barrier" addr="0x6"/><frame ref="9"/></backtrace></row>
<row><sample-time id="14">3</sample-time>
<thread id="15" fmt="worker B"><tid id="16">8</tid></thread>
<weight id="17" fmt="2.00 ms">2000000</weight><backtrace ref="5"/></row>
<row><sample-time id="18">4</sample-time><thread ref="15"/><weight ref="4"/>
<backtrace id="19"><frame id="20" name="ggml_graph_compute_thread_ready" addr="0x8"/>
<frame ref="9"/></backtrace></row>
<row><sample-time id="21">5</sample-time><thread ref="2"/><weight ref="4"/>
<backtrace id="22"><frame id="23" name="llama_model_load" addr="0x7"/></backtrace></row>
</node></trace-query-result>
"""


def test_short_name_drops_the_parameter_list() -> None:
    assert short_name("f(int, g(x))") == "f"
    assert short_name("void (anonymous namespace)::k<a>::m(long)") == (
        "void (anonymous namespace)::k<a>::m"
    )
    assert short_name("ggml_barrier") == "ggml_barrier"


def test_operation_is_the_outermost_forward_frame() -> None:
    frames = [
        "leaf",
        "ggml_compute_forward_mul_mat_one_chunk(x)",
        "ggml_compute_forward_mul_mat(x)",
    ]
    assert operation(frames) == "mul_mat"


def test_operation_labels_only_what_the_stack_proves() -> None:
    worker = "ggml_graph_compute_secondary_thread(void*)"
    spin = "ggml_thread_cpu_relax"
    assert operation([spin, "ggml_barrier", worker]) == AT_BARRIER
    # The barrier inlined into the dispatcher, or into the main thread's loop
    assert operation([spin, "ggml_compute_forward(ggml_compute_params*)", worker]) == AT_BARRIER
    assert operation([spin, "ggml_graph_compute(ggml_cgraph*)", "main"]) == AT_BARRIER
    # Inlined into the worker loop, the barrier and the polling look the same
    assert operation([spin, worker]) == WORKER_SPIN
    assert operation(["ggml_graph_compute_thread_ready", worker]) == IDLE
    assert operation(["__psynch_cvwait", "_pthread_cond_wait", worker]) == IDLE
    assert operation(["ggml_graph_plan", "ggml_graph_compute"]) == SCHEDULER


def test_parse_resolves_refs_and_separates_idle_and_loading() -> None:
    profile = parse(io.BytesIO(EXPORT.encode()))
    assert profile.compute_ns == 4_000_000
    assert profile.idle_ns == 1_000_000
    assert profile.other_ns == 1_000_000
    assert profile.self_ns == {
        "void tinyBLAS::gemm4xN<2>": 3_000_000,
        "ggml_thread_cpu_relax": 1_000_000,
    }
    assert profile.op_ns == {"mul_mat": 3_000_000, AT_BARRIER: 1_000_000}
    assert profile.thread_ns == {"worker A": 2_000_000, "worker B": 2_000_000}
    assert profile.total_ns["ggml_graph_compute_secondary_thread"] == 4_000_000
    assert profile.total_ns["ggml_barrier"] == 1_000_000


SPIN_EXPORT = """<?xml version="1.0"?>
<trace-query-result><node><schema name="time-profile"/>
<row><thread id="1" fmt="worker"/><weight id="2">3000000</weight>
<backtrace id="3"><frame id="4" name="ggml_thread_cpu_relax" addr="0x10"/>
<frame id="5" name="ggml_graph_compute_secondary_thread(void*)" addr="0x2efa">
<binary id="6" name="libggml-cpu.dylib" path="/b/libggml-cpu.dylib" load-addr="0x1000"/></frame>
</backtrace></row>
<row><thread ref="1"/><weight id="7">1000000</weight>
<backtrace id="8"><frame ref="4"/>
<frame id="9" name="ggml_graph_compute_secondary_thread(void*)" addr="0x2f40">
<binary ref="6"/></frame></backtrace></row>
</node></trace-query-result>
"""


def test_worker_spin_is_reported_by_source_line() -> None:
    profile = parse(io.BytesIO(SPIN_EXPORT.encode()))
    assert profile.spin_sites == {
        ("/b/libggml-cpu.dylib", "0x1000", "0x2efa"): 3_000_000,
        ("/b/libggml-cpu.dylib", "0x1000", "0x2f40"): 1_000_000,
    }

    def resolve(binary: str, load: str, addrs: list[str]) -> dict[str, str]:
        assert (binary, load) == ("/b/libggml-cpu.dylib", "0x1000")
        return {"0x2efa": "ggml_graph_compute_thread (ggml-cpu.c:3155)"}

    rows = summary(profile, top=5, resolve=resolve)["worker_spin_lines"]
    assert rows == [
        {"name": "ggml_graph_compute_thread (ggml-cpu.c:3155)", "ms": 3.0, "share": 0.75},
        {"name": "/b/libggml-cpu.dylib 0x2f40", "ms": 1.0, "share": 0.25},
    ]
    unresolved = summary(profile, top=5)["worker_spin_lines"]
    assert [r["name"] for r in unresolved] == [
        "/b/libggml-cpu.dylib 0x2efa",
        "/b/libggml-cpu.dylib 0x2f40",
    ]


def test_summary_and_markdown() -> None:
    data = summary(parse(io.BytesIO(EXPORT.encode())), top=1)
    assert data["compute_ms"] == 4.0
    assert data["idle_between_graphs_ms"] == 1.0
    assert data["self"] == [{"name": "void tinyBLAS::gemm4xN<2>", "ms": 3.0, "share": 0.75}]
    assert len(data["operations"]) == 2
    text = markdown("q8_0 pp", data)
    assert text.startswith("# q8_0 pp\n")
    assert "idle between graphs: 1 ms" in text
    assert "| 75.0% | 3 | `mul_mat` |" in text
