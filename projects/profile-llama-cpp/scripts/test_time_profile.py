import io

from time_profile import markdown, operation, parse, short_name, summary

# Three samples in xctrace's shape: the second reuses the first's thread,
# weight and two of its frames by ref; the third repeats the first backtrace
# by ref; a fourth, during model loading, is outside the graph computation.
EXPORT = """<?xml version="1.0"?>
<trace-query-result><node><schema name="time-profile"/>
<row><sample-time id="1">1</sample-time>
<thread id="2" fmt="Main Thread"><tid id="3">7</tid></thread>
<weight id="4" fmt="1.00 ms">1000000</weight>
<backtrace id="5">
<frame id="6" name="void tinyBLAS::gemm4xN&lt;2&gt;(long long, long long)" addr="0x1"/>
<frame id="7" name="ggml_compute_forward_mul_mat_one_chunk(ggml_compute_params const*)" addr="0x2"/>
<frame id="8" name="ggml_compute_forward_mul_mat(ggml_compute_params const*, ggml_tensor*)"
 addr="0x3"/>
<frame id="9" name="ggml_graph_compute_thread" addr="0x4"/>
</backtrace></row>
<row><sample-time id="10">2</sample-time><thread ref="2"/><weight ref="4"/>
<backtrace id="11"><frame id="12" name="ggml_thread_cpu_relax" addr="0x5"/>
<frame id="13" name="ggml_barrier" addr="0x6"/><frame ref="9"/></backtrace></row>
<row><sample-time id="14">3</sample-time>
<thread id="15" fmt="worker"><tid id="16">8</tid></thread>
<weight id="17" fmt="2.00 ms">2000000</weight><backtrace ref="5"/></row>
<row><sample-time id="18">4</sample-time><thread ref="2"/><weight ref="4"/>
<backtrace id="19"><frame id="20" name="llama_model_load" addr="0x7"/></backtrace></row>
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
    assert operation(["ggml_barrier", "ggml_graph_compute_thread"]) == "(between operations)"


def test_parse_resolves_refs_and_leaves_out_loading() -> None:
    profile = parse(io.BytesIO(EXPORT.encode()))
    assert profile.compute_ns == 4_000_000
    assert profile.other_ns == 1_000_000
    assert profile.self_ns == {
        "void tinyBLAS::gemm4xN<2>": 3_000_000,
        "ggml_thread_cpu_relax": 1_000_000,
    }
    assert profile.op_ns == {"mul_mat": 3_000_000, "(between operations)": 1_000_000}
    assert profile.thread_ns == {"Main Thread": 2_000_000, "worker": 2_000_000}
    assert profile.total_ns["ggml_graph_compute_thread"] == 4_000_000
    assert profile.total_ns["ggml_barrier"] == 1_000_000


def test_summary_and_markdown() -> None:
    data = summary(parse(io.BytesIO(EXPORT.encode())), top=1)
    assert data["compute_ms"] == 4.0
    assert data["self"] == [{"name": "void tinyBLAS::gemm4xN<2>", "ms": 3.0, "share": 0.75}]
    assert len(data["operations"]) == 2
    text = markdown("q8_0 pp", data)
    assert text.startswith("# q8_0 pp\n")
    assert "| 75.0% | 3 | `mul_mat` |" in text
