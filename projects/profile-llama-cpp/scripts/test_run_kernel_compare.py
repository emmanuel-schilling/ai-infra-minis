from pathlib import Path

from run_kernel_compare import (
    SHAPES_HEADER,
    Case,
    case_line,
    cases,
    compare,
    enum_values,
    gpt2_shapes,
    latest_quant_matmul_run,
    parse_output,
)

HEADER = """
    enum ggml_type {
        GGML_TYPE_F32     = 0,
        GGML_TYPE_F16     = 1,
        // GGML_TYPE_Q4_2 = 4, removed
        GGML_TYPE_Q8_0    = 8,
        GGML_TYPE_Q8_1,
        GGML_TYPE_COUNT,
    };
    enum ggml_op {
        GGML_OP_NONE = 0,
        GGML_OP_DUP, /* a comment, with a comma */
        GGML_OP_ADD,
    };
"""


def test_enum_values_follow_c_rules() -> None:
    assert enum_values(HEADER, "ggml_type") == {
        "GGML_TYPE_F32": 0,
        "GGML_TYPE_F16": 1,
        "GGML_TYPE_Q8_0": 8,
        "GGML_TYPE_Q8_1": 9,
        "GGML_TYPE_COUNT": 10,
    }
    assert enum_values(HEADER, "ggml_op")["GGML_OP_ADD"] == 2


def test_gpt2_shapes_come_from_quant_matmuls_header() -> None:
    shapes = gpt2_shapes(SHAPES_HEADER.read_text())
    assert shapes[0] == ("attn_qkv", 2304, 768)
    assert ("lm_head", 50257, 768) in shapes
    assert len(shapes) == 5


def test_cases_match_quant_matmuls_benchmarks() -> None:
    all_cases = cases([("attn_out", 768, 768), ("lm_head", 50257, 768)])
    keys = [c.key for c in all_cases]
    assert "q8_0/attn_out/tokens:1024" in keys
    assert "q8_0/lm_head/tokens:128" in keys
    assert "q8_0/lm_head/tokens:1024" not in keys
    assert len(all_cases) == 2 * (3 + 2)  # two weight types


def test_case_line_layout() -> None:
    types = {"GGML_TYPE_F32": 0, "GGML_TYPE_Q8_0": 8}
    line = case_line(Case("q8_0", "mlp_down", 768, 3072, 128), 28, types)
    assert line == "28 0 768 128 1 1 0 2 8 3072 768 1 1 0 0 0 0 0 3072 128 1 1 0 0 0 0 -"


def test_parse_output_reads_times_through_color_codes() -> None:
    text = (
        "Backend 1/1: CPU\n"
        "  MUL_MAT(type=f32,ne=[768,1,1,1],op_params=[],sources=q8_0[768,768,1,1],"
        "f32[768,1,1,1]):     24576 runs -    46.09 us/run -      618 kB/run - "
        "\x1b[1;34m  12.79 GB/s\x1b[0m\n"
        "  Backend CPU: \x1b[1;32mOK\x1b[0m\n"
    )
    assert parse_output(text) == {"q8_0/768x768/tokens:1": 46.09}


def test_compare_with_quant_matmul() -> None:
    case = Case("q8_0", "attn_out", 768, 768, 1)
    qmm = {"q8/avx2_v2/attn_out/tokens:1": {"time_ns": 23_045.0}}
    (row,) = compare([case], {"q8_0/768x768/tokens:1": 46.09}, qmm)
    assert row["qmm_us"] == 23.045
    assert round(row["time_ratio"], 2) == 2.0
    assert round(row["ggml_gflops"], 1) == round(2 * 768 * 768 / 46.09 / 1e3, 1)
    (missing,) = compare([case], {"q8_0/768x768/tokens:1": 46.09}, {})
    assert missing["time_ratio"] is None


def test_latest_quant_matmul_run_skips_accuracy_runs(tmp_path: Path) -> None:
    for name in ("2026-11-12_cpu", "2026-11-14_cpu", "2026-11-15_cpu_accuracy", "notes"):
        (tmp_path / name).mkdir()
    for name in ("2026-11-12_cpu", "2026-11-14_cpu", "2026-11-15_cpu_accuracy"):
        (tmp_path / name / "summary.json").write_text("{}")
    assert latest_quant_matmul_run(tmp_path) == tmp_path / "2026-11-14_cpu"
    assert latest_quant_matmul_run(tmp_path / "notes") is None
