from pathlib import Path

from run_accuracy import summary_tables, weights_comments, worst


def row(data: str, source: str, shape: str, kernel: str, rel: float) -> dict[str, object]:
    return {
        "data": data,
        "source": source,
        "shape": shape,
        "kernel": kernel,
        "relative_error": rel,
        "max_abs_error": rel * 10,
    }


ROWS = [
    row("uniform", "random", "attn_out", "fp32/scalar", 1e-7),
    row("uniform", "random", "attn_out", "q8/avx2_v2", 5e-3),
    row("gpt2", "h0", "attn_out", "q8/avx2_v2", 2e-3),
    row("gpt2", "h1", "attn_out", "q8/avx2_v2", 7e-3),
    row("gpt2", "wte", "lm_head", "q8/avx2_v2", 4e-3),
]


def test_worst_takes_the_largest_over_layers() -> None:
    assert worst(ROWS, "gpt2", "attn_out", "q8/avx2_v2", "relative_error") == 7e-3
    assert worst(ROWS, "gpt2", "mlp_up", "q8/avx2_v2", "relative_error") is None


def test_summary_tables_have_one_column_per_kernel_present() -> None:
    text = summary_tables(ROWS)
    assert "| Shape | fp32/scalar | q8/avx2_v2 |" in text
    assert "| attn_out | 1.0e-07 | 5.0e-03 |" in text
    # gpt2 has no fp32/scalar rows, so that cell is empty.
    assert "| attn_out |  | 7.0e-03 |" in text
    assert "| lm_head |  | 4.0e-03 |" in text
    assert "Max |error|, uniform random W:" in text


def test_weights_comments(tmp_path: Path) -> None:
    (tmp_path / "index.txt").write_text("# repo revision abc\nh0 attn_out 2 2 f.f32\n")
    assert weights_comments(tmp_path) == ["repo revision abc"]
    assert weights_comments(tmp_path / "missing") == []
