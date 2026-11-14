from pathlib import Path

import pytest
from roofline import Q8_CEILING_FACTOR, ceiling, chart, points, summary_by_kernel, table

PEAK = 100e9  # FLOP/s
BANDWIDTH = 10e9  # bytes/s, so the ridge point is at 10 FLOP/byte


def summary() -> dict[str, dict[str, float]]:
    return {
        "roofline/peak_fma": {"FLOP/s": PEAK},
        "roofline/read_bandwidth": {"bytes_per_second": BANDWIDTH},
        # 0.5 FLOP/byte: memory-bound, ceiling 5 GFLOP/s, reaches 4.
        "fp32/avx2_v4/attn_out/tokens:1": {"FLOP/s": 4e9, "bytes_per_second": 8e9},
        # 100 FLOP/byte: compute-bound, ceiling 100 GFLOP/s, reaches 50.
        "q8/avx2_v2/attn_out/tokens:128": {"FLOP/s": 50e9, "bytes_per_second": 0.5e9},
        "q8/avx2_v2/attn_out/tokens:1": {"FLOP/s": 2e9, "bytes_per_second": 1e9},
    }


def test_ceiling_is_bandwidth_then_peak() -> None:
    assert ceiling(1.0, PEAK, BANDWIDTH) == pytest.approx(10e9)
    assert ceiling(10.0, PEAK, BANDWIDTH) == pytest.approx(PEAK)
    assert ceiling(1000.0, PEAK, BANDWIDTH) == pytest.approx(PEAK)


def test_points_skip_the_roofline_benchmarks() -> None:
    pts, peak, bandwidth = points(summary())
    assert (peak, bandwidth) == (PEAK, BANDWIDTH)
    assert {(p.kernel, p.tokens) for p in pts} == {
        ("fp32/avx2_v4", 1),
        ("q8/avx2_v2", 128),
        ("q8/avx2_v2", 1),
    }


def test_points_compute_intensity_and_share() -> None:
    pts, _, _ = points(summary())
    gemv = next(p for p in pts if p.kernel == "fp32/avx2_v4")
    assert gemv.intensity == pytest.approx(0.5)
    assert gemv.ceiling == pytest.approx(5e9)
    assert gemv.efficiency == pytest.approx(0.8)
    assert gemv.memory_bound
    gemm = next(p for p in pts if p.tokens == 128)
    assert gemm.efficiency == pytest.approx(0.5)
    assert not gemm.memory_bound


def test_table_lists_every_benchmark() -> None:
    text = table(*points(summary()))
    assert "ridge point 10.00 FLOP/byte" in text
    assert "| fp32/avx2_v4/attn_out/tokens:1 | 0.50 | 4.0 | 5.0 | memory | 80% |" in text


def test_summary_uses_the_q8_estimate_for_q8_gemm() -> None:
    pts, peak, _ = points(summary())
    text = summary_by_kernel(pts, peak)
    q8_share = 50e9 / (PEAK * Q8_CEILING_FACTOR)
    assert f"| q8/avx2_v2 | 10% | 50% | {q8_share:.0%} |" in text
    assert "| fp32/avx2_v4 | 80% |  |  |" in text


def test_chart_writes_an_svg(tmp_path: Path) -> None:
    pts, peak, bandwidth = points(summary())
    out = tmp_path / "roofline.svg"
    chart(pts, peak, bandwidth, "Test CPU", out)
    text = out.read_text()
    assert text.lstrip().startswith("<?xml")
    assert "Test CPU" in text
