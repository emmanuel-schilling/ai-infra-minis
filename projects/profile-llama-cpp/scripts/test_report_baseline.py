import pytest
from report_baseline import Ceilings, best, markdown, parse_ceilings, rows, scaling


def test_best_takes_the_higher_reading_and_flags_disagreement() -> None:
    before = Ceilings({1: 1.5e11, 6: 6.0e11}, {1: 6.0e9})
    after = Ceilings({1: 1.45e11}, {1: 1.5e10, 6: 2.6e10})
    merged, disagree = best([before, after])
    assert merged.flops == {1: 1.5e11, 6: 6.0e11}
    assert merged.bandwidth == {1: 1.5e10, 6: 2.6e10}
    assert disagree == ["bandwidth on 1 threads: 6e+09, 1.5e+10"]


# Google Benchmark's output with repetitions: one row per repetition, then
# aggregates; only the medians count.
OUTPUT = {
    "benchmarks": [
        {"run_name": "roofline/peak_fma", "threads": 1, "items_per_second": 1e11},
        {
            "run_name": "roofline/peak_fma",
            "aggregate_name": "median",
            "threads": 1,
            "items_per_second": 1.5e11,
        },
        {
            "run_name": "roofline/peak_fma/real_time/threads:6",
            "aggregate_name": "median",
            "threads": 6,
            "items_per_second": 6e11,
        },
        {
            "run_name": "roofline/read_bandwidth/real_time/threads:6",
            "aggregate_name": "median",
            "threads": 6,
            "bytes_per_second": 3e10,
        },
        {
            "run_name": "roofline/read_bandwidth/real_time/threads:6",
            "aggregate_name": "mean",
            "threads": 6,
            "bytes_per_second": 1.0,
        },
    ]
}

SUMMARY = {
    "models": {
        "q8_0": {
            "values": 1_100,
            "bytes": 2_000,
            "embedding_values": 100,
            "embedding_bytes": 500,
        }
    },
    "tests": {
        "q8_0/tg128/threads:6": {"tokens_per_second": 10.0e6},
        "q8_0/pp512/threads:1": {"tokens_per_second": 15.0e6},
        "q8_0/pp512/threads:6": {"tokens_per_second": 60.0e6},
        "q8_0/pp512/threads:3": {"tokens_per_second": 40.0e6},
        "q8_0/pp512+tg128/threads:6": {"tokens_per_second": 1.0},
    },
}


def test_parse_ceilings_keeps_the_medians() -> None:
    ceilings = parse_ceilings(OUTPUT)
    assert ceilings.flops == {1: 1.5e11, 6: 6e11}
    assert ceilings.bandwidth == {6: 3e10}


def test_rows_place_each_phase_on_its_ceiling() -> None:
    table = rows(SUMMARY, parse_ceilings(OUTPUT))
    by_key = {(r.phase, r.threads): r for r in table}
    assert len(table) == 4  # the combined test is left out

    tg = by_key[("tg", 6)]
    assert tg.achieved == pytest.approx(1_500 * 10.0e6)  # bytes without the embedding
    assert tg.same_threads == pytest.approx(0.5)

    pp = by_key[("pp", 6)]
    assert pp.achieved == pytest.approx(2 * 1_000 * 60.0e6)  # 2 FLOPs per non-embedding weight
    assert pp.same_threads == pytest.approx(0.2)
    assert pp.q8_estimate == pytest.approx(0.2 / (4 / 6))

    unmeasured = by_key[("pp", 3)]
    assert unmeasured.same_threads is None
    assert unmeasured.machine == pytest.approx(2 * 1_000 * 40.0e6 / 6e11)


def test_scaling_is_relative_to_the_fewest_threads() -> None:
    table = rows(SUMMARY, parse_ceilings(OUTPUT))
    pp = dict((t, (s, p)) for t, s, p in scaling(table)[("q8_0", "pp")])
    assert pp[1] == (1.0, 1.0)
    assert pp[6][0] == pytest.approx(4.0)
    assert pp[6][1] == pytest.approx(4.0 / 6)


def test_markdown_has_every_section() -> None:
    ceilings = Ceilings({1: 1.5e11}, {1: 1.5e10})
    text = markdown(rows(SUMMARY, parse_ceilings(OUTPUT)), ceilings)
    for heading in ("## Ceilings measured", "## Token generation", "## Prompt processing"):
        assert heading in text
    assert "| q8_0 | pp | 6 | 4.00× | 67% |" in text
