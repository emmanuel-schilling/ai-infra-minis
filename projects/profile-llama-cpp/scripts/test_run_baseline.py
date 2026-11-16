from pathlib import Path
from typing import Any

import pytest
from benchkit import rounds_beyond
from run_baseline import (
    BenchCase,
    bench_key,
    ceilings_command,
    llama_bench_command,
    parse_threads,
    plan_tests,
    round_order,
    round_ratios,
    summarize,
)


def bench_test(prompt: int, gen: int, threads: int, samples: list[float]) -> dict[str, Any]:
    return {
        "n_prompt": prompt,
        "n_gen": gen,
        "n_threads": threads,
        "samples_ts": samples,
        "avg_ts": sum(samples) / len(samples),
    }


def test_parse_threads_takes_lists_and_ranges() -> None:
    assert parse_threads("1,6,12") == [1, 6, 12]
    assert parse_threads("1-4,6,8-8,6") == [1, 2, 3, 4, 6, 8]
    for bad in ("0", "4-2", "x"):
        with pytest.raises(ValueError):
            parse_threads(bad)


def test_plan_tests_covers_every_model_phase_and_thread_count() -> None:
    tests = plan_tests(["q8_0", "q4_k_m"], [1, 6])
    assert len(tests) == 8
    assert BenchCase("q4_k_m", "tg", 6) in tests
    assert BenchCase("q8_0", "pp", 1).stem == "q8_0_pp_threads1"


def test_round_order_is_a_seeded_shuffle_per_round() -> None:
    tests = plan_tests(["q8_0", "q4_k_m"], [1, 6, 12])
    first = round_order(tests, seed=1, round_number=1)
    assert sorted(first, key=str) == sorted(tests, key=str)  # the same tests
    assert round_order(tests, seed=1, round_number=1) == first  # repeatable
    orders = {tuple(round_order(tests, seed=1, round_number=n)) for n in range(1, 4)}
    assert len(orders) == 3  # each round differs
    assert round_order(tests, seed=2, round_number=1) != first


def test_llama_bench_command_runs_one_phase_on_the_cpu() -> None:
    exe, model = Path("llama-bench"), Path("m.gguf")
    pp = llama_bench_command(exe, model, BenchCase("q8_0", "pp", 6), 512, 128, 5)
    tg = llama_bench_command(exe, model, BenchCase("q8_0", "tg", 1), 512, 128, 5)
    assert pp[pp.index("-p") + 1 : pp.index("-p") + 4] == ["512", "-n", "0"]
    assert tg[tg.index("-p") + 1 : tg.index("-p") + 4] == ["0", "-n", "128"]
    assert pp[pp.index("-t") + 1] == "6"
    assert pp[pp.index("-ngl") + 1] == "0"
    assert pp[pp.index("-o") + 1] == "json"


def test_ceilings_command_runs_only_the_roofline() -> None:
    command = ceilings_command(Path("qmm_bench"), Path("out.json"), 5)
    assert "--benchmark_filter=^roofline/" in command
    assert "--benchmark_out=out.json" in command


def test_bench_key_names_the_phase() -> None:
    assert bench_key("q8_0", bench_test(512, 0, 6, [1.0])) == "q8_0/pp512/threads:6"
    assert bench_key("q8_0", bench_test(0, 128, 1, [1.0])) == "q8_0/tg128/threads:1"
    assert bench_key("q8_0", bench_test(512, 128, 6, [1.0])) == "q8_0/pp512+tg128/threads:6"


def test_summarize_takes_the_median_over_every_round() -> None:
    rounds = [
        {"q8_0": [bench_test(512, 0, 6, [100.0, 102.0]), bench_test(0, 128, 6, [20.0])]},
        {"q8_0": [bench_test(512, 0, 6, [98.0]), bench_test(0, 128, 6, [22.0])]},
    ]
    assert summarize(rounds) == {
        "q8_0/pp512/threads:6": {"tokens_per_second": 100.0, "samples": 3},
        "q8_0/tg128/threads:6": {"tokens_per_second": 21.0, "samples": 2},
    }


def test_round_ratios_and_slow_rounds() -> None:
    rounds = [
        {"m": [bench_test(512, 0, 6, [100.0]), bench_test(0, 128, 6, [20.0])]},
        {"m": [bench_test(512, 0, 6, [101.0]), bench_test(0, 128, 6, [20.2])]},
        {"m": [bench_test(512, 0, 6, [90.0]), bench_test(0, 128, 6, [18.0])]},
    ]
    ratios = round_ratios(rounds)
    assert [round(r, 3) for r in ratios] == [1.0, 1.01, 0.9]
    assert rounds_beyond(ratios, 0.05) == [3]
