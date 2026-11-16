from pathlib import Path
from typing import Any

from run_baseline import (
    bench_key,
    ceilings_command,
    llama_bench_command,
    round_ratios,
    slow_rounds,
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


def test_llama_bench_command_keeps_everything_on_the_cpu() -> None:
    command = llama_bench_command(Path("llama-bench"), Path("m.gguf"), 512, 128, "1,6,12", 5)
    assert command[0] == "llama-bench"
    assert command[command.index("-ngl") + 1] == "0"
    assert command[command.index("-t") + 1] == "1,6,12"
    assert command[command.index("-o") + 1] == "json"


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
    assert slow_rounds(ratios, 0.05) == [3]
