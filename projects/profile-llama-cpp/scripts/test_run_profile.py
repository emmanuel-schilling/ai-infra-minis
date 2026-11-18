from pathlib import Path

import pytest
from run_profile import phase_tokens, phases_in, record_command


def test_phases_in_maps_recording_numbers_to_phases() -> None:
    recordings = [{"round": 1, "phase": "pp"}, {"round": 2, "phase": "tg"}]
    assert phases_in(recordings, [2]) == ["tg"]
    assert phases_in(recordings, []) == []


def test_phase_tokens() -> None:
    assert phase_tokens("pp", 512, 128) == (512, 0)
    assert phase_tokens("tg", 512, 128) == (0, 128)
    with pytest.raises(ValueError, match="pp or tg"):
        phase_tokens("pg", 512, 128)


def test_record_command_launches_llama_bench_on_the_cpu() -> None:
    command = record_command(
        "Time Profiler",
        Path("t.trace"),
        Path("t.bench.json"),
        Path("llama-bench"),
        Path("m.gguf"),
        (0, 128),
        6,
        3,
    )
    assert command[:3] == ["xcrun", "xctrace", "record"]
    # llama-bench's own output is kept, not dropped by xctrace
    assert command[command.index("--target-stdout") + 1] == "t.bench.json"
    assert command.index("--target-stdout") < command.index("--launch")
    launched = command[command.index("--") + 1 :]
    assert launched[0] == "llama-bench"
    assert launched[launched.index("-p") + 1] == "0"
    assert launched[launched.index("-n") + 1] == "128"
    assert launched[launched.index("-ngl") + 1] == "0"
