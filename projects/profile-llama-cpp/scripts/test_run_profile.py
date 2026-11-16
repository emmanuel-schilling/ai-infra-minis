from pathlib import Path

import pytest
from run_profile import export_command, phase_tokens, record_command


def test_phase_tokens() -> None:
    assert phase_tokens("pp", 512, 128) == (512, 0)
    assert phase_tokens("tg", 512, 128) == (0, 128)
    with pytest.raises(ValueError, match="pp or tg"):
        phase_tokens("pg", 512, 128)


def test_record_command_launches_llama_bench_on_the_cpu() -> None:
    command = record_command(
        "Time Profiler", Path("t.trace"), Path("llama-bench"), Path("m.gguf"), (0, 128), 6, 3
    )
    assert command[:3] == ["xcrun", "xctrace", "record"]
    launched = command[command.index("--") + 1 :]
    assert launched[0] == "llama-bench"
    assert launched[launched.index("-p") + 1] == "0"
    assert launched[launched.index("-n") + 1] == "128"
    assert launched[launched.index("-ngl") + 1] == "0"


def test_export_command_selects_the_time_profile() -> None:
    command = export_command(Path("t.trace"))
    assert command[command.index("--input") + 1] == "t.trace"
    assert 'schema="time-profile"' in command[-1]
