from pathlib import Path

from summarize_profile import export_command, parse_atos, table_schemas

TOC = """<?xml version="1.0"?>
<trace-toc><run number="1"><data>
<table schema="kdebug" target="ALL"/>
<table schema="counters-profile" pmc-events="&quot;CORE_ACTIVE_CYCLE&quot;"/>
</data></run></trace-toc>
"""


def test_parse_atos_keeps_each_addresss_innermost_line() -> None:
    output = (
        "ggml_barrier (in libggml-cpu.dylib) (ggml-cpu.c:609)\n"
        "ggml_graph_compute_thread (in libggml-cpu.dylib) (ggml-cpu.c:3165)\n"
        "ggml_graph_compute_secondary_thread (in libggml-cpu.dylib) (ggml-cpu.c:3281)\n"
        "\n"
        "0x2f40 (in libggml-cpu.dylib)\n"
        "\n"
    )
    assert parse_atos(output, ["0x2efa", "0x2f40"]) == {"0x2efa": "ggml_barrier (ggml-cpu.c:609)"}


def test_export_command_selects_one_table_or_the_table_of_contents() -> None:
    command = export_command(Path("t.trace"), "counters-profile")
    assert command[command.index("--input") + 1] == "t.trace"
    assert 'schema="counters-profile"' in command[-1]
    assert export_command(Path("t.trace"))[-1] == "--toc"


def test_table_schemas_lists_the_traces_tables() -> None:
    assert table_schemas(TOC) == {"kdebug", "counters-profile"}
