import struct
from pathlib import Path

import pytest
from gguf_tensors import Tensor, read_tensors, weight_summary


def gguf_string(text: str) -> bytes:
    data = text.encode()
    return struct.pack("<Q", len(data)) + data


def write_gguf(path: Path, tensors: list[tuple[str, int, tuple[int, ...]]]) -> None:
    """A GGUF header with one of each kind of key-value pair, then the tensor entries."""
    kvs = [
        gguf_string("general.architecture") + struct.pack("<I", 8) + gguf_string("qwen2"),
        gguf_string("qwen2.block_count") + struct.pack("<II", 4, 28),
        gguf_string("qwen2.rope.freq_base") + struct.pack("<If", 6, 1e6),
        gguf_string("tokenizer.ggml.tokens")
        + struct.pack("<IIQ", 9, 8, 2)
        + gguf_string("a")
        + gguf_string("bc"),
        gguf_string("tokenizer.ggml.scores") + struct.pack("<IIQ", 9, 6, 3) + b"\0" * 12,
    ]
    out = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(kvs)) + b"".join(kvs)
    for name, kind, shape in tensors:
        out += (
            gguf_string(name)
            + struct.pack("<I", len(shape))
            + struct.pack("<" + "Q" * len(shape), *shape)
        )
        out += struct.pack("<IQ", kind, 0)
    path.write_bytes(out)


def test_reads_tensors_after_every_kind_of_value(tmp_path: Path) -> None:
    path = tmp_path / "model.gguf"
    write_gguf(path, [("token_embd.weight", 8, (64, 10)), ("blk.0.ffn_up.weight", 12, (256, 4))])
    tensors = read_tensors(path)
    assert tensors == [
        Tensor("token_embd.weight", 8, (64, 10)),
        Tensor("blk.0.ffn_up.weight", 12, (256, 4)),
    ]


def test_bytes_by_block_size() -> None:
    assert Tensor("a", 0, (3, 2)).bytes == 24  # f32
    assert Tensor("b", 8, (64, 10)).bytes == 20 * 34  # Q8_0: 34 bytes per 32 values
    assert Tensor("c", 12, (256, 4)).bytes == 4 * 144  # Q4_K: 144 bytes per 256
    assert Tensor("d", 14, (256,)).bytes == 210  # Q6_K


def test_bytes_reject_unknown_types_and_partial_blocks() -> None:
    with pytest.raises(ValueError, match="no known block size"):
        _ = Tensor("a", 99, (32,)).bytes
    with pytest.raises(ValueError, match="whole blocks"):
        _ = Tensor("b", 8, (33,)).bytes


def test_weight_summary_separates_the_embedding() -> None:
    tensors = [Tensor("token_embd.weight", 8, (64, 10)), Tensor("output.weight", 8, (64, 10))]
    assert weight_summary(tensors) == {
        "values": 1280,
        "bytes": 1360,
        "embedding_values": 640,
        "embedding_bytes": 680,
    }


def test_rejects_other_files(tmp_path: Path) -> None:
    path = tmp_path / "x.gguf"
    path.write_bytes(b"GGML")
    with pytest.raises(ValueError, match="not a GGUF"):
        read_tensors(path)
