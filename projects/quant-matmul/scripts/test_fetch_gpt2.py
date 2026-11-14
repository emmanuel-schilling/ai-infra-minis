import hashlib
import json
import struct
from array import array
from pathlib import Path

import pytest
from fetch_gpt2 import (
    LAYERS,
    download,
    gpt2_tensors,
    read_header,
    read_tensor,
    save_matrices,
    transpose,
)


def write_safetensors(path: Path, tensors: dict[str, tuple[list[int], list[float]]]) -> None:
    """A minimal safetensors file of fp32 tensors, as Hugging Face writes them."""
    header: dict[str, object] = {"__metadata__": {"format": "pt"}}
    data = b""
    for name, (shape, values) in tensors.items():
        raw = array("f", values).tobytes()
        header[name] = {
            "dtype": "F32",
            "shape": shape,
            "data_offsets": [len(data), len(data) + len(raw)],
        }
        data += raw
    encoded = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + data)


def test_transpose() -> None:
    # 2 x 3 -> 3 x 2
    assert list(transpose(array("f", [1, 2, 3, 4, 5, 6]), 2, 3)) == [1, 4, 2, 5, 3, 6]


def test_read_tensor(tmp_path: Path) -> None:
    path = tmp_path / "model.safetensors"
    write_safetensors(path, {"a": ([2], [1.0, 2.0]), "b": ([1, 3], [3.0, 4.0, 5.0])})
    header, start = read_header(path)
    assert header["b"]["shape"] == [1, 3]
    assert list(read_tensor(path, header, start, "b")) == [3.0, 4.0, 5.0]


def test_read_tensor_rejects_other_dtypes(tmp_path: Path) -> None:
    path = tmp_path / "model.safetensors"
    write_safetensors(path, {"a": ([2], [1.0, 2.0])})
    header, start = read_header(path)
    header["a"]["dtype"] = "F16"
    with pytest.raises(ValueError, match="F16"):
        read_tensor(path, header, start, "a")


def test_save_matrices_transposes_and_indexes(tmp_path: Path) -> None:
    path = tmp_path / "model.safetensors"
    # Stored k x n = 2 x 3, so saved as n x k = 3 x 2; the embedding is kept.
    write_safetensors(path, {"conv": ([2, 3], [1, 2, 3, 4, 5, 6]), "wte": ([2, 2], [7, 8, 9, 10])})
    tensors = [("conv", "h0", "attn_out", True), ("wte", "wte", "lm_head", False)]
    save_matrices(path, tmp_path, tensors, ["revision abc"])

    index = (tmp_path / "index.txt").read_text().splitlines()
    assert index == [
        "# revision abc",
        "h0 attn_out 3 2 h0_attn_out.f32",
        "wte lm_head 2 2 wte_lm_head.f32",
    ]
    saved = array("f")
    saved.frombytes((tmp_path / "h0_attn_out.f32").read_bytes())
    assert list(saved) == [1, 4, 2, 5, 3, 6]


def test_gpt2_tensors_cover_every_layer_and_the_embedding() -> None:
    tensors = gpt2_tensors()
    assert len(tensors) == 4 * LAYERS + 1
    assert tensors[0] == ("h.0.attn.c_attn.weight", "h0", "attn_qkv", True)
    assert tensors[-1] == ("wte.weight", "wte", "lm_head", False)


def test_download_checks_the_hash(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"weights")
    good = hashlib.sha256(b"weights").hexdigest()
    dest = tmp_path / "out" / "model.safetensors"

    download(source.as_uri(), dest, good)
    assert dest.read_bytes() == b"weights"

    other = tmp_path / "other.bin"
    with pytest.raises(ValueError, match="SHA-256"):
        download(source.as_uri(), other, "0" * 64)
    assert not other.exists()
    assert not other.with_name("other.bin.part").exists()
