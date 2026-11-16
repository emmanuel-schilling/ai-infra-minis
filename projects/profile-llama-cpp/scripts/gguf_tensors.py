"""The tensors of a GGUF model file: names, types, shapes and sizes in bytes.

Reads only the header, with the standard library. The roofline needs the
bytes of the weights that token generation reads per token: every matrix,
but only one row of the token embedding, which is looked up rather than
multiplied.

GGUF (version 2 and later): the magic "GGUF", a uint32 version, uint64 tensor
and key-value counts, the key-value pairs, then one entry per tensor: its
name, dimension count, dimensions, type and data offset. Everything is
little-endian.
"""

import struct
from dataclasses import dataclass
from math import prod
from pathlib import Path
from typing import BinaryIO

# ggml type number: (bytes per block, values per block), from ggml's type traits.
BLOCK = {
    0: (4, 1),  # F32
    1: (2, 1),  # F16
    2: (18, 32),  # Q4_0
    3: (20, 32),  # Q4_1
    6: (22, 32),  # Q5_0
    7: (24, 32),  # Q5_1
    8: (34, 32),  # Q8_0
    10: (84, 256),  # Q2_K
    11: (110, 256),  # Q3_K
    12: (144, 256),  # Q4_K
    13: (176, 256),  # Q5_K
    14: (210, 256),  # Q6_K
    30: (2, 1),  # BF16
}
TYPE_NAMES = {0: "f32", 1: "f16", 2: "q4_0", 3: "q4_1", 6: "q5_0", 7: "q5_1", 8: "q8_0"}
TYPE_NAMES |= {10: "q2_k", 11: "q3_k", 12: "q4_k", 13: "q5_k", 14: "q6_k", 30: "bf16"}

# Key-value types: fixed-size scalars by struct format, then string (8) and array (9).
SCALAR = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
STRING, ARRAY = 8, 9

TOKEN_EMBEDDING = "token_embd.weight"


@dataclass(frozen=True)
class Tensor:
    name: str
    type: int
    shape: tuple[int, ...]

    @property
    def values(self) -> int:
        return prod(self.shape)

    @property
    def bytes(self) -> int:
        if self.type not in BLOCK:
            raise ValueError(f"{self.name}: ggml type {self.type} has no known block size")
        block_bytes, block_values = BLOCK[self.type]
        if self.values % block_values:
            raise ValueError(f"{self.name}: {self.values} values aren't whole blocks")
        return self.values // block_values * block_bytes


def _read(f: BinaryIO, fmt: str) -> tuple:
    size = struct.calcsize("<" + fmt)
    data = f.read(size)
    if len(data) != size:
        raise ValueError("truncated GGUF header")
    return struct.unpack("<" + fmt, data)


def _string(f: BinaryIO) -> str:
    (length,) = _read(f, "Q")
    return f.read(length).decode("utf-8")


def _skip_value(f: BinaryIO, kind: int) -> None:
    if kind in SCALAR:
        f.read(struct.calcsize("<" + SCALAR[kind]))
    elif kind == STRING:
        _string(f)
    elif kind == ARRAY:
        item_kind, count = _read(f, "IQ")
        if item_kind in SCALAR:
            f.read(struct.calcsize("<" + SCALAR[item_kind]) * count)
        else:
            for _ in range(count):
                _skip_value(f, item_kind)
    else:
        raise ValueError(f"unknown GGUF value type {kind}")


def read_tensors(path: Path) -> list[Tensor]:
    with path.open("rb") as f:
        if f.read(4) != b"GGUF":
            raise ValueError(f"{path}: not a GGUF file")
        (version,) = _read(f, "I")
        if version < 2:
            raise ValueError(f"{path}: GGUF version {version} is not supported")
        n_tensors, n_kv = _read(f, "QQ")
        for _ in range(n_kv):
            _string(f)
            (kind,) = _read(f, "I")
            _skip_value(f, kind)
        tensors = []
        for _ in range(n_tensors):
            name = _string(f)
            (n_dims,) = _read(f, "I")
            shape = _read(f, "Q" * n_dims)
            kind, _offset = _read(f, "IQ")
            tensors.append(Tensor(name, kind, tuple(shape)))
    return tensors


def weight_summary(tensors: list[Tensor]) -> dict[str, int]:
    """Total values and bytes, and those of the token embedding."""
    embedding = [t for t in tensors if t.name == TOKEN_EMBEDDING]
    return {
        "values": sum(t.values for t in tensors),
        "bytes": sum(t.bytes for t in tensors),
        "embedding_values": sum(t.values for t in embedding),
        "embedding_bytes": sum(t.bytes for t in embedding),
    }
