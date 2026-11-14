"""Download GPT-2 (124M)'s weights and save its matrices for qmm_accuracy.

Downloads model.safetensors from Hugging Face at a pinned revision, checks its
SHA-256, and writes each matrix the benchmarks model as raw little-endian fp32
in the kernels' n x k layout, with an index.txt that qmm_accuracy reads:

- for each of the 12 layers: attn_qkv (c_attn), attn_out (attn.c_proj),
  mlp_up (c_fc) and mlp_down (mlp.c_proj). GPT-2 stores these k x n (its
  Conv1D layers), so they are transposed.
- lm_head: the token embedding, wte, which GPT-2 reuses as its output layer.
  It is already n x k (vocabulary x 768).

Uses only the standard library. The files take about 1 GB and are not
committed; the default folder, data/gpt2 in this project, is ignored by git.

Usage, from anywhere in the repository:
    uv run python projects/quant-matmul/scripts/fetch_gpt2.py [--dir DIR]
"""

import argparse
import hashlib
import json
import struct
import sys
import urllib.request
from array import array
from pathlib import Path
from typing import Any

PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = PROJECT / "data" / "gpt2"

REPO_ID = "openai-community/gpt2"
REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
WEIGHTS_SHA256 = "248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707"
WEIGHTS_URL = f"https://huggingface.co/{REPO_ID}/resolve/{REVISION}/model.safetensors"
LAYERS = 12

# (tensor name pattern, benchmark shape, stored k x n and so transposed)
LAYER_TENSORS = [
    ("h.{layer}.attn.c_attn.weight", "attn_qkv", True),
    ("h.{layer}.attn.c_proj.weight", "attn_out", True),
    ("h.{layer}.mlp.c_fc.weight", "mlp_up", True),
    ("h.{layer}.mlp.c_proj.weight", "mlp_down", True),
]
EMBEDDING = ("wte.weight", "lm_head", False)

CHUNK = 1 << 20


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, dest: Path, sha256: str) -> None:
    """Download url to dest unless dest already has the expected SHA-256.

    Writes to a temporary file first, so an interrupted download never leaves
    a file that looks complete.
    """
    if dest.exists() and sha256_of(dest) == sha256:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + ".part")
    digest = hashlib.sha256()
    with urllib.request.urlopen(url) as response, partial.open("wb") as out:
        while chunk := response.read(CHUNK):
            digest.update(chunk)
            out.write(chunk)
    if digest.hexdigest() != sha256:
        partial.unlink()
        raise ValueError(f"{url}: SHA-256 {digest.hexdigest()}, expected {sha256}")
    partial.replace(dest)


def read_header(path: Path) -> tuple[dict[str, Any], int]:
    """A safetensors file's header and the offset where its tensor data starts.

    The format is an 8-byte little-endian header length, a JSON header, then
    the tensors' bytes, which the header locates by offsets from its end.
    """
    with path.open("rb") as f:
        (length,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(length))
    return header, 8 + length


def read_tensor(path: Path, header: dict[str, Any], data_start: int, name: str) -> array:
    """One fp32 tensor's values, flattened in row-major order."""
    info = header[name]
    if info["dtype"] != "F32":
        raise ValueError(f"{name}: dtype {info['dtype']}, expected F32")
    begin, end = info["data_offsets"]
    with path.open("rb") as f:
        f.seek(data_start + begin)
        data = f.read(end - begin)
    values = array("f")
    values.frombytes(data)
    if sys.byteorder == "big":
        values.byteswap()  # safetensors is little-endian
    return values


def transpose(values: array, rows: int, cols: int) -> array:
    """A rows x cols row-major matrix as cols x rows."""
    assert len(values) == rows * cols
    out = array("f")
    for col in range(cols):
        out.extend(values[col::cols])
    return out


def write_matrix(values: array, path: Path) -> None:
    out = array("f", values)
    if sys.byteorder == "big":
        out.byteswap()
    path.write_bytes(out.tobytes())


def save_matrices(
    weights: Path, out_dir: Path, tensors: list[tuple[str, str, str, bool]], comments: list[str]
) -> None:
    """Write each (tensor, source, shape, transposed) as <source>_<shape>.f32, and index.txt."""
    header, data_start = read_header(weights)
    lines = [f"# {c}" for c in comments]
    for name, source, shape, transposed in tensors:
        rows, cols = header[name]["shape"]
        values = read_tensor(weights, header, data_start, name)
        n, k = (cols, rows) if transposed else (rows, cols)
        if transposed:
            values = transpose(values, rows, cols)
        file_name = f"{source}_{shape}.f32"
        write_matrix(values, out_dir / file_name)
        lines.append(f"{source} {shape} {n} {k} {file_name}")
    (out_dir / "index.txt").write_text("\n".join(lines) + "\n")


def gpt2_tensors() -> list[tuple[str, str, str, bool]]:
    tensors = [
        (pattern.format(layer=layer), f"h{layer}", shape, transposed)
        for layer in range(LAYERS)
        for pattern, shape, transposed in LAYER_TENSORS
    ]
    name, shape, transposed = EMBEDDING
    tensors.append((name, "wte", shape, transposed))
    return tensors


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch GPT-2's weights for qmm_accuracy.")
    parser.add_argument("--dir", type=Path, default=DEFAULT_DIR, help="where to save them")
    args = parser.parse_args()

    weights = args.dir / "model.safetensors"
    print(f"fetch_gpt2: downloading {WEIGHTS_URL}", file=sys.stderr)
    download(WEIGHTS_URL, weights, WEIGHTS_SHA256)
    comments = [f"{REPO_ID} revision {REVISION}", f"model.safetensors sha256 {WEIGHTS_SHA256}"]
    save_matrices(weights, args.dir, gpt2_tensors(), comments)
    print(f"fetch_gpt2: matrices and index.txt in {args.dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
