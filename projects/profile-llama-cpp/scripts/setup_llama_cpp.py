"""Set up llama.cpp and the Qwen2.5-1.5B models for profiling.

Both live outside this repository, at paths taken from the environment:

- LLAMA_CPP_DIR: a clone of llama.cpp with its full history (blobless, so
  file contents download on demand), which can later carry a fork remote and
  branches for upstream work. Each pinned release gets its own git worktree
  beside it, <LLAMA_CPP_DIR>-<tag>, so a later release can be compared without
  disturbing the first.
- MODELS_DIR: the GGUF models, downloaded at a pinned revision and checked by
  SHA-256, where other projects can reuse them.

In the worktree, it builds release binaries with debug information (which
doesn't change the generated code) in one folder per configuration:

- build-cpu: no Metal, no BLAS and no Accelerate, so nothing runs on a GPU
  and every operation runs in ggml's own kernels, as in a default Linux
  build. The main profiles use it.
- build-accelerate (macOS only): macOS builds' default use of Accelerate, as
  BLAS and for fp32 element-wise operations. BLAS takes over prompt
  processing's large matrix multiplies, and this shows how much of that speed
  comes from it.

Then it runs a short generation with each model as a smoke test. Re-running
it skips whatever is already in place.

Usage, from anywhere in the repository, with both variables set:
    uv run python projects/profile-llama-cpp/scripts/setup_llama_cpp.py
"""

import hashlib
import os
import subprocess
import sys
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

REPO_URL = "https://github.com/ggml-org/llama.cpp"
TAG = "b12377"
COMMIT = "32bdc3a89e6528d3e419024c833c5dabf32c7541"

MODEL_REPO = "Qwen/Qwen2.5-1.5B-Instruct-GGUF"
MODEL_REVISION = "91cad51170dc346986eccefdc2dd33a9da36ead9"
MODEL_FOLDER = "qwen2.5-1.5b-instruct"


@dataclass(frozen=True)
class Model:
    file: str
    sha256: str

    @property
    def url(self) -> str:
        return f"https://huggingface.co/{MODEL_REPO}/resolve/{MODEL_REVISION}/{self.file}"


MODELS = [
    Model(
        "qwen2.5-1.5b-instruct-q8_0.gguf",
        "d7efb072e7724d25048a4fda0a3e10b04bdef5d06b1403a1c93bd9f1240a63c8",
    ),
    Model(
        "qwen2.5-1.5b-instruct-q4_k_m.gguf",
        "6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e",
    ),
]

TARGETS = ["llama-bench", "llama-simple", "test-backend-ops"]
SMOKE_PROMPT = "The capital of France is"
SMOKE_TOKENS = 16

CHUNK = 1 << 20


def required_dir(env: Mapping[str, str], name: str) -> Path:
    """The folder an environment variable names, with ~ expanded."""
    value = env.get(name)
    if not value:
        raise ValueError(f"set {name} to the folder to use (see this script's docstring)")
    return Path(value).expanduser()


def worktree_path(clone: Path, tag: str) -> Path:
    return clone.with_name(f"{clone.name}-{tag}")


def cmake_configs(platform: str) -> dict[str, list[str]]:
    """The CMake options of each build folder."""
    common = ["-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release", "-DGGML_NATIVE=ON", "-DGGML_METAL=OFF"]
    # PLATFORM: unix — -g is GCC and Clang's flag for debug information, which
    # the profilers use to map samples to source lines; MSVC would need /Zi.
    if platform != "win32":
        common += ["-DCMAKE_C_FLAGS=-g", "-DCMAKE_CXX_FLAGS=-g"]
    # GGML_ACCELERATE, which only acts on macOS, would also hand fp32
    # element-wise operations (adds, scaling, sums, maxima, norms) to
    # Accelerate's vDSP instead of ggml's own SIMD code.
    configs = {"build-cpu": [*common, "-DGGML_BLAS=OFF", "-DGGML_ACCELERATE=OFF"]}
    # PLATFORM: macos — Accelerate is macOS's BLAS and its builds' default;
    # elsewhere the comparable run would use OpenBLAS (GGML_BLAS_VENDOR=OpenBLAS).
    if platform == "darwin":
        configs["build-accelerate"] = [*common, "-DGGML_BLAS=ON", "-DGGML_BLAS_VENDOR=Apple"]
    return configs


def executable(build_dir: Path, name: str, platform: str) -> Path:
    # PLATFORM: windows — executables carry .exe; elsewhere they have no suffix.
    suffix = ".exe" if platform == "win32" else ""
    return build_dir / "bin" / f"{name}{suffix}"


def run(cmd: list[str]) -> None:
    """Run a command, with its output going straight to the terminal."""
    print("$", " ".join(cmd), file=sys.stderr, flush=True)
    subprocess.run(cmd, check=True)


def git(repo: Path, *args: str) -> str:
    """A git command's output, stripped."""
    cmd = ["git", "-C", str(repo), *args]
    print("$", " ".join(cmd), file=sys.stderr, flush=True)
    return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout.strip()


def ensure_clone(url: str, clone: Path) -> None:
    if (clone / ".git").exists():
        return
    clone.parent.mkdir(parents=True, exist_ok=True)
    run(["git", "clone", "--filter=blob:none", url, str(clone)])


def ensure_worktree(clone: Path, tag: str, commit: str, dest: Path) -> None:
    """Check out the tag in its own worktree, after checking it is the pinned commit."""
    git(clone, "fetch", "--tags", "origin")
    tagged = git(clone, "rev-parse", f"{tag}^{{commit}}")
    if tagged != commit:
        raise ValueError(f"tag {tag} is commit {tagged}, expected {commit}")
    if not dest.exists():
        git(clone, "worktree", "add", "--detach", str(dest), commit)
    head = git(dest, "rev-parse", "HEAD")
    if head != commit:
        raise ValueError(f"{dest} is at {head}, expected {commit}")


def build(source: Path, build_dir: Path, options: list[str]) -> None:
    run(["cmake", "-S", str(source), "-B", str(build_dir), *options])
    run(["cmake", "--build", str(build_dir), "--target", *TARGETS])


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


def main() -> int:
    try:
        clone = required_dir(os.environ, "LLAMA_CPP_DIR")
        models = required_dir(os.environ, "MODELS_DIR") / MODEL_FOLDER
    except ValueError as e:
        print(f"setup_llama_cpp: {e}", file=sys.stderr)
        return 2

    ensure_clone(REPO_URL, clone)
    source = worktree_path(clone, TAG)
    ensure_worktree(clone, TAG, COMMIT, source)
    configs = cmake_configs(sys.platform)
    for folder, options in configs.items():
        build(source, source / folder, options)

    for model in MODELS:
        print(f"setup_llama_cpp: downloading {model.url}", file=sys.stderr)
        download(model.url, models / model.file, model.sha256)

    simple = executable(source / "build-cpu", "llama-simple", sys.platform)
    for model in MODELS:
        cmd = [str(simple), "-m", str(models / model.file), "-n", str(SMOKE_TOKENS)]
        run([*cmd, "-ngl", "0", SMOKE_PROMPT])

    print(f"setup_llama_cpp: llama.cpp {TAG} in {source}, models in {models}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
