import hashlib
import subprocess
from pathlib import Path

import pytest
from setup_llama_cpp import (
    cmake_configs,
    download,
    ensure_clone,
    ensure_worktree,
    executable,
    required_dir,
    worktree_path,
)


def test_required_dir_expands_home() -> None:
    assert required_dir({"X": "~/models"}, "X") == Path.home() / "models"


@pytest.mark.parametrize("env", [{}, {"X": ""}])
def test_required_dir_names_a_missing_variable(env: dict[str, str]) -> None:
    with pytest.raises(ValueError, match="set X"):
        required_dir(env, "X")


def test_worktree_sits_beside_the_clone() -> None:
    assert worktree_path(Path("src") / "llama.cpp", "b1") == Path("src") / "llama.cpp-b1"


def test_cpu_build_has_no_gpu_and_no_blas() -> None:
    for platform in ["darwin", "linux", "win32"]:
        cpu = cmake_configs(platform)["build-cpu"]
        assert "-DGGML_METAL=OFF" in cpu
        assert "-DGGML_BLAS=OFF" in cpu
        assert "-DGGML_ACCELERATE=OFF" in cpu
        assert "-DCMAKE_BUILD_TYPE=Release" in cpu


def test_accelerate_build_only_on_macos() -> None:
    assert cmake_configs("darwin")["build-accelerate"][-2:] == [
        "-DGGML_BLAS=ON",
        "-DGGML_BLAS_VENDOR=Apple",
    ]
    assert list(cmake_configs("linux")) == ["build-cpu"]


def test_debug_information_except_with_msvc() -> None:
    assert "-DCMAKE_CXX_FLAGS=-g" in cmake_configs("linux")["build-cpu"]
    assert "-DCMAKE_CXX_FLAGS=-g" not in cmake_configs("win32")["build-cpu"]


def test_executable_suffix() -> None:
    assert executable(Path("b"), "llama-bench", "linux") == Path("b") / "bin" / "llama-bench"
    assert executable(Path("b"), "llama-bench", "win32") == Path("b") / "bin" / "llama-bench.exe"


def git(repo: Path, *args: str) -> str:
    identity = ["-c", "user.name=test", "-c", "user.email=test@example.invalid"]
    cmd = ["git", *identity, "-C", str(repo), *args]
    return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def upstream(tmp_path: Path) -> Path:
    """A local repository with a tagged commit, standing in for llama.cpp."""
    repo = tmp_path / "upstream"
    repo.mkdir()
    git(repo, "init", "-q")
    (repo / "README").write_text("v1\n")
    git(repo, "add", "README")
    git(repo, "commit", "-q", "-m", "v1")
    git(repo, "tag", "b1")
    return repo


def test_clone_and_worktree_at_the_pinned_commit(tmp_path: Path, upstream: Path) -> None:
    commit = git(upstream, "rev-parse", "HEAD")
    clone = tmp_path / "src" / "llama.cpp"
    ensure_clone(upstream.as_uri(), clone)
    ensure_clone(upstream.as_uri(), clone)  # a second run keeps the clone

    dest = worktree_path(clone, "b1")
    ensure_worktree(clone, "b1", commit, dest)
    ensure_worktree(clone, "b1", commit, dest)  # and the worktree
    assert git(dest, "rev-parse", "HEAD") == commit
    assert (dest / "README").read_text() == "v1\n"


def test_worktree_refuses_a_tag_at_another_commit(tmp_path: Path, upstream: Path) -> None:
    clone = tmp_path / "llama.cpp"
    ensure_clone(upstream.as_uri(), clone)
    with pytest.raises(ValueError, match="expected 0000"):
        ensure_worktree(clone, "b1", "0" * 40, worktree_path(clone, "b1"))
    assert not worktree_path(clone, "b1").exists()


def test_download_checks_the_hash(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"weights")
    good = hashlib.sha256(b"weights").hexdigest()
    dest = tmp_path / "out" / "model.gguf"

    download(source.as_uri(), dest, good)
    assert dest.read_bytes() == b"weights"

    other = tmp_path / "other.gguf"
    with pytest.raises(ValueError, match="SHA-256"):
        download(source.as_uri(), other, "0" * 64)
    assert not other.exists()
    assert not other.with_name("other.gguf.part").exists()
