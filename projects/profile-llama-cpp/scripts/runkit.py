"""Helpers specific to this project's run scripts: llama.cpp's paths, build and revisions.

The helpers every project's benchmark scripts share (machine details, thermal
state, run folders, keeping the machine awake) are in common/benchkit.
"""

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from benchkit import output_of, parse_cmake_cache
from setup_llama_cpp import MODEL_FOLDER, MODELS, TAG, Model, required_dir, worktree_path

PROJECT = Path(__file__).resolve().parents[1]
REPO = PROJECT.parents[1]
RESULTS = PROJECT / "results"
QUANT_MATMUL = REPO / "projects" / "quant-matmul"

# Short names for the models, as the scripts' options take them.
MODEL_KEYS = {"q8_0": MODELS[0], "q4_k_m": MODELS[1]}


def llama_source(env: Mapping[str, str]) -> Path:
    """The pinned llama.cpp worktree, from LLAMA_CPP_DIR."""
    return worktree_path(required_dir(env, "LLAMA_CPP_DIR"), TAG)


def models_dir(env: Mapping[str, str]) -> Path:
    """The Qwen models' folder, from MODELS_DIR."""
    return required_dir(env, "MODELS_DIR") / MODEL_FOLDER


def model_for(key: str) -> Model:
    if key not in MODEL_KEYS:
        raise ValueError(f"unknown model {key!r}; choose from {', '.join(MODEL_KEYS)}")
    return MODEL_KEYS[key]


def split_list(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def build_details(build_dir: Path) -> dict[str, Any]:
    """The llama.cpp build's ggml options, flags and compiler."""
    cache = parse_cmake_cache((build_dir / "CMakeCache.txt").read_text())
    wanted = {"CMAKE_BUILD_TYPE", "CMAKE_C_FLAGS", "CMAKE_CXX_FLAGS", "CMAKE_CXX_COMPILER"}
    settings = {k: v for k, v in cache.items() if k in wanted or k.startswith("GGML_")}
    compiler: dict[str, str] = {}
    for info in build_dir.glob("CMakeFiles/*/CMakeCXXCompiler.cmake"):
        for key in ("CMAKE_CXX_COMPILER_ID", "CMAKE_CXX_COMPILER_VERSION"):
            match = re.search(rf'set\({key} "([^"]*)"\)', info.read_text())
            if match:
                compiler[key] = match.group(1)
    return {"folder": build_dir.name, "cache": settings, "compiler": compiler}


def revisions(source: Path) -> dict[str, Any]:
    """This repository's and llama.cpp's checked-out revisions, and whether either differs.

    This project's results are left out, so earlier runs that aren't committed
    yet don't mark the repository as changed.
    """
    excluded = f":(exclude){RESULTS.relative_to(REPO).as_posix()}"
    repo_status = output_of(["git", "-C", str(REPO), "status", "--porcelain", "--", ".", excluded])
    llama_status = output_of(["git", "-C", str(source), "status", "--porcelain"])
    return {
        "repo": output_of(["git", "-C", str(REPO), "rev-parse", "HEAD"]),
        "repo_uncommitted_changes": repo_status != "",
        "llama_cpp_tag": TAG,
        "llama_cpp": output_of(["git", "-C", str(source), "rev-parse", "HEAD"]),
        "llama_cpp_uncommitted_changes": llama_status != "",
    }
