from pathlib import Path

import pytest
from runkit import MODEL_KEYS, llama_source, model_for, models_dir, split_list
from setup_llama_cpp import MODEL_FOLDER, TAG


def test_paths_from_the_environment() -> None:
    env = {"LLAMA_CPP_DIR": "/src/llama.cpp", "MODELS_DIR": "/models"}
    assert llama_source(env) == Path("/src") / f"llama.cpp-{TAG}"
    assert models_dir(env) == Path("/models") / MODEL_FOLDER
    with pytest.raises(ValueError, match="MODELS_DIR"):
        models_dir({})


def test_model_keys() -> None:
    assert model_for("q8_0").file.endswith("q8_0.gguf")
    assert model_for("q4_k_m").file.endswith("q4_k_m.gguf")
    assert set(MODEL_KEYS) == {"q8_0", "q4_k_m"}
    with pytest.raises(ValueError, match="q8_0, q4_k_m"):
        model_for("f16")


def test_split_list() -> None:
    assert split_list(" pp, tg ,") == ["pp", "tg"]
