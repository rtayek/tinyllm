from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


REPO_ROOT = Path(__file__).parents[2]


def _load_script(name: str) -> ModuleType:
    path = REPO_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ngram_training_counts_final_prediction() -> None:
    module = _load_script("ngram_baseline")
    model = module.NgramModel(2)

    model.train(b"abc")

    assert model.counts[(97,)][98] == 1
    assert model.counts[(98,)][99] == 1
    assert model.cross_entropy(b"bc") < 6.0


def test_context_sizes_include_configured_block_size() -> None:
    module = _load_script("destruction_experiments")

    assert module.context_sizes(256) == [1, 2, 4, 8, 16, 32, 64, 128, 256]
