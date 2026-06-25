from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest
import torch

from llm.research_books import RESEARCH_BOOKS, research_book_pairs


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


def test_shared_research_book_registry_has_expected_books() -> None:
    pairs = research_book_pairs()

    assert pairs == [(book.name, book.validationPath) for book in RESEARCH_BOOKS]
    assert pairs[0][0] == "Pride and Prejudice"
    assert pairs[-1][0] == "Alice in Wonderland"


def test_context_sizes_include_configured_block_size() -> None:
    module = _load_script("destruction_experiments")

    assert module.context_sizes(256) == [1, 2, 4, 8, 16, 32, 64, 128, 256]


def test_context_start_indices_allow_single_valid_window() -> None:
    module = _load_script("destruction_experiments")
    generator = torch.Generator()
    generator.manual_seed(0)

    starts = module.context_start_indices(
        token_count=4,
        window=4,
        batch_size=3,
        generator=generator,
    )

    assert starts.tolist() == [0, 0, 0]


def test_context_start_indices_reject_short_sequence() -> None:
    module = _load_script("destruction_experiments")

    with pytest.raises(ValueError, match="Sequence too short"):
        module.context_start_indices(
            token_count=3,
            window=4,
            batch_size=1,
            generator=torch.Generator(),
        )


def test_per_book_generator_is_stable_per_book() -> None:
    module = _load_script("eval_per_book")

    first = module.book_generator(42, "Emma")
    second = module.book_generator(42, "Emma")
    other = module.book_generator(42, "Persuasion")

    assert torch.equal(first.get_state(), second.get_state())
    assert not torch.equal(first.get_state(), other.get_state())


def test_ngram_reference_losses_allow_missing_book() -> None:
    module = _load_script("ngram_baseline")

    losses = module.reference_losses_for_books(
        {"Emma": 1.0},
        [("Emma", b"abc"), ("Persuasion", b"def")],
    )

    assert losses is None


def test_ngram_reference_losses_load_from_eval_json(tmp_path: Path) -> None:
    module = _load_script("ngram_baseline")
    reference_path = tmp_path / "eval.json"
    reference_path.write_text(
        '{"books": [{"book": "Emma", "loss": 1.23}]}',
        encoding="utf-8",
    )

    assert module.load_transformer_references(reference_path) == {"Emma": 1.23}


def test_corrupt_with_seed_is_independent_of_call_order() -> None:
    module = _load_script("destruction_experiments")

    def corrupt(raw: bytes, _book: str) -> bytes:
        values = [module.random.randrange(0, 256) for _ in raw]
        return bytes(values)

    first = module.corrupt_with_seed(b"abcdef", "Emma", "random", 42, corrupt)
    _ = module.corrupt_with_seed(b"abcdef", "Persuasion", "random", 42, corrupt)
    second = module.corrupt_with_seed(b"abcdef", "Emma", "random", 42, corrupt)

    assert first == second
