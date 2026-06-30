from __future__ import annotations

import importlib.util
import sys
from importlib import import_module
from pathlib import Path
from types import ModuleType
from typing import cast

import pytest
import torch

from llm.research_books import RESEARCH_BOOKS, research_book_pairs
from llm.research_reports import BookLossRow, PerBookReport


REPO_ROOT = Path(__file__).parents[2]


def _load_script(name: str) -> ModuleType:
    path = REPO_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_canonical_package_tool_modules_are_importable() -> None:
    for name in (
        "llm.research.eval_per_book",
        "llm.research.destruction_experiments",
        "llm.research.ngram_baseline",
        "llm.training_regression",
    ):
        module = import_module(name)
        assert hasattr(module, "main")


def test_canonical_diagnostic_scripts_are_importable() -> None:
    for name in ("check_gpu", "data_bottleneck_profiler"):
        module = _load_script(name)
        assert hasattr(module, "main")


def test_legacy_diagnostic_script_wrappers_are_importable() -> None:
    for name in ("checkGPU", "DataBottleneckProfiler"):
        module = _load_script(name)
        assert hasattr(module, "main")


def test_legacy_diagnostic_wrappers_reexport_canonical_names() -> None:
    check_gpu = _load_script("checkGPU")
    profiler = _load_script("DataBottleneckProfiler")

    assert hasattr(check_gpu, "parse_args")
    assert hasattr(profiler, "runBottleneckTest")


def test_data_bottleneck_profiler_rejects_invalid_steps() -> None:
    module = _load_script("data_bottleneck_profiler")

    with pytest.raises(SystemExit):
        module.parse_args(["--steps", "0"])

    with pytest.raises(SystemExit):
        module.parse_args(["--warmup-steps", "-1"])


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


def test_context_target_batches_reuse_targets_across_context_lengths() -> None:
    module = _load_script("destruction_experiments")
    generator = torch.Generator().manual_seed(123)

    batches = module.context_target_batches(
        token_count=32,
        max_context=8,
        batch_size=4,
        eval_iters=3,
        generator=generator,
    )

    assert len(batches) == 3
    for targets in batches:
        assert targets.min().item() >= 8
        assert targets.max().item() < 32
        short_starts = targets - 2
        long_starts = targets - 8
        assert torch.equal(short_starts + 2, long_starts + 8)


def test_context_target_indices_reject_short_sequence() -> None:
    module = _load_script("destruction_experiments")

    with pytest.raises(ValueError, match="Sequence too short"):
        module.context_target_indices(
            token_count=8,
            max_context=8,
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


def test_eval_per_book_accepts_full_split_flag() -> None:
    module = _load_script("eval_per_book")

    args = module.parse_args(["--full-split"])

    assert args.full_split is True


def test_per_book_report_serializes_evaluation_policy_metadata() -> None:
    report = PerBookReport(
        checkpoint="runs/example/checkpoints/best.pt",
        device="cpu",
        iters=10,
        full_split=True,
        method="full_stride",
        stride=16,
        seed=42,
        books=[
            BookLossRow(
                book="Emma",
                path="corpora/emma/splits/validation.txt",
                seed=123,
                loss=1.5,
                perplexity=4.48,
                method="full_stride",
                stride=16,
                nTokens=1024,
                nWindows=64,
            )
        ],
        average_loss=1.5,
    )

    data = report.to_json_dict()

    assert data["method"] == "full_stride"
    assert data["stride"] == 16
    books = cast(list[object], data["books"])
    book = books[0]
    assert isinstance(book, dict)
    assert book["method"] == "full_stride"
    assert book["nTokens"] == 1024


@pytest.mark.parametrize("script_name", ["eval_per_book", "destruction_experiments"])
def test_research_scripts_reject_non_positive_iters(script_name: str) -> None:
    module = _load_script(script_name)

    with pytest.raises(SystemExit):
        module.parse_args(["--iters", "0"])

    with pytest.raises(SystemExit):
        module.parse_args(["--iters", "-1"])


def test_ngram_reference_losses_allow_missing_book() -> None:
    module = _load_script("ngram_baseline")

    losses = module.reference_losses_for_books(
        {"Emma": 1.0},
        [("Emma", b"abc"), ("Persuasion", b"def")],
    )

    assert losses is None


def test_ngram_baseline_rejects_non_positive_max_n() -> None:
    module = _load_script("ngram_baseline")

    with pytest.raises(SystemExit):
        module.parse_args(["--max-n", "0"])

    with pytest.raises(SystemExit):
        module.parse_args(["--max-n", "-1"])


def test_ngram_reference_losses_load_from_eval_json(tmp_path: Path) -> None:
    module = _load_script("ngram_baseline")
    reference_path = tmp_path / "eval.json"
    reference_path.write_text(
        '{"books": [{"book": "Emma", "loss": 1.23}]}',
        encoding="utf-8",
    )

    assert module.load_transformer_references(reference_path) == {"Emma": 1.23}


def test_ngram_reference_defaults_to_no_transformer_comparison() -> None:
    module = _load_script("ngram_baseline")

    assert module.load_transformer_references(None) == {}


def test_ngram_reference_can_opt_into_historical_builtin() -> None:
    module = _load_script("ngram_baseline")

    references = module.load_transformer_references(None, use_builtin=True)

    assert references["Emma"] == module.TRANSFORMER_LOSSES_BY_BOOK["Emma"]


def test_corrupt_with_seed_is_independent_of_call_order() -> None:
    module = _load_script("destruction_experiments")

    def corrupt(raw: bytes, _book: str) -> bytes:
        values = [module.random.randrange(0, 256) for _ in raw]
        return bytes(values)

    first = module.corrupt_with_seed(b"abcdef", "Emma", "random", 42, corrupt)
    _ = module.corrupt_with_seed(b"abcdef", "Persuasion", "random", 42, corrupt)
    second = module.corrupt_with_seed(b"abcdef", "Emma", "random", 42, corrupt)

    assert first == second
