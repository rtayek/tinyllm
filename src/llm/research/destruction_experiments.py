"""Structure-destruction experiments for tinyllm.

Each experiment corrupts the validation text in a specific way and measures
how much the loss increases relative to the baseline. The gap reveals which
kinds of structure the model has actually learned.

Usage
-----
    tinyllm-destruction-experiments
    tinyllm-destruction-experiments --checkpoint runs/austen-byte/checkpoints/best.pt
    python -m llm.research.destruction_experiments
    python -m llm.research.destruction_experiments --iters 200 --seed 42

Compatibility wrapper:
    python scripts/destruction_experiments.py
"""
from __future__ import annotations

import argparse
import hashlib
import random
from pathlib import Path
from typing import Callable, Sequence

import torch
import torch.nn.functional as F

from llm.Config import ModelConfig, TrainConfig
from llm.Model import TinyGPTLanguageModel
from llm.cli_utils import require_positive
from llm.corruptions import (
    context_sizes,
    corrupt_random_letters,
    corrupt_reverse,
    corrupt_shuffle_letters,
    corrupt_shuffle_middle,
    corrupt_shuffle_words,
    make_corrupt_replace_names,
)
from llm.json_utils import write_json
from llm.research.research_eval import (
    ResearchBookData,
    estimate_validation_loss,
    load_checkpoint_model,
    load_research_book_data,
)
from llm.research_reports import (
    BaselineRow,
    ContextProbeRow,
    CorruptionBookRow,
    DestructionReport,
    ExperimentRow,
)
from llm.tensor_utils import resolve_device

ExperimentSpec = tuple[str, Callable[[bytes, str], bytes]]


def load_tokens(path: Path) -> torch.Tensor:
    return torch.tensor(bytearray(path.read_bytes()), dtype=torch.long)


def fresh_generator(seed: int) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def stable_seed(seed: int, experiment: str, book_name: str) -> int:
    digest = hashlib.sha256(
        f"{seed}:{experiment}:{book_name}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], byteorder="big")


def corrupt_with_seed(
    raw: bytes,
    book_name: str,
    experiment: str,
    seed: int,
    corrupt_fn: Callable[[bytes, str], bytes],
) -> bytes:
    state = random.getstate()
    try:
        random.seed(stable_seed(seed, experiment, book_name))
        return corrupt_fn(raw, book_name)
    finally:
        random.setstate(state)


def context_start_indices(
    token_count: int,
    window: int,
    batch_size: int,
    generator: torch.Generator,
) -> torch.Tensor:
    high = token_count - window + 1
    if high <= 0:
        raise ValueError(
            f"Sequence too short ({token_count}) for context window {window}"
        )
    return torch.randint(0, high, (batch_size,), generator=generator)


def estimate_loss(
    model: TinyGPTLanguageModel,
    tokens: torch.Tensor,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    seed: int,
) -> float:
    return estimate_validation_loss(
        model,
        tokens,
        model_cfg,
        train_cfg,
        seed,
        "validation",
    )


def estimate_loss_context(
    model: TinyGPTLanguageModel,
    tokens: torch.Tensor,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    seed: int,
    context_len: int,
) -> float:
    """Estimate next-byte loss given exactly ``context_len`` bytes of history."""
    model.eval()
    device = train_cfg.device
    losses: list[float] = []
    generator = fresh_generator(seed)
    window = context_len + 1

    with torch.no_grad():
        for _ in range(train_cfg.evalIters):
            idx = context_start_indices(
                tokens.size(0),
                window,
                train_cfg.batchSize,
                generator,
            )
            offsets = torch.arange(window)
            positions = idx.unsqueeze(1) + offsets.unsqueeze(0)
            block = tokens[positions].to(device)
            block_x = block[:, :context_len]
            target = block[:, context_len]

            logits, _, _ = model(block_x)
            final_logits = logits[:, -1, :]
            loss = F.cross_entropy(final_logits, target)
            losses.append(float(loss.item()))

    return sum(losses) / len(losses)


def fmt_delta(delta: float) -> str:
    if abs(delta) > 1.0:
        return "!!"
    if abs(delta) > 0.2:
        return " !"
    return "  "


def print_experiment_results(
    name: str,
    book_results: list[tuple[str, float, float]],
) -> None:
    print(f"\n  Experiment: {name}")
    print(f"  {'Book':<26}  {'Baseline':>8}  {'Corrupted':>9}  {'Delta':>7}  {'Delta%':>7}")
    print("  " + "-" * 68)

    for book, baseline, corrupted in book_results:
        delta = corrupted - baseline
        pct = 100.0 * delta / baseline if baseline > 0 else 0.0
        marker = fmt_delta(delta)
        print(f"{marker} {book:<26}  {baseline:>8.4f}  {corrupted:>9.4f}  {delta:>+7.4f}  {pct:>+6.1f}%")

    avg_baseline = sum(b for _, b, _ in book_results) / len(book_results)
    avg_corrupted = sum(c for _, _, c in book_results) / len(book_results)
    avg_delta = avg_corrupted - avg_baseline
    avg_pct = 100.0 * avg_delta / avg_baseline
    print("  " + "-" * 68)
    print(f"   {'Average':<26}  {avg_baseline:>8.4f}  {avg_corrupted:>9.4f}  {avg_delta:>+7.4f}  {avg_pct:>+6.1f}%")


def run_baselines(
    model: TinyGPTLanguageModel,
    book_data: list[ResearchBookData],
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    seed: int,
) -> tuple[dict[str, float], list[BaselineRow], float]:
    print("=" * 72)
    print("BASELINE")
    print("=" * 72)
    print(f"  {'Book':<26}  {'Val Loss':>8}  {'Perplexity':>10}")
    print("  " + "-" * 50)

    baselines: dict[str, float] = {}
    rows: list[BaselineRow] = []
    for book in book_data:
        loss = estimate_loss(model, book.tokens, model_cfg, train_cfg, seed)
        perplexity = torch.exp(torch.tensor(loss)).item()
        baselines[book.name] = loss
        rows.append(BaselineRow(book=book.name, loss=loss, perplexity=perplexity))
        print(f"  {book.name:<26}  {loss:>8.4f}  {perplexity:>10.2f}")

    avg_base = sum(baselines.values()) / len(baselines)
    print("  " + "-" * 50)
    print(f"  {'Average':<26}  {avg_base:>8.4f}")
    return baselines, rows, avg_base


def run_context_probe(
    model: TinyGPTLanguageModel,
    book_data: list[ResearchBookData],
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    seed: int,
    avg_base: float,
) -> list[ContextProbeRow]:
    print("\n" + "=" * 72)
    print("CONTEXT WINDOW PROBE")
    print("=" * 72)
    print("  Metric: next-byte loss after exactly N bytes of context")
    print(f"  {'Context':<10}  {'Avg Loss':>8}  {'Delta':>7}  {'Delta%':>7}  {'Marginal':>9}")
    print("  " + "-" * 55)

    rows: list[ContextProbeRow] = []
    prev_avg: float | None = None
    for ctx in context_sizes(model_cfg.blockSize):
        ctx_losses = [
            estimate_loss_context(model, book.tokens, model_cfg, train_cfg, seed, ctx)
            for book in book_data
        ]
        avg = sum(ctx_losses) / len(ctx_losses)
        delta = avg - avg_base
        pct = 100.0 * delta / avg_base
        marginal = f"{avg - prev_avg:+.4f}" if prev_avg is not None else "       -"
        rows.append(
            ContextProbeRow(
                context=ctx,
                average_loss=avg,
                delta=delta,
                delta_pct=pct,
                marginal=avg - prev_avg if prev_avg is not None else None,
            )
        )
        print(f"  context_{ctx:<4}  {avg:>8.4f}  {delta:>+7.4f}  {pct:>+6.1f}%  {marginal:>9}")
        prev_avg = avg
    return rows


def experiment_specs() -> list[ExperimentSpec]:
    return [
        ("shuffle_letters", lambda raw, _: corrupt_shuffle_letters(raw)),
        ("shuffle_middle", lambda raw, _: corrupt_shuffle_middle(raw)),
        ("shuffle_words", lambda raw, _: corrupt_shuffle_words(raw)),
        ("reverse", lambda raw, _: corrupt_reverse(raw)),
        ("random_letters", lambda raw, _: corrupt_random_letters(raw)),
        ("replace_names", lambda raw, book: make_corrupt_replace_names(book)(raw)),
    ]


def run_corruption_experiments(
    model: TinyGPTLanguageModel,
    book_data: list[ResearchBookData],
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    seed: int,
    baselines: dict[str, float],
    no_per_book: bool,
) -> list[ExperimentRow]:
    print("\n" + "=" * 72)
    print("DESTRUCTION EXPERIMENTS")
    print("=" * 72)

    rows: list[ExperimentRow] = []
    for exp_name, corrupt_fn in experiment_specs():
        book_results: list[tuple[str, float, float]] = []
        for book in book_data:
            corrupted_bytes = corrupt_with_seed(
                book.raw,
                book.name,
                exp_name,
                seed,
                corrupt_fn,
            )
            corrupted_tokens = torch.tensor(bytearray(corrupted_bytes), dtype=torch.long)
            corrupted_loss = estimate_loss(
                model, corrupted_tokens, model_cfg, train_cfg, seed
            )
            book_results.append((book.name, baselines[book.name], corrupted_loss))

        rows.append(
            ExperimentRow(
                name=exp_name,
                per_book=[
                    CorruptionBookRow(
                        book=book,
                        baseline=baseline,
                        corrupted=corrupted,
                        delta=corrupted - baseline,
                        delta_pct=100.0 * (corrupted - baseline) / baseline,
                    )
                    for book, baseline, corrupted in book_results
                ],
            )
        )
        if no_per_book:
            print_aggregate_experiment_result(exp_name, book_results)
        else:
            print_experiment_results(exp_name, book_results)
    return rows


def print_aggregate_experiment_result(
    exp_name: str,
    book_results: list[tuple[str, float, float]],
) -> None:
    avg_b = sum(b for _, b, _ in book_results) / len(book_results)
    avg_c = sum(c for _, _, c in book_results) / len(book_results)
    delta = avg_c - avg_b
    pct = 100.0 * delta / avg_b
    marker = fmt_delta(delta)
    print(f"{marker} {exp_name:<26}  {avg_b:>8.4f}  {avg_c:>9.4f}  {delta:>+7.4f}  {pct:>+6.1f}%")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Structure-destruction experiments")
    parser.add_argument("--checkpoint", default="runs/austen-byte/checkpoints/best.pt")
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, default=None, help="Optional JSON output path")
    parser.add_argument(
        "--no-per-book",
        action="store_true",
        help="Skip per-book breakdown; report only aggregate results",
    )
    args = parser.parse_args(argv)
    require_positive(parser, "--iters", args.iters)
    return args


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    print(f"Device:     {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Iters:      {args.iters}  Seed: {args.seed}\n")

    try:
        checkpoint_model = load_checkpoint_model(args.checkpoint, device, args.iters)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return
    model_cfg = checkpoint_model.model_config
    train_cfg = checkpoint_model.train_config
    model = checkpoint_model.model

    book_data, missing = load_research_book_data()
    for book_name, path in missing:
        print(f"  Skipping {book_name} -- {path} not found")
    if not book_data:
        print("No corpus files found.")
        return

    baselines, baseline_rows, avg_base = run_baselines(
        model, book_data, model_cfg, train_cfg, args.seed
    )
    context_rows = run_context_probe(
        model, book_data, model_cfg, train_cfg, args.seed, avg_base
    )
    experiment_rows = run_corruption_experiments(
        model,
        book_data,
        model_cfg,
        train_cfg,
        args.seed,
        baselines,
        args.no_per_book,
    )
    output = DestructionReport(
        checkpoint=args.checkpoint,
        device=device,
        iters=args.iters,
        seed=args.seed,
        block_size=model_cfg.blockSize,
        baseline=baseline_rows,
        context_probe=context_rows,
        experiments=experiment_rows,
    )
    if args.out is not None:
        write_json(args.out, output.to_json_dict())


if __name__ == "__main__":
    main()
