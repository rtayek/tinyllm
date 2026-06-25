"""Structure-destruction experiments for tinyllm.

Each experiment corrupts the validation text in a specific way and measures
how much the loss increases relative to the baseline.  The gap reveals which
kinds of structure the model has actually learned.

Experiments
-----------
baseline            Raw validation text, no modification.
context_N           Restrict effective context to powers of two through blockSize.
shuffle_letters     Shuffle all characters within each word.
shuffle_middle      Shuffle only the middle characters; preserve first and last letter.
shuffle_words       Shuffle word order within each sentence.
reverse             Reverse the entire byte sequence.
random_letters      Replace every ASCII letter with a random letter a-z.
replace_names       Replace Austen character names with placeholders.

All experiments use the same checkpoint, block size, batch size, and loss
calculation.  Only the text changes.

Usage
-----
    python scripts/destruction_experiments.py
    python scripts/destruction_experiments.py --checkpoint runs/austen-byte/checkpoints/best.pt
    python scripts/destruction_experiments.py --iters 200 --seed 42
    python scripts/destruction_experiments.py --no-per-book
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import torch.nn.functional as F

from llm.Checkpoint import Checkpoint
from llm.Config import ModelConfig, TrainConfig
from llm.DataModule import SequenceDataModule
from llm.Evaluator import Evaluator
from llm.EarlyStopping import EarlyStopping
from llm.Model import TinyGPTLanguageModel
from llm.research_books import RESEARCH_BOOKS
from llm.tensor_utils import resolve_device
from llm.corruptions import (
    PLACEHOLDERS,
    context_sizes,
    corrupt_random_letters,
    corrupt_reverse,
    corrupt_shuffle_letters,
    corrupt_shuffle_middle,
    corrupt_shuffle_words,
    make_corrupt_replace_names,
)

# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def load_tokens(path: Path) -> torch.Tensor:
    return torch.tensor(bytearray(path.read_bytes()), dtype=torch.long)


def fresh_generator(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


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
    data_module = SequenceDataModule(
        model_cfg, train_cfg,
        sequence=tokens,
        validationSequence=tokens,
    )
    evaluator = Evaluator(
        model, data_module, train_cfg, EarlyStopping(patience=1, delta=0.0)
    )
    return evaluator.estimate_split("val", fresh_generator(seed))


def estimate_loss_context(
    model: TinyGPTLanguageModel,
    tokens: torch.Tensor,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    seed: int,
    context_len: int,
) -> float:
    """Estimate next-byte loss given exactly ``context_len`` bytes of history.

    A genuinely shorter sequence of length ``context_len`` is fed to the model
    (relying on the positional-embedding slice for variable-length input), and
    loss is measured only on the single final prediction.  This is a true
    fixed-context measurement: the model predicts the next byte using exactly
    ``context_len`` real preceding bytes.

    This deliberately avoids the earlier zero-padding approach, which left the
    leading positions filled with null bytes (token 0 is a real, embedded
    token).  Padding contaminated small-context measurements by making the
    model condition on a run of null bytes rather than on a genuinely short
    context.
    """
    model.eval()
    device = train_cfg.device
    losses: list[float] = []
    g = fresh_generator(seed)

    # context_len real bytes of history, predicting the byte that follows.
    window = context_len + 1

    with torch.no_grad():
        for _ in range(train_cfg.evalIters):
            idx = context_start_indices(
                tokens.size(0),
                window,
                train_cfg.batchSize,
                g,
            )
            offsets = torch.arange(window)
            positions = idx.unsqueeze(1) + offsets.unsqueeze(0)
            block = tokens[positions].to(device)        # (B, context_len + 1)
            block_x = block[:, :context_len]            # (B, context_len)
            target = block[:, context_len]              # (B,) the next byte

            logits, _, _ = model(block_x)               # (B, context_len, vocab)
            final_logits = logits[:, -1, :]             # (B, vocab): predict next
            loss = F.cross_entropy(final_logits, target)
            losses.append(float(loss.item()))

    return sum(losses) / len(losses)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def fmt_delta(delta: float) -> str:
    marker = "  "
    if abs(delta) > 1.0:
        marker = "!!"
    elif abs(delta) > 0.2:
        marker = " !"
    return marker


def print_experiment_results(
    name: str,
    book_results: list[tuple[str, float, float]],  # (book, baseline, corrupted)
) -> None:
    """Print per-book and aggregate results for one experiment."""
    print(f"\n  Experiment: {name}")
    print(f"  {'Book':<26}  {'Baseline':>8}  {'Corrupted':>9}  {'Delta':>7}  {'Delta%':>7}")
    print("  " + "-" * 68)

    deltas: list[float] = []
    for book, baseline, corrupted in book_results:
        delta = corrupted - baseline
        pct = 100.0 * delta / baseline if baseline > 0 else 0.0
        marker = fmt_delta(delta)
        print(f"{marker} {book:<26}  {baseline:>8.4f}  {corrupted:>9.4f}  {delta:>+7.4f}  {pct:>+6.1f}%")
        deltas.append(delta)

    avg_baseline = sum(b for _, b, _ in book_results) / len(book_results)
    avg_corrupted = sum(c for _, _, c in book_results) / len(book_results)
    avg_delta = avg_corrupted - avg_baseline
    avg_pct = 100.0 * avg_delta / avg_baseline
    print("  " + "-" * 68)
    print(f"   {'Average':<26}  {avg_baseline:>8.4f}  {avg_corrupted:>9.4f}  {avg_delta:>+7.4f}  {avg_pct:>+6.1f}%")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Structure-destruction experiments")
    parser.add_argument("--checkpoint", default="runs/austen-byte/checkpoints/best.pt")
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, default=None, help="Optional JSON output path")
    parser.add_argument(
        "--no-per-book", action="store_true",
        help="Skip per-book breakdown; report only aggregate results",
    )
    args = parser.parse_args()

    device = resolve_device(args.device)

    print(f"Device:     {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Iters:      {args.iters}  Seed: {args.seed}\n")

    # Load model
    checkpoint = Checkpoint.load(args.checkpoint, device)
    if checkpoint.modelConfig is None:
        print("ERROR: checkpoint has no modelConfig")
        return
    model_cfg = ModelConfig.fromDict(checkpoint.modelConfig)
    train_cfg = TrainConfig(device=device, evalIters=args.iters)
    model = TinyGPTLanguageModel(model_cfg).to(device)
    model.load_state_dict(checkpoint.modelState)
    model.eval()

    # Load all book data
    book_data: list[tuple[str, bytes, torch.Tensor]] = []
    for book in RESEARCH_BOOKS:
        book_name = book.name
        path = book.validation_path
        if not path.exists():
            print(f"  Skipping {book_name} — {path} not found")
            continue
        raw = path.read_bytes()
        tokens = load_tokens(path)
        book_data.append((book_name, raw, tokens))

    if not book_data:
        print("No corpus files found.")
        return

    # --- Baselines ---
    print("=" * 72)
    print("BASELINE")
    print("=" * 72)
    print(f"  {'Book':<26}  {'Val Loss':>8}  {'Perplexity':>10}")
    print("  " + "-" * 50)
    baselines: dict[str, float] = {}
    output: dict[str, object] = {
        "checkpoint": args.checkpoint,
        "device": device,
        "iters": args.iters,
        "seed": args.seed,
        "block_size": model_cfg.blockSize,
        "baseline": [],
        "context_probe": [],
        "experiments": [],
    }
    for book_name, _, tokens in book_data:
        loss = estimate_loss(model, tokens, model_cfg, train_cfg, args.seed)
        perp = torch.exp(torch.tensor(loss)).item()
        baselines[book_name] = loss
        baseline_rows = output["baseline"]
        assert isinstance(baseline_rows, list)
        baseline_rows.append(
            {"book": book_name, "loss": loss, "perplexity": perp}
        )
        print(f"  {book_name:<26}  {loss:>8.4f}  {perp:>10.2f}")
    avg_base = sum(baselines.values()) / len(baselines)
    print("  " + "-" * 50)
    print(f"  {'Average':<26}  {avg_base:>8.4f}")

    # --- Context window probe ---
    print("\n" + "=" * 72)
    print("CONTEXT WINDOW PROBE")
    print("=" * 72)
    print("  Metric: next-byte loss after exactly N bytes of context")
    print(f"  {'Context':<10}  {'Avg Loss':>8}  {'Delta':>7}  {'Delta%':>7}  {'Marginal':>9}")
    print("  " + "-" * 55)
    prev_avg: float | None = None
    for ctx in context_sizes(model_cfg.blockSize):
        ctx_losses = []
        for book_name, _, tokens in book_data:
            loss = estimate_loss_context(model, tokens, model_cfg, train_cfg, args.seed, ctx)
            ctx_losses.append(loss)
        avg = sum(ctx_losses) / len(ctx_losses)
        delta = avg - avg_base
        pct = 100.0 * delta / avg_base
        marginal = f"{avg - prev_avg:+.4f}" if prev_avg is not None else "       -"
        context_rows = output["context_probe"]
        assert isinstance(context_rows, list)
        context_rows.append(
            {
                "context": ctx,
                "average_loss": avg,
                "delta": delta,
                "delta_pct": pct,
                "marginal": avg - prev_avg if prev_avg is not None else None,
            }
        )
        print(f"  context_{ctx:<4}  {avg:>8.4f}  {delta:>+7.4f}  {pct:>+6.1f}%  {marginal:>9}")
        prev_avg = avg

    # --- Corruption experiments ---
    print("\n" + "=" * 72)
    print("DESTRUCTION EXPERIMENTS")
    print("=" * 72)

    experiments: list[tuple[str, Callable[[bytes, str], bytes]]] = [
        ("shuffle_letters",  lambda raw, _:   corrupt_shuffle_letters(raw)),
        ("shuffle_middle",   lambda raw, _:   corrupt_shuffle_middle(raw)),
        ("shuffle_words",    lambda raw, _:   corrupt_shuffle_words(raw)),
        ("reverse",          lambda raw, _:   corrupt_reverse(raw)),
        ("random_letters",   lambda raw, _:   corrupt_random_letters(raw)),
        ("replace_names",    lambda raw, bk:  make_corrupt_replace_names(bk)(raw)),
    ]

    for exp_name, corrupt_fn in experiments:
        book_results: list[tuple[str, float, float]] = []
        for book_name, raw, _ in book_data:
            corrupted_bytes = corrupt_with_seed(
                raw,
                book_name,
                exp_name,
                args.seed,
                corrupt_fn,
            )
            corrupted_tokens = torch.tensor(bytearray(corrupted_bytes), dtype=torch.long)
            corrupted_loss = estimate_loss(
                model, corrupted_tokens, model_cfg, train_cfg, args.seed
            )
            book_results.append((book_name, baselines[book_name], corrupted_loss))
        experiment_rows = output["experiments"]
        assert isinstance(experiment_rows, list)
        experiment_rows.append(
            {
                "name": exp_name,
                "per_book": [
                    {
                        "book": book,
                        "baseline": baseline,
                        "corrupted": corrupted,
                        "delta": corrupted - baseline,
                        "delta_pct": 100.0 * (corrupted - baseline) / baseline,
                    }
                    for book, baseline, corrupted in book_results
                ],
            }
        )

        if args.no_per_book:
            avg_b = sum(b for _, b, _ in book_results) / len(book_results)
            avg_c = sum(c for _, _, c in book_results) / len(book_results)
            delta = avg_c - avg_b
            pct = 100.0 * delta / avg_b
            marker = fmt_delta(delta)
            print(f"{marker} {exp_name:<26}  {avg_b:>8.4f}  {avg_c:>9.4f}  {delta:>+7.4f}  {pct:>+6.1f}%")
        else:
            print_experiment_results(exp_name, book_results)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
