"""N-gram baseline models for tinyllm.

Trains unigram through N-gram models on the combined training corpus and
evaluates them on each book's validation split.  Reports cross-entropy loss
in nats so results are directly comparable to the transformer's val loss.

Usage
-----
    python scripts/ngram_baseline.py
    python scripts/ngram_baseline.py --max-n 5
    python scripts/ngram_baseline.py --train corpora/jane-austen/combined/splits/train.txt
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence, cast

from llm.cli_utils import require_positive
from llm.json_utils import write_json
from llm.research_books import RESEARCH_BOOKS
from llm.research_reports import NgramBookLoss, NgramReport, NgramResult

DEFAULT_TRAIN = "corpora/jane-austen/combined/splits/train.txt"
VOCAB_SIZE = 256

# Transformer val losses from step 9500 run (seed 42, 200 iters)
TRANSFORMER_LOSSES_BY_BOOK = {
    "Pride and Prejudice": 1.4305,
    "Sense and Sensibility": 1.4724,
    "Emma": 1.4220,
    "Mansfield Park": 1.4774,
    "Persuasion": 1.4312,
    "Northanger Abbey": 1.4811,
    "Sherlock Holmes": 1.6764,
    "Alice in Wonderland": 1.7941,
}


class NgramModel:
    """Byte-level n-gram language model with Laplace (add-1) smoothing.

    Cross-entropy is reported in nats to match PyTorch cross_entropy,
    making results directly comparable to the transformer's val loss.
    """

    def __init__(self, n: int, vocab_size: int = VOCAB_SIZE) -> None:
        self.n = n
        self.vocab_size = vocab_size
        self.counts: defaultdict[tuple[int, ...], Counter[int]] = defaultdict(Counter)
        self.context_totals: Counter[tuple[int, ...]] = Counter()

    def train(self, data: bytes) -> None:
        n = self.n
        for i in range(len(data) - n + 1):
            context = tuple(data[i:i + n - 1]) if n > 1 else ()
            next_byte = data[i + n - 1]
            self.counts[context][next_byte] += 1
            self.context_totals[context] += 1

    def log_prob(self, context: tuple[int, ...], next_byte: int) -> float:
        count = self.counts[context][next_byte]
        total = self.context_totals[context]
        prob = (count + 1) / (total + self.vocab_size)
        return math.log(prob)

    def cross_entropy(self, data: bytes) -> float:
        n = self.n
        total_log_prob = 0.0
        num_predictions = 0
        for i in range(len(data) - n + 1):
            context = tuple(data[i:i + n - 1]) if n > 1 else ()
            next_byte = data[i + n - 1]
            total_log_prob += self.log_prob(context, next_byte)
            num_predictions += 1
        if num_predictions == 0:
            return float("inf")
        return -total_log_prob / num_predictions


def load_transformer_references(path: Path | None) -> dict[str, float]:
    if path is None:
        return dict(TRANSFORMER_LOSSES_BY_BOOK)

    raw_data: Any = json.loads(path.read_text(encoding="utf-8"))
    data = cast(object, raw_data)
    if not isinstance(data, dict):
        raise ValueError(f"Expected reference JSON to contain an object, got {type(data).__name__}")
    data_dict = cast(dict[str, object], data)
    books = data_dict.get("books")
    if not isinstance(books, list):
        raise ValueError("Reference JSON must contain a 'books' list")

    losses: dict[str, float] = {}
    for item in cast(list[object], books):
        if not isinstance(item, dict):
            raise ValueError("Reference JSON 'books' entries must be objects")
        item_dict = cast(dict[str, object], item)
        book = item_dict.get("book")
        loss = item_dict.get("loss")
        if not isinstance(book, str) or not isinstance(loss, (int, float)):
            raise ValueError("Reference JSON book entries need string 'book' and numeric 'loss'")
        losses[book] = float(loss)
    return losses


def reference_losses_for_books(
    references: dict[str, float],
    val_books: list[tuple[str, bytes]],
) -> list[float] | None:
    losses: list[float] = []
    for name, _ in val_books:
        loss = references.get(name)
        if loss is None:
            return None
        losses.append(loss)
    return losses


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="N-gram baselines")
    parser.add_argument("--train", default=DEFAULT_TRAIN)
    parser.add_argument("--max-n", type=int, default=5)
    parser.add_argument(
        "--reference-json",
        type=Path,
        default=None,
        help="Optional eval_per_book JSON to use as transformer reference",
    )
    parser.add_argument("--out", type=Path, default=None, help="Optional JSON output path")
    args = parser.parse_args(argv)
    require_positive(parser, "--max-n", args.max_n)
    return args


def main() -> None:
    args = parse_args()

    train_path = Path(args.train)
    if not train_path.exists():
        print(f"ERROR: training corpus not found: {train_path}")
        sys.exit(1)

    train_data = train_path.read_bytes()
    print(f"Training corpus: {train_path}  ({len(train_data):,} bytes)\n")

    # Load available validation books
    val_books: list[tuple[str, bytes]] = []
    for book in RESEARCH_BOOKS:
        name = book.name
        p = book.validation_path
        if p.exists():
            val_books.append((name, p.read_bytes()))
        else:
            print(f"  Skipping {name} -- {p} not found")

    if not val_books:
        print("No validation files found.")
        sys.exit(1)

    transformer_losses = load_transformer_references(args.reference_json)

    # Header
    col = 14
    print(f"  {'Model':<{col}}", end="")
    for name, _ in val_books:
        short = name.split()[0][:8]
        print(f"  {short:>8}", end="")
    print(f"  {'Average':>8}")
    print("  " + "-" * (col + 10 * len(val_books) + 10))

    # Transformer reference row, if available for every loaded book.
    t_avg: float | None = None
    t_losses = reference_losses_for_books(transformer_losses, val_books)
    if t_losses is not None:
        t_avg = sum(t_losses) / len(t_losses)
        print(f"  {'Transformer':<{col}}", end="")
        for loss in t_losses:
            print(f"  {loss:>8.4f}", end="")
        print(f"  {t_avg:>8.4f}  <- reference")
        print()
    else:
        print("  Transformer reference skipped; missing losses for one or more books.")
        print()

    # N-gram rows
    transformer_reference: dict[str, object] | None = (
        {
            "losses": t_losses,
            "average": t_avg,
            "source": str(args.reference_json) if args.reference_json else "built-in",
        }
        if t_losses is not None and t_avg is not None
        else None
    )
    model_rows: list[NgramResult] = []

    for n in range(1, args.max_n + 1):
        model = NgramModel(n)
        model.train(train_data)

        losses: list[float] = []
        per_book: list[NgramBookLoss] = []
        for name, val_data in val_books:
            loss = model.cross_entropy(val_data)
            losses.append(loss)
            per_book.append(NgramBookLoss(book=name, loss=loss))

        avg = sum(losses) / len(losses)
        marker = (
            "<- transformer wins" if t_avg is not None and avg > t_avg
            else "<- n-gram wins" if t_avg is not None
            else ""
        )
        model_rows.append(
            NgramResult(
                n=n,
                average_loss=avg,
                per_book=per_book,
                comparison=marker.removeprefix("<- ") if marker else None,
            )
        )

        print(f"  {n}-gram{'':<{col - 6}}", end="")
        for loss in losses:
            print(f"  {loss:>8.4f}", end="")
        print(f"  {avg:>8.4f}  {marker}")

    print()
    print("All losses in nats. Lower is better.")
    if t_avg is not None:
        print("Transformer reference: lower is better; source recorded in JSON output.")
    if args.out is not None:
        output = NgramReport(
            train=str(train_path),
            train_bytes=len(train_data),
            transformer_reference=transformer_reference,
            models=model_rows,
        )
        write_json(args.out, output.to_json_dict())


if __name__ == "__main__":
    main()
