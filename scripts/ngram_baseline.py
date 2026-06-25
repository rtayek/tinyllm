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
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

BOOKS: list[tuple[str, str]] = [
    ("Pride and Prejudice",   "corpora/jane-austen/pride-and-prejudice/splits/validation.txt"),
    ("Sense and Sensibility", "corpora/jane-austen/sense-and-sensibility/splits/validation.txt"),
    ("Emma",                  "corpora/jane-austen/emma/splits/validation.txt"),
    ("Mansfield Park",        "corpora/jane-austen/mansfield-park/splits/validation.txt"),
    ("Persuasion",            "corpora/jane-austen/persuasion/splits/validation.txt"),
    ("Northanger Abbey",      "corpora/jane-austen/northanger-abbey/splits/validation.txt"),
    ("Sherlock Holmes",       "corpora/arthur-conan-doyle/adventures-of-sherlock-holmes/splits/validation.txt"),
    ("Alice in Wonderland",   "corpora/lewis-carroll/alices-adventures-in-wonderland/splits/validation.txt"),
]

DEFAULT_TRAIN = "corpora/jane-austen/combined/splits/train.txt"
VOCAB_SIZE = 256

# Transformer val losses from step 9500 run (seed 42, 200 iters)
TRANSFORMER_LOSSES = [1.4305, 1.4724, 1.4220, 1.4774, 1.4312, 1.4811, 1.6764, 1.7941]


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
        for i in range(len(data) - n):
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


def main() -> None:
    parser = argparse.ArgumentParser(description="N-gram baselines")
    parser.add_argument("--train", default=DEFAULT_TRAIN)
    parser.add_argument("--max-n", type=int, default=5)
    args = parser.parse_args()

    train_path = Path(args.train)
    if not train_path.exists():
        print(f"ERROR: training corpus not found: {train_path}")
        sys.exit(1)

    train_data = train_path.read_bytes()
    print(f"Training corpus: {train_path}  ({len(train_data):,} bytes)\n")

    # Load available validation books
    val_books: list[tuple[str, bytes]] = []
    for name, path in BOOKS:
        p = Path(path)
        if p.exists():
            val_books.append((name, p.read_bytes()))
        else:
            print(f"  Skipping {name} -- {path} not found")

    if not val_books:
        print("No validation files found.")
        sys.exit(1)

    # Header
    col = 14
    print(f"  {'Model':<{col}}", end="")
    for name, _ in val_books:
        short = name.split()[0][:8]
        print(f"  {short:>8}", end="")
    print(f"  {'Average':>8}")
    print("  " + "-" * (col + 10 * len(val_books) + 10))

    # Transformer reference row
    t_losses = TRANSFORMER_LOSSES[:len(val_books)]
    t_avg = sum(t_losses) / len(t_losses)
    print(f"  {'Transformer':<{col}}", end="")
    for loss in t_losses:
        print(f"  {loss:>8.4f}", end="")
    print(f"  {t_avg:>8.4f}  <- step 9500")
    print()

    # N-gram rows
    for n in range(1, args.max_n + 1):
        model = NgramModel(n)
        model.train(train_data)

        losses: list[float] = []
        for _, val_data in val_books:
            losses.append(model.cross_entropy(val_data))

        avg = sum(losses) / len(losses)
        marker = "<- transformer wins" if avg > t_avg else "<- n-gram wins"

        print(f"  {n}-gram{'':<{col - 6}}", end="")
        for loss in losses:
            print(f"  {loss:>8.4f}", end="")
        print(f"  {avg:>8.4f}  {marker}")

    print()
    print("All losses in nats. Lower is better.")
    print("Transformer: step 9500, combined Austen corpus, seed 42, 200 eval iters.")


if __name__ == "__main__":
    main()
