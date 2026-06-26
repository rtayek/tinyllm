"""Evaluate best.pt against each book's validation split individually.

Usage:
    python scripts/eval_per_book.py
    python scripts/eval_per_book.py --checkpoint runs/austen-byte/checkpoints/best.pt
    python scripts/eval_per_book.py --iters 200
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import torch

from llm.cli_utils import require_positive
from llm import research_eval as _research_eval
from llm.research_eval import (
    book_seed,
    estimate_validation_loss,
    load_checkpoint_model,
    load_research_book_data,
)
from llm.tensor_utils import resolve_device

book_generator = _research_eval.book_generator


@dataclass(frozen=True)
class BookLossRow:
    book: str
    path: str
    seed: int
    loss: float
    perplexity: float


@dataclass(frozen=True)
class PerBookReport:
    checkpoint: str
    device: str
    iters: int
    full_split: bool
    seed: int
    books: list[BookLossRow]
    average_loss: float | None

    def to_json_dict(self) -> dict[str, object]:
        data = asdict(self)
        if self.average_loss is None:
            data.pop("average_loss")
        return data


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Per-book validation loss")
    parser.add_argument(
        "--checkpoint",
        default="runs/austen-byte/checkpoints/best.pt",
    )
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, default=None, help="Optional JSON output path")
    parser.add_argument(
        "--full-split",
        action="store_true",
        help="Evaluate every valid window instead of sampled batches",
    )
    args = parser.parse_args(argv)
    require_positive(parser, "--iters", args.iters)
    return args


def build_report(args: argparse.Namespace, device: str) -> PerBookReport | None:
    try:
        checkpoint_model = load_checkpoint_model(args.checkpoint, device, args.iters)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return None

    print(f"  {'Book':<26}  {'Val Loss':>8}  {'Perplexity':>10}")
    print("  " + "-" * 50)

    rows: list[BookLossRow] = []
    books, missing = load_research_book_data()
    for name, _path in missing:
        print(f"  {name:<26}  {'(not found)':>8}")

    for book in books:
        loss = estimate_validation_loss(
            checkpoint_model.model,
            book.tokens,
            checkpoint_model.model_config,
            checkpoint_model.train_config,
            args.seed,
            book.name,
            full_split=args.full_split,
        )
        perplexity = torch.exp(torch.tensor(loss)).item()
        rows.append(
            BookLossRow(
                book=book.name,
                path=str(book.path),
                seed=book_seed(args.seed, book.name),
                loss=loss,
                perplexity=perplexity,
            )
        )
        print(f"  {book.name:<26}  {loss:>8.4f}  {perplexity:>10.2f}")

    average_loss = None
    if rows:
        average_loss = sum(row.loss for row in rows) / len(rows)
        print("  " + "-" * 50)
        print(f"  {'Average':<26}  {average_loss:>8.4f}")

    return PerBookReport(
        checkpoint=args.checkpoint,
        device=device,
        iters=args.iters,
        full_split=args.full_split,
        seed=args.seed,
        books=rows,
        average_loss=average_loss,
    )


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}\n")

    report = build_report(args, device)
    if report is not None and args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(report.to_json_dict(), indent=2) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
