"""Evaluate best.pt against each book's validation split individually.

Usage:
    python scripts/eval_per_book.py
    python scripts/eval_per_book.py --checkpoint runs/austen-byte/checkpoints/best.pt
    python scripts/eval_per_book.py --iters 200
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from llm.cli_utils import require_positive
from llm.json_utils import write_json
from llm.research import research_eval as _research_eval
from llm.research_reports import BookLossRow, PerBookReport
from llm.research.research_eval import (
    book_seed,
    estimate_validation_result,
    load_checkpoint_model,
    load_research_book_data,
)
from llm.tensor_utils import resolve_device

book_generator = _research_eval.book_generator


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
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help=(
            "With --full-split, window stride. Defaults to blockSize "
            "(non-overlapping); use 1 for maximal-context sliding-window "
            "perplexity, or blockSize//2 for the common compromise."
        ),
    )
    args = parser.parse_args(argv)
    require_positive(parser, "--iters", args.iters)
    if args.stride is not None:
        require_positive(parser, "--stride", args.stride)
        if not args.full_split:
            parser.error("--stride only applies with --full-split")
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
        result = estimate_validation_result(
            checkpoint_model.model,
            book.tokens,
            checkpoint_model.model_config,
            checkpoint_model.train_config,
            args.seed,
            book.name,
            full_split=args.full_split,
            stride=args.stride,
            checkpoint=args.checkpoint,
            corpus=str(book.path),
        )
        rows.append(
            BookLossRow(
                book=book.name,
                path=str(book.path),
                seed=book_seed(args.seed, book.name),
                loss=result.loss,
                perplexity=result.perplexity,
            )
        )
        print(f"  {book.name:<26}  {result.loss:>8.4f}  {result.perplexity:>10.2f}")

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
        write_json(args.out, report.to_json_dict())


if __name__ == "__main__":
    main()
