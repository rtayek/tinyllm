"""Evaluate best.pt against each book's validation split individually.

Usage:
    python scripts/eval_per_book.py
    python scripts/eval_per_book.py --checkpoint runs/austen-byte/checkpoints/best.pt
    python scripts/eval_per_book.py --iters 200
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch

from llm.Checkpoint import Checkpoint
from llm.Config import ModelConfig, TrainConfig
from llm.DataModule import SequenceDataModule
from llm.Evaluator import Evaluator
from llm.EarlyStopping import EarlyStopping
from llm.Model import TinyGPTLanguageModel
from llm.research_books import RESEARCH_BOOKS
from llm.tensor_utils import resolve_device


def load_tokens(path: Path) -> torch.Tensor:
    return torch.tensor(bytearray(path.read_bytes()), dtype=torch.long)


def book_seed(seed: int, book_name: str) -> int:
    digest = hashlib.sha256(f"{seed}:{book_name}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big") % (2**63)


def book_generator(seed: int, book_name: str) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(book_seed(seed, book_name))
    return generator


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
    args = parser.parse_args(argv)
    if args.iters < 1:
        parser.error("--iters must be greater than zero")
    return args


def main() -> None:
    args = parse_args()

    device = resolve_device(args.device)
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}\n")

    checkpoint = Checkpoint.load(args.checkpoint, device)
    if not checkpoint.modelConfig:
        print("ERROR: checkpoint has no modelConfig")
        return

    model_cfg = ModelConfig.fromDict(checkpoint.modelConfig)
    train_cfg = TrainConfig(device=device, evalIters=args.iters)
    model = TinyGPTLanguageModel(model_cfg).to(device)
    model.load_state_dict(checkpoint.modelState)
    model.eval()

    print(f"  {'Book':<26}  {'Val Loss':>8}  {'Perplexity':>10}")
    print("  " + "-" * 50)

    results: list[tuple[str, float]] = []
    output: dict[str, object] = {
        "checkpoint": args.checkpoint,
        "device": device,
        "iters": args.iters,
        "seed": args.seed,
        "books": [],
    }
    for book in RESEARCH_BOOKS:
        name = book.name
        path = book.validation_path
        if not path.exists():
            print(f"  {name:<26}  {'(not found)':>8}")
            continue

        val_tokens = load_tokens(path)
        # SequenceDataModule needs a train sequence too; reuse val as a dummy
        # since we only call estimate_split("val").
        data_module = SequenceDataModule(
            model_cfg, train_cfg,
            sequence=val_tokens,
            validationSequence=val_tokens,
        )
        evaluator = Evaluator(
            model, data_module, train_cfg, EarlyStopping(patience=1, delta=0.0)
        )

        generator = book_generator(args.seed, name)
        loss = evaluator.estimate_split("val", generator)
        perplexity = torch.exp(torch.tensor(loss)).item()
        results.append((name, loss))
        books = output["books"]
        assert isinstance(books, list)
        books.append(
            {
                "book": name,
                "path": str(path),
                "seed": book_seed(args.seed, name),
                "loss": loss,
                "perplexity": perplexity,
            }
        )
        print(f"  {name:<26}  {loss:>8.4f}  {perplexity:>10.2f}")

    if results:
        avg = sum(l for _, l in results) / len(results)
        output["average_loss"] = avg
        print("  " + "-" * 50)
        print(f"  {'Average':<26}  {avg:>8.4f}")
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
