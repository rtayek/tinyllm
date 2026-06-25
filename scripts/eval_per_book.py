"""Evaluate best.pt against each book's validation split individually.

Usage:
    python scripts/eval_per_book.py
    python scripts/eval_per_book.py --checkpoint runs/austen-byte/checkpoints/best.pt
    python scripts/eval_per_book.py --iters 200
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch

from llm.Checkpoint import Checkpoint
from llm.Config import ModelConfig, TrainConfig
from llm.DataModule import SequenceDataModule
from llm.Evaluator import Evaluator
from llm.EarlyStopping import EarlyStopping
from llm.Model import TinyGPTLanguageModel

BOOKS = [
    ("Pride and Prejudice",   "corpora/jane-austen/pride-and-prejudice/splits/validation.txt"),
    ("Sense and Sensibility", "corpora/jane-austen/sense-and-sensibility/splits/validation.txt"),
    ("Emma",                  "corpora/jane-austen/emma/splits/validation.txt"),
    ("Mansfield Park",        "corpora/jane-austen/mansfield-park/splits/validation.txt"),
    ("Persuasion",            "corpora/jane-austen/persuasion/splits/validation.txt"),
    ("Northanger Abbey",      "corpora/jane-austen/northanger-abbey/splits/validation.txt"),
    ("Sherlock Holmes",       "corpora/arthur-conan-doyle/adventures-of-sherlock-holmes/splits/validation.txt"),
    ("Alice in Wonderland",   "corpora/lewis-carroll/alices-adventures-in-wonderland/splits/validation.txt"),
]


def load_tokens(path: Path) -> torch.Tensor:
    return torch.tensor(bytearray(path.read_bytes()), dtype=torch.long)


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-book validation loss")
    parser.add_argument(
        "--checkpoint",
        default="runs/austen-byte/checkpoints/best.pt",
    )
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}\n")

    checkpoint = Checkpoint.load(args.checkpoint, device)
    if checkpoint.modelConfig is None:
        print("ERROR: checkpoint has no modelConfig")
        return

    model_cfg = ModelConfig.fromDict(checkpoint.modelConfig)
    train_cfg = TrainConfig(device=device, evalIters=args.iters)
    model = TinyGPTLanguageModel(model_cfg).to(device)
    model.load_state_dict(checkpoint.modelState)
    model.eval()

    generator = torch.Generator()
    generator.manual_seed(42)

    print(f"  {'Book':<26}  {'Val Loss':>8}  {'Perplexity':>10}")
    print("  " + "-" * 50)

    results: list[tuple[str, float]] = []
    for name, val_path in BOOKS:
        path = Path(val_path)
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

        loss = evaluator.estimate_split("val", generator)
        perplexity = torch.exp(torch.tensor(loss)).item()
        results.append((name, loss))
        print(f"  {name:<26}  {loss:>8.4f}  {perplexity:>10.2f}")

    if results:
        avg = sum(l for _, l in results) / len(results)
        print("  " + "-" * 50)
        print(f"  {'Average':<26}  {avg:>8.4f}")


if __name__ == "__main__":
    main()
