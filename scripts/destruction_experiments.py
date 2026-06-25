"""Structure-destruction experiments for tinyllm.

Each experiment corrupts the validation text in a specific way and measures
how much the loss increases relative to the baseline.  The gap reveals which
kinds of structure the model has actually learned.

Experiments
-----------
baseline            Raw validation text, no modification.
context_N           Restrict effective context to N bytes (N in 1,2,4,8,16,32,64,128).
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
import random
import re
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

# ---------------------------------------------------------------------------
# Books
# ---------------------------------------------------------------------------

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

# Character names to replace, grouped by book.
# Replacements cycle through a fixed list of neutral placeholders.
AUSTEN_NAMES: dict[str, list[str]] = {
    "Pride and Prejudice": [
        "Elizabeth", "Darcy", "Bennet", "Bingley", "Jane", "Lydia",
        "Wickham", "Collins", "Charlotte", "Kitty", "Mary", "Longbourn",
    ],
    "Sense and Sensibility": [
        "Elinor", "Marianne", "Dashwood", "Willoughby", "Brandon", "Edward",
        "Ferrars", "Jennings", "Palmer", "Middleton",
    ],
    "Emma": [
        "Emma", "Knightley", "Woodhouse", "Weston", "Churchill", "Fairfax",
        "Elton", "Harriet", "Smith", "Bates",
    ],
    "Mansfield Park": [
        "Fanny", "Edmund", "Crawford", "Bertram", "Norris", "Price",
        "Rushworth", "Yates", "Grant",
    ],
    "Persuasion": [
        "Anne", "Wentworth", "Elliot", "Musgrove", "Benwick", "Harville",
        "Smith", "Russell", "Walter",
    ],
    "Northanger Abbey": [
        "Catherine", "Tilney", "Morland", "Thorpe", "Isabella", "Henry",
        "Allen", "Woodston",
    ],
}
PLACEHOLDERS = ["Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta", "Eta", "Theta"]


# ---------------------------------------------------------------------------
# Corruption functions  (bytes in → bytes out)
# ---------------------------------------------------------------------------

def corrupt_shuffle_letters(text: bytes) -> bytes:
    """Shuffle all characters within each word."""
    decoded = text.decode("utf-8", errors="replace")
    def _shuffle(m: re.Match[str]) -> str:
        chars = list(m.group(0))
        random.shuffle(chars)
        return "".join(chars)
    return re.sub(r"[A-Za-z]+", _shuffle, decoded).encode("utf-8", errors="replace")


def corrupt_shuffle_middle(text: bytes) -> bytes:
    """Shuffle only the middle characters of each word; keep first and last."""
    decoded = text.decode("utf-8", errors="replace")
    def _shuffle_middle(m: re.Match[str]) -> str:
        word = m.group(0)
        if len(word) <= 3:
            return word
        middle = list(word[1:-1])
        random.shuffle(middle)
        return word[0] + "".join(middle) + word[-1]
    return re.sub(r"[A-Za-z]+", _shuffle_middle, decoded).encode("utf-8", errors="replace")


def corrupt_shuffle_words(text: bytes) -> bytes:
    """Shuffle word order within each sentence."""
    decoded = text.decode("utf-8", errors="replace")
    sentences = re.split(r"(?<=[.!?])\s+", decoded)
    result = []
    for sentence in sentences:
        words = sentence.split(" ")
        random.shuffle(words)
        result.append(" ".join(words))
    return " ".join(result).encode("utf-8", errors="replace")


def corrupt_reverse(text: bytes) -> bytes:
    """Reverse the entire byte sequence."""
    return text[::-1]


def corrupt_random_letters(text: bytes) -> bytes:
    """Replace every ASCII letter with a random letter a-z."""
    result = bytearray(text)
    for i, b in enumerate(result):
        if 65 <= b <= 90 or 97 <= b <= 122:
            result[i] = random.randint(97, 122)
    return bytes(result)


def make_corrupt_replace_names(book_name: str) -> Callable[[bytes], bytes]:
    """Return a corruption function that replaces character names for a given book."""
    names = AUSTEN_NAMES.get(book_name, [])

    def _corrupt(text: bytes) -> bytes:
        if not names:
            return text
        decoded = text.decode("utf-8", errors="replace")
        for i, name in enumerate(names):
            placeholder = PLACEHOLDERS[i % len(PLACEHOLDERS)]
            decoded = re.sub(r"\b" + re.escape(name) + r"\b", placeholder, decoded)
        return decoded.encode("utf-8", errors="replace")

    return _corrupt


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def load_tokens(path: Path) -> torch.Tensor:
    return torch.tensor(bytearray(path.read_bytes()), dtype=torch.long)


def fresh_generator(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


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
    """Estimate loss using only the last context_len bytes of each block."""
    model.eval()
    block_size = model_cfg.blockSize
    device = train_cfg.device
    losses: list[float] = []
    g = fresh_generator(seed)

    with torch.no_grad():
        for _ in range(train_cfg.evalIters):
            high = tokens.size(0) - block_size
            idx = torch.randint(0, high, (train_cfg.batchSize,), generator=g)
            offsets = torch.arange(block_size)
            positions = idx.unsqueeze(1) + offsets.unsqueeze(0)
            block_x = tokens[positions].to(device)
            block_y = tokens[positions + 1].to(device)
            if context_len < block_size:
                block_x[:, :block_size - context_len] = 0
            logits, _, _ = model(block_x)
            logits_slice = logits[:, -context_len:, :]
            targets_slice = block_y[:, -context_len:]
            loss = F.cross_entropy(
                logits_slice.reshape(-1, logits_slice.size(-1)),
                targets_slice.reshape(-1),
            )
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

CONTEXT_SIZES = [1, 2, 4, 8, 16, 32, 64, 128]


def main() -> None:
    parser = argparse.ArgumentParser(description="Structure-destruction experiments")
    parser.add_argument("--checkpoint", default="runs/austen-byte/checkpoints/best.pt")
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--no-per-book", action="store_true",
        help="Skip per-book breakdown; report only aggregate results",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    device = args.device if torch.cuda.is_available() else "cpu"

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
    for book_name, val_path in BOOKS:
        path = Path(val_path)
        if not path.exists():
            print(f"  Skipping {book_name} — {val_path} not found")
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
    for book_name, _, tokens in book_data:
        loss = estimate_loss(model, tokens, model_cfg, train_cfg, args.seed)
        perp = torch.exp(torch.tensor(loss)).item()
        baselines[book_name] = loss
        print(f"  {book_name:<26}  {loss:>8.4f}  {perp:>10.2f}")
    avg_base = sum(baselines.values()) / len(baselines)
    print("  " + "-" * 50)
    print(f"  {'Average':<26}  {avg_base:>8.4f}")

    # --- Context window probe ---
    print("\n" + "=" * 72)
    print("CONTEXT WINDOW PROBE")
    print("=" * 72)
    print(f"  {'Context':<10}  {'Avg Loss':>8}  {'Delta':>7}  {'Delta%':>7}  {'Marginal':>9}")
    print("  " + "-" * 55)
    prev_avg: float | None = None
    for ctx in CONTEXT_SIZES:
        ctx_losses = []
        for book_name, _, tokens in book_data:
            loss = estimate_loss_context(model, tokens, model_cfg, train_cfg, args.seed, ctx)
            ctx_losses.append(loss)
        avg = sum(ctx_losses) / len(ctx_losses)
        delta = avg - avg_base
        pct = 100.0 * delta / avg_base
        marginal = f"{avg - prev_avg:+.4f}" if prev_avg is not None else "       -"
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
            corrupted_bytes = corrupt_fn(raw, book_name)
            corrupted_tokens = torch.tensor(bytearray(corrupted_bytes), dtype=torch.long)
            corrupted_loss = estimate_loss(
                model, corrupted_tokens, model_cfg, train_cfg, args.seed
            )
            book_results.append((book_name, baselines[book_name], corrupted_loss))

        if args.no_per_book:
            avg_b = sum(b for _, b, _ in book_results) / len(book_results)
            avg_c = sum(c for _, _, c in book_results) / len(book_results)
            delta = avg_c - avg_b
            pct = 100.0 * delta / avg_b
            marker = fmt_delta(delta)
            print(f"{marker} {exp_name:<26}  {avg_b:>8.4f}  {avg_c:>9.4f}  {delta:>+7.4f}  {pct:>+6.1f}%")
        else:
            print_experiment_results(exp_name, book_results)


if __name__ == "__main__":
    main()
