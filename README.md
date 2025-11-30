# Tiny LLM Project (Byte-Level GPT in PyTorch)

Tiny byte-level GPT in PyTorch with checkpointing (`src/Checkpoint.py`) and a simple trainer/inference flow.

## Quickstart
1) Ensure training text exists (defaults to `fixtureData/input.txt`; you can generate a fresh corpus with `src/make_tender_buttons_dataset.py` which writes `data/input.txt`).
2) Train: `python Main.py`
3) Infer (after training writes checkpoints): `python src/infer.py`

Checkpoints live under `checkpoints/` by default. The trainer saves a plot to `plots/` without opening a GUI window.
