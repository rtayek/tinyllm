# Tiny LLM Project (Byte-Level GPT in PyTorch)

This is a tiny byte-level GPT model in PyTorch.

## Files

- `Config.py` — ModelConfig with fields like `vocabSize`, `blockSize`, `maxSteps`.
- `Model.py` — TinyGpt model with multi-head self-attention (`MultiHeadSelfAttention`, `Block`, `TinyGpt`).
- `DataModule.py` — ByteDataModule that loads raw bytes, splits train/val, and produces batches via `getBatch`.
- `Checkpoints.py` — CheckpointManager for saving/loading model + optimizer state.
- `Trainer.py` — Trainer class with `train`, `estimateLoss`, `plotTrainingCurve`, `printSample`.
- `Main.py` — Entry point wiring everything together.
- `runTinyLlm.sh` — Startup script (no -u).

## How to Run

```sh
python Main.py
```

You will also need a `data` directory with an `input.txt`:

```text
  data/
    input.txt
```

`input.txt` can be any text you want the model to learn from.
