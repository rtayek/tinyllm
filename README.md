# Tiny LLM Project (Byte-Level GPT in PyTorch)

Tiny byte-level GPT in PyTorch packaged under `llm/` with checkpointing, training, and inference scripts.

## Install
Create/activate your env and install the package (editable for dev):
```sh
pip install -e .
```
Runtime deps: `torch`, `matplotlib`. Dev extras in `pyproject.toml` (`pytest`, `pyright`).

## Data
Training text defaults to `fixtureData/input.txt`. You can fetch a sample corpus:
```sh
python src/make_tender_buttons_dataset.py
```
This writes `data/input.txt`; point the config to your desired file.

## Train
```sh
tinyllm-train
```
Checkpoints are written under `checkpoints/`. Plots go to `plots/` (no blocking GUI).

## Infer
After training (or with a saved checkpoint):
```sh
tinyllm-infer
```
This loads the latest checkpoint and prints generated text.

Pass a prompt and optionally choose how many new byte tokens to generate:
```sh
sh infer.sh --prompt "Mr. Sherlock Holmes"
sh infer.sh --prompt "To Sherlock Holmes she is always the woman." --tokens 200
```

Running `sh infer.sh` without a prompt preserves unconditional generation.

## Imports
Library components are under `llm`, e.g.:
```python
from llm import RunConfig, TinyGpt, Trainer, ByteDataModule
```
