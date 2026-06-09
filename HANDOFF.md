# tinyllm Handoff

Last verified: June 8, 2026

## Project

Tiny byte-level GPT implemented in PyTorch.

```text
C:/Users/ray/work/tiny
```

Environment:

```sh
conda activate tinyllm
```

The package uses a `src/` layout and exposes:

```text
tinyllm-train
tinyllm-infer
tinyllm-prepare-corpora
```

## Current Status

The maintenance changes described below and this handoff file are currently
uncommitted.

Verification:

```text
pytest: 43 passed
pyright: 0 errors
branch coverage: 80.0%
```

Current checkpoint:

```text
checkpoints/tiny_llm.pt
step: 3100
best validation loss: 2.0223001837730408
corpus: removed legacy `fixtureData/sherlock.txt`
```

Model configuration stored in the checkpoint:

```text
vocabSize: 256
blockSize: 128
nEmbed: 256
nHead: 4
nLayer: 4
dropout: 0.2
use_cache: false
```

## Architecture

Important modules:

```text
src/llm/Config.py          ModelConfig, TrainConfig, RunConfig
src/llm/DataModule.py      UTF-8 byte tokenization and batch sampling
src/llm/Transformer.py     decoder blocks, attention, and KV cache
src/llm/Model.py           language model, loss, and generation sampling
src/llm/Trainer.py         training loop and checkpoint resume
src/llm/Evaluator.py       train/validation loss estimation
src/llm/Checkpoint.py      full training checkpoint representation and manager
src/llm/TextGenerator.py   prompt encoding and byte-to-text decoding
src/llm/infer.py           inference CLI and checkpoint-based model construction
src/llm/Main.py            training CLI and trainer construction
```

Research findings about what the current checkpoint has learned, including
n-gram baselines, causal ablations, attention measurements, and recommended
experiments, are documented in `LEARNED_STRUCTURE.md`.

The vocabulary is the 256 possible byte values. Prompts are encoded with
UTF-8. `generateBytes()` is lossless; generated text uses UTF-8 replacement
characters for invalid sequences by default, with an explicit error-policy
override available to callers.

### Tokenizer Direction

The current model already uses tokens: each token is one byte. This keeps the
implementation simple, lossless, and independent of a trained tokenizer, but
it produces longer sequences than a subword tokenizer and limits how much text
fits in the 128-token context window.

Keep byte tokens while the project is focused on transformer mechanics. If the
goal shifts toward better generated language and more efficient context usage,
the recommended next step is byte-level BPE rather than word-level tokens. A
reasonable starting point would be a vocabulary of 2,000-8,000 tokens and a
context length of 256-512 tokens, with tied input/output embeddings to limit
parameter growth.

Changing tokenization is a model-format change. Existing checkpoints would not
be compatible, training would need to restart, and checkpoints should store
tokenizer metadata that inference validates before loading the model.

## Checkpoints

Normal training and inference use one authoritative file:

```text
checkpoints/tiny_llm.pt
```

It contains:

- Model weights
- Optimizer state
- Model and training configuration
- Training step and best validation loss
- Learning-rate scheduler state
- Batch generator state

Checkpoint saves are atomic at the filesystem level: data is written to a
temporary file in the checkpoint directory and then installed with
`os.replace()`. If serialization fails, the previous checkpoint remains
untouched and the temporary file is removed.

Inference reads the saved `ModelConfig`, constructs the matching model, and
loads `modelState`. A separate model-only file is not used automatically.

Model-only export remains available as an explicit utility:

```sh
python -m llm.persistence export-model \
  --ckpt checkpoints/tiny_llm.pt \
  --out checkpoints/exported_model.pt
```

## Training

Resume training from the existing checkpoint:

```sh
tinyllm-train
```

Select another corpus and its held-out splits:

```sh
tinyllm-train \
  --corpus corpora/lewis-carroll/alices-adventures-in-wonderland/splits/train.txt \
  --validation-corpus corpora/lewis-carroll/alices-adventures-in-wonderland/splits/validation.txt \
  --test-corpus corpora/lewis-carroll/alices-adventures-in-wonderland/splits/test.txt
```

Start a clean run:

```sh
sh run.sh
```

Important: `run.sh` intentionally deletes files under `checkpoints/` and
`plots/` before training.

Training sample output creates `tmp/` when needed, so a fresh clone can write
`tmp/sample.txt` even though the directory is ignored by Git.

Training flags:

```text
--corpus PATH
--validation-corpus PATH
--test-corpus PATH
--checkpoint PATH
--plot
--log-level DEBUG|INFO|WARNING|ERROR
```

CUDA is preferred by the default configuration. Training and inference both
fall back to CPU when CUDA is unavailable.

## Inference

Basic prompt generation:

```sh
tinyllm-infer \
  --prompt "Mr. Sherlock Holmes" \
  --tokens 400
```

Controlled and reproducible sampling:

```sh
tinyllm-infer \
  --prompt "Mr. Sherlock Holmes" \
  --tokens 400 \
  --temperature 0.8 \
  --top-k 50 \
  --seed 123
```

Inference flags:

```text
--prompt TEXT
--tokens COUNT
--temperature FLOAT
--top-k COUNT
--seed INTEGER
```

Defaults preserve the original behavior:

```text
prompt: ""
tokens: 400
temperature: 1.0
top-k: unrestricted
seed: ambient random state
```

An explicit seed uses a local PyTorch generator and produces repeatable output.

## KV Cache

KV-cache positional handling was fixed and pinned with tests.

The implementation:

- Offsets learned positional embeddings by the cached sequence length.
- Rejects cache overflow rather than silently shifting position-encoded keys.
- Rebuilds the cache from the current context when `blockSize` is reached.
- Emits a `DEBUG` log when a rebuild occurs.

Relevant tests:

```text
test_cached_logits_match_uncached_logits
test_cached_generation_sanity
test_cached_generation_matches_uncached_across_context_boundary
test_cached_generation_logs_context_rebuild
```

Run only cache tests:

```sh
pytest tests/unit/test_core.py -k cached -v
```

`ModelConfig.use_cache` is currently `False` in both defaults and the saved
checkpoint. Set it to `True` in a model configuration to use cached decoding.

## Data

Original sources and normalized outputs are organized under
`corpora/<author>/<work>/`. Rebuild manifests, clean text, logical units, and
splits with:

```sh
tinyllm-prepare-corpora
```

Each work contains `raw/`, `clean/`, `units/`, `splits/`, and `manifest.json`.
The current normal-fiction collection contains Sherlock Holmes, Alice's
Adventures in Wonderland, and Jane Austen's Pride and Prejudice.

```text
corpora/arthur-conan-doyle/adventures-of-sherlock-holmes/
corpora/lewis-carroll/alices-adventures-in-wonderland/
corpora/jane-austen/pride-and-prejudice/
```

The historical checkpoint was trained on the removed `fixtureData/sherlock.txt`
file. Treat it as a historical model; start a fresh run for canonical data.
New experiment outputs belong under `runs/<experiment>/`, while selected
reusable model artifacts belong under `models/`.

The DataModule sampling boundary was fixed. A split containing exactly
`blockSize + 1` bytes now produces its single valid training window.
Batch windows are assembled with vectorized tensor indexing rather than a
Python loop.

## Training Reliability

- `Evaluator` is the sole owner of the `EarlyStopping` instance.
- Checkpoint resume resets the evaluator's actual early-stopping state.
- Checkpoints use temporary-file plus atomic-replacement writes.
- Training curve figures are closed after saving to avoid figure accumulation.
- NumPy is declared in both `pyproject.toml` and `requirements.txt`.

## Tests

Run everything:

```sh
pytest
pyright
```

Run branch coverage:

```sh
pytest --cov --cov-report=term-missing
```

Coverage measures `src/llm` with branch coverage enabled. Standalone
diagnostic/data-preparation scripts are omitted from the project percentage:

```text
DataBottleneckProfiler.py
checkGPU.py
make_tender_buttons_dataset.py
```

Generate a browsable report:

```sh
pytest --cov --cov-report=html
```

Useful focused commands:

```sh
pytest -v
pytest tests/unit/test_core.py
pytest tests/unit/test_core.py -k cached -v
pytest tests/unit/test_infer.py
```

## Recent Completed Work

- Added prompt-based inference.
- Added `--tokens`, `--temperature`, `--top-k`, and `--seed`.
- Added deterministic seeded sampling.
- Fixed DataModule's final-window off-by-one error.
- Unified CUDA-to-CPU fallback for training and inference.
- Fixed KV-cache positional correctness and context-boundary rebuilding.
- Simplified checkpoint handling to one normal checkpoint path.
- Made checkpoint model configuration authoritative during inference.
- Added atomic checkpoint writes that preserve the previous file on failure.
- Removed duplicate early-stopping ownership from `LMTrainer`.
- Made sample output create its ignored `tmp/` directory.
- Declared NumPy as a runtime dependency.
- Closed matplotlib figures after saving training plots.
- Vectorized DataModule batch assembly.
- Made invalid generated UTF-8 bytes visible with replacement characters.
- Added `pytest-cov` and branch-coverage configuration.
- Updated README and expanded unit/integration coverage.

## Remaining Considerations

- Generated text is still mostly locally English-like rather than coherent.
  The model is small and the Sherlock corpus is approximately 600 KB.
- Sampling controls may improve local quality but cannot replace more data,
  capacity, or training.
- Inference does not currently expose `--log-level`, so the KV-cache rebuild
  debug entry is mainly visible through programmatic logging.
- `--plot` is redundant while `TrainConfig.plotCurve` defaults to `True`.
- `src/llm/persistence.py` currently has no direct coverage and is the clearest
  next target for improving the 80.4% branch-coverage baseline.
- Coverage is reported but no `fail_under` threshold is enforced yet.
- The codebase mixes camelCase and snake_case naming. Avoid broad renaming
  unless it is handled as a deliberate refactor.
