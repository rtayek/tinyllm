# tinyllm Handoff

Last verified: June 23, 2026

## Project

Tiny byte-level GPT implemented in PyTorch.

```text
C:/Users/ray/work/tiny
```

Environment:

```sh
conda activate tinyllm
```

The package uses a `src/` layout and exposes three CLI entry points:

```text
tinyllm-train
tinyllm-infer
tinyllm-prepare-corpora
```

## Current Status

Verification:

```text
pytest: 54 collected, 53 passed (1 CUDA skip: test_checkpoint_roundtrip)
pyright: 0 errors
branch coverage: 80.1%
```

Default checkpoint:

```text
runs/sherlock-byte-default/checkpoints/best.pt
step: 4500
best validation loss: 1.7905686581134796
corpus: canonical Sherlock training split (8 stories train / 2 val / 2 test)
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
src/llm/EarlyStopping.py   patience-based early stopping, state_dict round-trip
src/llm/LRScheduleStrategy.py  warmup + cosine LR schedule
src/llm/Checkpoint.py      full training checkpoint representation and manager
src/llm/RunArtifacts.py    run.json metadata and metrics.jsonl writer
src/llm/TextGenerator.py   prompt encoding and byte-to-text decoding
src/llm/infer.py           inference CLI and checkpoint-based model construction
src/llm/Main.py            training CLI and trainer construction
src/llm/persistence.py     model-only export/load utility (no test coverage)
src/llm/plot_utils.py      training curve plotting
src/llm/tensor_utils.py    device helpers (also contains unused distributed stubs)
```

Research findings about what the current checkpoint has learned, including
n-gram baselines, causal ablations, attention measurements, and recommended
experiments, are documented in `LEARNED_STRUCTURE.md`. Note that those
measurements were taken against an older checkpoint on the legacy corpus;
they have not yet been reproduced against the current canonical split.

The vocabulary is the 256 possible byte values. Prompts are encoded with
UTF-8. `generateBytes()` is lossless; generated text uses UTF-8 replacement
characters for invalid sequences by default, with an explicit error-policy
override available to callers.

### Tokenizer Direction

The current model uses byte-level tokens: each token is one UTF-8 byte. This
keeps the implementation simple, lossless, and independent of a trained
tokenizer, but it produces longer sequences than a subword tokenizer and limits
how much text fits in the 128-token context window.

Keep byte tokens while the project is focused on transformer mechanics and
representation research. The recommended scaling path is:

1. Add more training data (more Gutenberg corpora).
2. Scale up model capacity (more layers, wider embeddings, longer blockSize).
3. Add BPE tokenization as a controlled comparison — not a replacement.

If BPE is added, a reasonable starting point is 2,000–8,000 tokens and a
context length of 256–512. Changing tokenization is a model-format break:
existing checkpoints are incompatible, training must restart, and checkpoints
should store tokenizer metadata that inference validates on load.

## Config Validation

Both `ModelConfig` and `TrainConfig` validate their fields in `__post_init__`:

`ModelConfig` rejects:
- `blockSize < 1`
- `vocabSize < 1`

`TrainConfig` rejects:
- `batchSize < 1`
- `learningRate <= 0`
- `warmupFrac` outside `[0, 1]`
- `evalIters < 1`

These checks fire at construction time, including when restoring from a
checkpoint dict via `fromDict`.

## Checkpoints

Normal training and inference use one authoritative default file:

```text
runs/sherlock-byte-default/checkpoints/best.pt
```

Training also writes `latest.pt` at every evaluation and periodic
`step-NNNNNN.pt` snapshots. Resume prefers `latest.pt`, while inference keeps
using `best.pt` unless `--checkpoint` selects another file. Snapshots default
to every 1,000 steps with the newest three retained.

`best.pt` tracks every new absolute validation-loss minimum.
`earlyStopDelta` is separate and only determines whether patience resets.

After training, `best.pt` is loaded and evaluated on deterministically sampled
held-out test batches. That result is appended to `metrics.jsonl` and is not
used for model selection.

Each checkpoint contains:

- Model weights
- Optimizer state
- Model and training configuration
- Training step and best validation loss
- Learning-rate scheduler state
- Batch generator RNG state
- Evaluator generator RNG state
- Early-stopping counter and reference loss

Checkpoint saves are atomic at the filesystem level: data is written to a
temporary file in the checkpoint directory and then installed with
`os.replace()`. If serialization fails, the previous checkpoint remains
untouched and the temporary file is removed.

Inference reads the saved `ModelConfig`, constructs the matching model, and
loads `modelState`. A separate model-only file is not used automatically.

Model-only export remains available as an explicit utility:

```sh
python -m llm.persistence export-model \
  --ckpt runs/sherlock-byte-default/checkpoints/best.pt \
  --out models/exported_model.pt
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
sh train.sh
```

`train.sh` uses the canonical Sherlock splits and clears its prior checkpoints,
metrics, plots, and samples under `runs/sherlock-byte-default/`. Override
`RUN_DIR` to isolate another experiment.

Run-specific checkpoints under `runs/<experiment>/checkpoints/` place plots
and generated samples under the same run directory. Training writes `run.json`
with configuration and corpus hashes, and `metrics.jsonl` with validation and
final test results.

Training flags:

```text
--corpus PATH
--validation-corpus PATH
--test-corpus PATH
--checkpoint PATH
--seed INTEGER
--early-stop-patience COUNT
--reset-early-stopping
--snapshot-interval STEPS
--max-snapshots COUNT
--plot
--log-level DEBUG|INFO|WARNING|ERROR
```

CUDA is preferred by the default configuration. Training and inference both
fall back to CPU when CUDA is unavailable.

An exhausted early-stopping checkpoint is treated as a completed run instead
of repeating its final evaluation on every launch. Continue explicitly with:

```sh
tinyllm-train --reset-early-stopping --early-stop-patience 5
```

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
--checkpoint PATH
--prompt TEXT
--tokens COUNT
--temperature FLOAT
--top-k COUNT
--seed INTEGER
```

Defaults:

```text
checkpoint: runs/sherlock-byte-default/checkpoints/best.pt
prompt: ""
tokens: 400
temperature: 1.0
top-k: unrestricted
seed: ambient random state
```

An explicit seed uses a local PyTorch generator and produces repeatable output.
Inference does not currently expose `--log-level`; KV-cache rebuild debug
messages are only visible through programmatic logging.

## KV Cache

KV-cache positional handling is implemented and tested.

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

`ModelConfig.use_cache` is `False` in both defaults and the saved checkpoint.
Set it to `True` in a model configuration to use cached decoding.

## Data

Original sources and normalized outputs are organized under
`corpora/<author>/<work>/`. Rebuild manifests, clean text, logical units, and
splits with:

```sh
tinyllm-prepare-corpora
```

Each work contains `raw/`, `clean/`, `units/`, `splits/`, and `manifest.json`.
The current collection contains three public-domain fiction works:

```text
corpora/arthur-conan-doyle/adventures-of-sherlock-holmes/
corpora/lewis-carroll/alices-adventures-in-wonderland/
corpora/jane-austen/pride-and-prejudice/
```

New experiment outputs belong under `runs/<experiment>/`, while selected
reusable model artifacts belong under `models/`.

### getBatch requires an explicit generator

`SequenceDataModule.getBatch` requires a `torch.Generator` argument and raises
`ValueError` if `None` is passed. There is no internal default fallback.
All callers — `Trainer`, `Evaluator`, and tests — must supply a generator.

## Training Reliability

- `Evaluator` is the sole owner of the `EarlyStopping` instance.
- Checkpoint resume restores the evaluator's early-stopping counter and
  reference loss from the saved state.
- Checkpoints use temporary-file plus atomic-replacement writes.
- Training keeps separate best, latest, and retained periodic snapshots.
- Training curve figures are closed after saving to avoid accumulation.
- NumPy is declared as a runtime dependency in both `pyproject.toml` and
  `requirements.txt`.

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

## Known Issues and Technical Debt

These are confirmed issues from code review. None are blocking but all should
be addressed before the next major feature addition.

**Design:**

- `LRScheduleStrategy.load_state_dict` manually sets `scheduler._step_count`,
  a PyTorch private attribute, to keep internal counters aligned after resume.
  This will break silently if PyTorch changes its scheduler internals. The
  `align_after_resume` fallback path (calling `step()` N times) is safer and
  should replace the private-field approach.

- `tensor_utils.py` imports `torch.distributed` and `numpy` unconditionally
  and defines `seed_everything`, `get_master_process`, `get_num_gpus`, and
  `get_ddp_free_model` — none of which are used anywhere in the codebase.
  These are dead code from an earlier distributed design and should be removed.

- `persistence.py` has zero test coverage. Its `load-model` subcommand
  distinguishes a full checkpoint from a model-only file by checking for a
  `"modelState"` key — a heuristic that would misfire on a custom model whose
  `state_dict` happens to contain that key.

- `__init__.py` exports `ByteDataModule` but not `TokenDataModule` or
  `SequenceDataModule`, even though `TokenDataModule` is what the default
  training path uses. The public API should export all three, or the
  asymmetry should be documented as intentional.

**Style:**

- A commented-out `logSample` call remains at the bottom of `Main.main()`.
  Remove it or replace with a `--log-sample` flag.

- `plot_utils.py` uses `vars(modelConfig)` to dump config, but
  `modelConfig.toDict()` already exists for this purpose. Use the dedicated
  method for consistency.

- No `fail_under` coverage threshold is enforced. Consider adding one to
  `pyproject.toml` once `persistence.py` is covered.

## Naming Convention

Production code uses `camelCase` for methods and attributes. PyTorch-protocol
methods (`state_dict`, `load_state_dict`, `step`) use `snake_case` to match
PyTorch's own conventions. Test functions use `snake_case` throughout. Avoid
broad renaming unless handled as a deliberate refactor.

## Roadmap

Planned work in priority order:

1. **More training data** — add more Project Gutenberg corpora to increase
   total training text beyond the current ~600 KB across three works.
2. **Scale the model** — increase `nLayer`, `nEmbed`, and `blockSize` once
   there is data worth training on.
3. **Fix known issues** — address the `try/finally`, dead code, and
   `persistence.py` coverage gaps listed above.
4. **BPE tokenization** — add as a controlled comparison path after scaling,
   not as a replacement for byte tokens.
5. **RL / self-improvement** — reward-signal experiments once the base model
   generates coherent text.
