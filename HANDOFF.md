# tinyllm Handoff

Last verified: June 25, 2026

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
pytest: 88 collected, 88 passed
pyright: 0 errors
branch coverage: 85.7%
```

Current checkpoints exist under `runs/`. Checkpoint compatibility still depends
on the saved model architecture matching the requested run configuration.

## Architecture

Important modules:

```text
src/llm/Config.py             ModelConfig, TrainConfig, RunConfig
src/llm/DataModule.py         UTF-8 byte tokenization and batch sampling
src/llm/Transformer.py        decoder blocks, attention, and KV cache
src/llm/Model.py              language model, loss, and generation sampling
src/llm/Trainer.py            training loop and checkpoint resume
src/llm/Evaluator.py          train/validation loss estimation
src/llm/EarlyStopping.py      patience-based early stopping, state_dict round-trip
src/llm/LRScheduleStrategy.py warmup + cosine LR schedule
src/llm/Checkpoint.py         full training checkpoint, CheckpointLoadResult, CheckpointManager
src/llm/RunArtifacts.py       run.json metadata and metrics.jsonl writer
src/llm/TrainingCallback.py   TrainingCallback protocol and concrete implementations
src/llm/TextGenerator.py      prompt encoding and byte-to-text decoding
src/llm/infer.py              inference CLI and checkpoint-based model construction
src/llm/Main.py               training CLI, buildTrainer, callback wiring
src/llm/persistence.py        model-only export/load utility
src/llm/plot_utils.py         training curve plotting
src/llm/tensor_utils.py       device helpers
src/llm/corpus.py             corpus pipeline for all supported works
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

### MLP State Dict Keys

The MLP submodules inside `DecoderBlock` are now named:

```text
mlp.fc1.weight   mlp.fc1.bias
mlp.fc2.weight   mlp.fc2.bias
```

Checkpoints saved before this change used positional keys (`mlp.0.weight`,
`mlp.2.weight`) and are incompatible. Any existing checkpoint must be
discarded and the model retrained from scratch.

### Training Callbacks

The training loop fires events through a `TrainingCallback` protocol with two
hooks:

```python
on_eval(result: EvalResult, is_best: bool) -> None
on_train_end(curve: list[tuple[int, float, float]]) -> None
```

`buildTrainer` in `Main.py` registers four callbacks in order:

```text
LoggingCallback        â€” eval log lines and end-of-run summary
MetricsCallback        â€” metrics.jsonl writes via RunArtifacts
CheckpointCallback     â€” best.pt / latest.pt / snapshot saves
TrainingCurveCallback  â€” training curve plot at end of run
```

Tests that construct `LMTrainer` directly and need checkpoints to be saved
must register `CheckpointCallback` explicitly. Tests that only check
`trainingCurve` or `bestValLoss` need no callbacks.

### Tokenizer Direction

The current model uses byte-level tokens: each token is one UTF-8 byte. This
keeps the implementation simple, lossless, and independent of a trained
tokenizer, but it produces longer sequences than a subword tokenizer and limits
how much text fits in the 128-token context window.

Keep byte tokens while the project is focused on transformer mechanics and
representation research. The recommended scaling path is:

1. Add more training data (more Gutenberg corpora).
2. Scale up model capacity (more layers, wider embeddings). Note: the corrected
   context probe shows the current model uses only ~8 bytes of context, so a
   longer blockSize is not expected to help until the model is large enough to
   use the window it already has.
3. Add BPE tokenization as a controlled comparison - not a replacement.

If BPE is added, a reasonable starting point is 2,000â€“8,000 tokens and a
context length of 256â€“512. Changing tokenization is a model-format break:
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

`RunConfig.fromDict` raises `ValueError` if the `model` or `train` keys are
not dicts (previously it silently fell back to defaults).

## Checkpoints

Normal training and inference use one authoritative default file:

```text
runs/sherlock-byte-default/checkpoints/best.pt
```

Training also writes `latest.pt` at every evaluation and periodic
`step-NNNNNN.pt` snapshots. Resume prefers `latest.pt`, while inference keeps
using `best.pt` unless `--checkpoint` selects another file. Snapshots default
to every 1,000 steps with the newest three retained.

`best.pt` tracks every new absolute validation-loss minimum. When `best.pt` is
written, its early-stopping counter is reset to zero so that resuming from it
always starts with a clean patience budget. `earlyStopDelta` is separate and
only determines whether patience resets.

On resume, the first evaluation is skipped if it would duplicate the
evaluation already recorded at the resumed step. This prevents duplicate
entries in `metrics.jsonl`.

`run.json` is written once at the start of a fresh run and never overwritten on
resume, preserving the original `git_commit`, `created_at`, and
`parent_checkpoint` fields.

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

`CheckpointManager.loadCheckpoint` returns a `CheckpointLoadResult` dataclass
(replacing the previous 9-tuple) with named fields:

```text
step, bestValLoss, lrStateRestored, version, versionMatches,
configDrift, generatorState, evaluatorGeneratorState, earlyStoppingState
```

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
  --corpus corpora/jane-austen/pride-and-prejudice/splits/train.txt \
  --validation-corpus corpora/jane-austen/pride-and-prejudice/splits/validation.txt \
  --test-corpus corpora/jane-austen/pride-and-prejudice/splits/test.txt \
  --checkpoint runs/pride-byte/checkpoints/best.pt
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
--run-dir PATH
--seed INTEGER
--block-size INTEGER
--n-embed INTEGER
--n-head INTEGER
--n-layer INTEGER
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

Generate 400 new byte tokens from the unconditional start token:

```sh
tinyllm-infer
```

Generate from a prompt:

```sh
tinyllm-infer --prompt "Mr. Sherlock Holmes"
tinyllm-infer --prompt "To Sherlock Holmes she is always the woman." --tokens 200
tinyllm-infer --prompt "Mr. Sherlock Holmes" --temperature 0.8 --top-k 50 --seed 123
tinyllm-infer --checkpoint runs/pride-byte/checkpoints/best.pt --prompt "Elizabeth"
```

Sampling options:

- `--checkpoint`: selects a run checkpoint; defaults to the Sherlock run.
- `--temperature`: controls randomness; lower values favor likely tokens.
- `--top-k`: limits sampling to the K most likely next bytes.
- `--seed`: makes repeated runs reproducible.

The shell wrapper accepts the same arguments:

```sh
sh infer.sh --prompt "Mr. Sherlock Holmes" --tokens 200
```

Output is written directly to `sys.stdout.buffer` with the terminal's own
encoding, substituting `?` for characters the terminal cannot display. This
avoids `UnicodeEncodeError` on Windows cp1252 consoles when the model generates
invalid UTF-8.

Training and inference use CUDA when configured and available. If CUDA is
requested but unavailable, both paths fall back to CPU.

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
The current collection contains eight public-domain fiction works:

```text
corpora/arthur-conan-doyle/adventures-of-sherlock-holmes/   (12 stories)
corpora/lewis-carroll/alices-adventures-in-wonderland/       (12 chapters)
corpora/jane-austen/pride-and-prejudice/                     (61 chapters)
corpora/jane-austen/sense-and-sensibility/                   (50 chapters)
corpora/jane-austen/emma/                                    (55 chapters)
corpora/jane-austen/mansfield-park/                          (48 chapters)
corpora/jane-austen/persuasion/                              (24 chapters)
corpora/jane-austen/northanger-abbey/                        (31 chapters)
```

The five new Austen novels must be downloaded before the pipeline can process
them:

```sh
tinyllm-prepare-corpora --download-austen
```

`--download-pride` downloads Pride and Prejudice if its raw source is missing.
`--download-austen` downloads all five remaining Austen novels.

New experiment outputs belong under `runs/<experiment>/`, while selected
reusable model artifacts belong under `models/`.

### Training on all Austen novels combined

Build the combined splits and manifest before training:

```sh
tinyllm-prepare-corpora --combined-austen
```

Do not use `cat corpora/jane-austen/*/splits/... > combined/...`: once
`combined/` exists, the wildcard can include the output file as an input.

Then pass the combined paths to `tinyllm-train`.

The convenience script is:

```sh
sh train-austen.sh
```

`train-austin.sh` is retained as a compatibility wrapper.

### getBatch requires an explicit generator

`SequenceDataModule.getBatch` requires a `torch.Generator` argument and raises
`ValueError` if `None` is passed. There is no internal default fallback.
All callers â€” `Trainer`, `Evaluator`, and tests â€” must supply a generator.

## Training Reliability

- `Evaluator` is the sole owner of the `EarlyStopping` instance.
- Checkpoint resume restores the evaluator's early-stopping counter and
  reference loss from the saved state.
- Checkpoints use temporary-file plus atomic-replacement writes.
- Training keeps separate best, latest, and retained periodic snapshots.
- Training curve figures are closed after saving to avoid accumulation.
- `run.json` is written once and never overwritten on resume.
- Duplicate evaluation at resume step is suppressed via `_resumedFromStep`.

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

Coverage has `fail_under = 80`.

Research scripts accept `--out` for JSON result files:

```sh
python scripts/eval_per_book.py --out runs/austen-byte/eval-per-book.json
python scripts/ngram_baseline.py --out runs/austen-byte/ngram-baseline.json
python scripts/destruction_experiments.py --out runs/austen-byte/destruction.json
```

Useful focused commands:

```sh
pytest -v
pytest tests/unit/test_core.py
pytest tests/unit/test_core.py -k cached -v
pytest tests/unit/test_infer.py
pytest tests/unit/test_training_callbacks.py
```

## Known Issues and Technical Debt

These are confirmed issues from code review. None are blocking.

**Design:**

- `CheckpointCallback` holds a back-reference to `LMTrainer` (typed as `Any`)
  to access RNG state and the checkpoint manager. The clean fix is a
  `CheckpointContext` value object populated by the trainer at eval time and
  passed into `on_eval`, removing the circular dependency entirely.

- `CheckpointManager.loadCheckpoint` mutates its `model` and `optimizer`
  arguments by calling `load_state_dict` on them. The method name implies it
  returns data; the mutation is a hidden side effect. Rename to
  `restoreCheckpoint`, or split into a data-loading step and a restoration step.

- `TrainConfig.fromDict` is incompatible with `toSerializableDict`: path fields
  stripped by the latter are required by the former. Round-tripping through the
  `run.json` payload would crash. No caller does this today.

- `SequenceDataModule` takes a full `TrainConfig` but only uses `batchSize` and
  `device`. This over-wide dependency makes unit tests verbose.

- `AutoregressiveGenerator` takes a redundant `device: str` parameter; the
  model already knows its device via `next(model.parameters()).device`.

- `tensor_utils.get_device()` is defined but has no callers. Should be removed.

- Naming is inconsistent: `Evaluator` and `EarlyStopping` use `snake_case`
  methods; most other production classes use `camelCase`. The documented
  convention is camelCase for production code, snake_case for PyTorch protocol
  methods only.

- `build_data_module` in `Main.py` uses a string switch on
  `trainConfig.dataModule`. Adding a new data module requires editing this
  function (open/closed violation). A registry dict would be extensible.

## Naming Convention

Production code uses `camelCase` for methods and attributes. PyTorch-protocol
methods (`state_dict`, `load_state_dict`, `step`) use `snake_case` to match
PyTorch's own conventions. Test functions use `snake_case` throughout. Avoid
broad renaming unless handled as a deliberate refactor.

## Roadmap

Planned work in priority order:

1. **Reproduce n-gram baselines on the canonical Austen split** - done; the
   transformer (1.52 avg) beats the best n-gram (4-gram, 1.83 avg) decisively.
   See `LEARNED_STRUCTURE.md`.
2. **Scale capacity, not context** - the corrected context probe shows the
   model uses only ~8 bytes of context and gains nothing from the 128-byte
   window it already has. Increasing `blockSize` is not expected to help at
   this model size. Prioritize `nEmbed` (256 to 512) and `nLayer` (4 to 6-8),
   and re-run the context probe on the larger model to see whether it begins
   to use longer context.
3. **Layer ablations** - reproduce the legacy layer-contribution measurements
   on the current checkpoint to see whether layer 0 MLP still dominates.
4. **Fix known issues** - `CheckpointContext`, `get_device()` removal,
   `AutoregressiveGenerator` device param, naming consistency.
5. **BPE tokenization** - add as a controlled comparison path after scaling,
   not as a replacement for byte tokens.
6. **RL / self-improvement** - reward-signal experiments once the base model
   generates coherent text.

