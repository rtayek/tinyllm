# tinyllm Handoff

Last verified: June 27, 2026

## Project

Tiny byte-level GPT implemented in PyTorch.

```text
C:/Users/ray/work/tiny
```

Environment:

```sh
conda activate tinyllm
```

The package uses a `src/` layout and exposes these CLI entry points:

```text
tinyllm-train
tinyllm-infer
tinyllm-prepare-corpora
```

## Current Status

Verification:

```text
pyright: 0 errors
pytest: 174 passed
coverage: 88.56%
```

The current checkpoint layout still uses the canonical `runs/` directory.
Checkpoint compatibility depends on the saved model architecture matching the
requested run configuration.

## Architecture

`src/llm/` contains the reusable byte-level GPT implementation and the
training/inference/corpus-preparation code. `src/llm/research/` contains the
learned-structure experiments and baselines that use that reusable core.

Important modules:

```text
src/llm/Config.py             ModelConfig, TrainConfig, RunConfig
src/llm/DataModule.py         UTF-8 byte tokenization and batch sampling
src/llm/Transformer.py        decoder blocks, attention, and KV cache
src/llm/Model.py              language model, loss, and generation sampling
src/llm/Trainer.py            training loop and checkpoint resume
src/llm/Evaluator.py          train/validation loss estimation
src/llm/EvalResult.py         structured loss/perplexity evaluation records
src/llm/EvaluationMode.py     first-class evaluation mode wrappers
src/llm/EarlyStopping.py      patience-based early stopping, state_dict round-trip
src/llm/LRScheduleStrategy.py warmup + cosine LR schedule
src/llm/Checkpoint.py         full training checkpoint and resume logic
src/llm/RunArtifacts.py       run.json metadata and metrics.jsonl writer
src/llm/TrainingCallback.py   training callbacks
src/llm/TextGenerator.py      prompt encoding and byte-to-text decoding
src/llm/infer.py              inference CLI
src/llm/train_app.py          training CLI and callback wiring
src/llm/persistence.py        model-only export/load utility
src/llm/plot_utils.py         training curve plotting
src/llm/tensor_utils.py       device helpers
src/llm/corpus.py             corpus preparation pipeline
src/llm/corpus_sources.py     corpus cleaning and logical-unit specs
src/llm/research/eval_per_book.py              per-book validation evaluation
src/llm/research/destruction_experiments.py    corruption/destruction probes
src/llm/research/ngram_baseline.py             byte n-gram baselines
src/llm/research/research_eval.py              shared research eval helpers
```

Research findings, including n-gram baselines and destruction experiments, are
documented in `LEARNED_STRUCTURE.md`. Those measurements were taken on older
checkpoints and should be treated as historical until reproduced on the
current corpus.

The vocabulary is the 256 possible byte values. Prompts are encoded with
UTF-8. `generateBytes()` is lossless; generated text uses UTF-8 replacement
characters for invalid sequences by default, with an explicit error-policy
override available to callers.

## Callbacks

The training loop fires events through a `TrainingCallback` protocol with two
hooks:

```python
on_eval(result: EvalResult, is_best: bool) -> None
on_train_end(curve: list[tuple[int, float, float]]) -> None
```

`buildTrainer` in `train_app.py` registers four callbacks in order:

```text
LoggingCallback        eval log lines and end-of-run summary
MetricsCallback        metrics.jsonl writes via RunArtifacts
CheckpointCallback     best.pt / latest.pt / snapshot saves
TrainingCurveCallback  training curve plot at end of run
```

`CheckpointCallback` now consumes a narrower checkpoint context instead of a
full trainer back-reference, but it still closes over trainer-owned methods and
state through that context.

## Config

`ModelConfig` and `TrainConfig` validate their fields in `__post_init__`.

`TrainConfig.toDict()` now round-trips with `TrainConfig.fromDict()`.
Unknown training fields are ignored when restoring from dictionaries, so older
code can tolerate newer metadata fields while preserving validation for known
fields.
`RunArtifacts` writes a filtered training payload to `run.json` via
`TrainConfig.toRunJsonDict()`, so the stored run metadata keeps corpus paths in
the dedicated `corpora` section.

`RunConfig.fromDict` raises `ValueError` if the `model` or `train` keys are not
dicts.

## Evaluation Results

`llm.EvalResult.EvalResult` is the shared schema for loss/perplexity-style
evaluation records. It stores the evaluation name, split, loss, computed
perplexity, optional token/window counts, method (`sampled`, `full_nonoverlap`,
or `full_stride`), and optional checkpoint/corpus notes. It supports
`toDict()`/`fromDict()` for JSON-ready reports.

`llm.Evaluator.EvalResult` is the older training-loop callback result used for
step, train-loss, validation-loss, and early-stopping state. New research code
should prefer structured `llm.EvalResult.EvalResult` records over ad hoc dicts
or loose floats; the raw-float evaluator APIs remain available for
compatibility.

`EvaluationMode.py` names the currently supported evaluation modes explicitly:
`SampledLossEvaluator`, `FullSplitEvaluator`, `PerBookEvaluator`,
`CorruptionEvaluator`, and `BaselineEvaluator`. These are intentionally thin
wrappers over the existing computation paths so evaluation policy is visible
without changing model behavior or report formats.

## Checkpoints

Normal training and inference use one authoritative default file:

```text
runs/sherlock-byte-default/checkpoints/best.pt
```

Training also writes `latest.pt` at every evaluation and periodic
`step-NNNNNN.pt` snapshots. Resume prefers `latest.pt`, while inference keeps
using `best.pt` unless `--checkpoint` selects another file.

`best.pt` tracks every new absolute validation-loss minimum. When `best.pt` is
written, its early-stopping counter is reset to zero so that resuming from it
starts with a clean patience budget.

On resume, the first evaluation is skipped if it would duplicate the
evaluation already recorded at the resumed step.

`run.json` is written once at the start of a fresh run and never overwritten on
resume, preserving the original `git_commit`, `created_at`, and
`parent_checkpoint` fields.

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
`os.replace()`.

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

`train.sh` uses the canonical Sherlock splits and clears prior checkpoints,
metrics, plots, and samples under `runs/sherlock-byte-default/`. Override
`RUN_DIR` to isolate another experiment.

Run-specific checkpoints under `runs/<experiment>/checkpoints/` place plots
and generated samples under the same run directory.

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

CUDA is preferred by default. Training and inference both fall back to CPU
when CUDA is unavailable.

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

## Data

Original sources and normalized outputs are organized under
`corpora/<author>/<work>/`.

The current collection contains eight public-domain fiction works:

```text
corpora/arthur-conan-doyle/adventures-of-sherlock-holmes/   (12 stories)
corpora/lewis-carroll/alices-adventures-in-wonderland/      (12 chapters)
corpora/jane-austen/pride-and-prejudice/                    (61 chapters)
corpora/jane-austen/sense-and-sensibility/                  (50 chapters)
corpora/jane-austen/emma/                                   (55 chapters)
corpora/jane-austen/mansfield-park/                         (48 chapters)
corpora/jane-austen/persuasion/                             (24 chapters)
corpora/jane-austen/northanger-abbey/                       (31 chapters)
```

`tinyllm-prepare-corpora --download-austen` downloads the five remaining
Austen novels.

When you share a run archive, include metadata and selected analysis artifacts,
not `.pt` checkpoint files or generated local caches. The usual archive
contents are `run.json`, `metrics.jsonl`, notes, and source code. Transfer
selected plots, samples, or checkpoints separately when those artifacts are
needed.

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

Coverage has `fail_under = 80`.

Useful focused commands:

```sh
pytest -v
pytest tests/unit/test_core.py
pytest tests/unit/test_infer.py
pytest tests/unit/test_training_callbacks.py
```

## Known Technical Debt

- `CheckpointManager.loadCheckpoint` remains as a compatibility wrapper; new
  code should call `restoreCheckpoint`.

- `Evaluator.estimate_split_full` drops the trailing up-to-`block_size-1`
  target positions rather than scoring them in a final window. This is fine
  and conventional for the non-overlapping default (`stride == block_size`):
  the dropped tail is <1% of any real split and the result is fair across
  checkpoints. It is *slightly off* for the sliding-window case
  (`stride < block_size`), where the whole point is to score every target
  with good context, yet the final tail still goes unscored. **Before relying
  on any `--stride 1` or `--stride block_size//2` ("publication-quality")
  sliding-window perplexity number, tighten the tail handling**: append a final
  window anchored at `last_start` and compute its `scored` count from the
  actual gap to the previous window. The default-stride numbers in use today
  are unaffected.

## Roadmap

1. Scale capacity, not context.
2. Layer ablations on the current checkpoint.
3. Fix the remaining small cleanup items.
4. Add BPE tokenization as a controlled comparison path.
5. Explore reward-signal experiments later.
