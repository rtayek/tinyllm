# Design Review — tinyllm

Snapshot date: 2026-06-30  
Scope: `src/llm`, `src/llm/research`, command entry points, training regression, evaluation semantics, and near-term refactoring opportunities.

This is a structural and architectural review, not a line-level bug hunt.

## Current State

The project is in a healthy design state. The current repository has crossed an important threshold: the remaining risks are now mostly research-correctness, evaluation-accounting, and future-scaling risks rather than general messiness or unclear ownership.

Confirmed current state:

```text
✓ safe cleanup / design-stabilization committed
✓ training regression tool committed
✓ package-canonical command policy committed
✓ smoke training regression passed
✓ all-corpus training regression passed across 9 configured corpora
✓ unit/integration-style pytest suite passes
✓ pyright reported clean by local verification before this review
```

Sandbox verification during this review:

```text
python -m pytest -q
230 passed in 8.32s
```

`pyright` was not installed in the sandbox environment used for this review, but the project state provided before this review reported:

```text
python -m pyright -> 0 errors
```

The all-corpus training regression completed over these corpora:

```text
arthur-conan-doyle/adventures-of-sherlock-holmes
lewis-carroll/alices-adventures-in-wonderland
jane-austen/pride-and-prejudice
jane-austen/sense-and-sensibility
jane-austen/emma
jane-austen/mansfield-park
jane-austen/persuasion
jane-austen/northanger-abbey
jane-austen/combined
```

This means corpus paths, split loading, CPU training, checkpoint writing, run metadata writing, metrics writing, and final test evaluation all succeeded through the regression path.

## Executive Assessment

The project now has a coherent architecture:

```text
src/llm/
  canonical reusable implementation:
  model, transformer, data modules, training, evaluation, inference,
  corpus preparation, checkpointing, run artifacts, regression tooling

src/llm/research/
  learned-structure experiments and baselines:
  per-book evaluation, corruption experiments, n-gram baselines,
  report-oriented research utilities

scripts/
  compatibility wrappers and developer diagnostics only
```

The most important dependency rule is still intact:

```text
research imports core
core does not import research
```

That rule should be protected. It is the difference between a research codebase and a research swamp wearing a lab coat.

## Command and Entry-Point Architecture

The package-canonical command policy is now settled:

```text
Implementation canonical:
    src/llm/...

Developer canonical:
    python -m llm...

User/package canonical:
    tinyllm-*

scripts/:
    compatibility wrappers and developer utilities only
```

Current package console scripts include:

```text
tinyllm-train
tinyllm-infer
tinyllm-prepare-corpora
tinyllm-eval-per-book
tinyllm-destruction-experiments
tinyllm-ngram-baseline
tinyllm-training-regression
```

For day-to-day development, the recommended working style remains:

```bash
python -m llm.train_app
python -m llm.infer
python -m llm.training_regression --profile smoke
python -m llm.training_regression --profile all
python -m llm.research.eval_per_book
python -m llm.research.ngram_baseline
python -m llm.research.destruction_experiments
```

The `tinyllm-*` commands are installed-package conveniences. They are good for README examples and end-user command surfaces, but they do not need to be the primary mental model during development.

## Layering and Dependencies

The design has clean strata:

```text
Config / json_utils / serialization_types / tensor_utils
        |
Transformer -> Model
        |
DataModule / EarlyStopping / LRSchedule / OptimizerFactory
        |
Evaluator / EvalResult / EvaluationMode / EvaluationProbe
        |
Trainer / TrainingCallback / RunArtifacts / Checkpoint
        |
train_app / train_cli / infer / corpus / training_regression
        |
research/*
```

Two architectural wins matter most:

1. `src/llm/research` is a consumer of the core package, not something the core package knows about.
2. Research measurements are increasingly represented as structured values rather than loose printed numbers.

The project is still small enough that the current mostly-flat `src/llm` module layout is acceptable. A broad move to subpackages such as `llm.training`, `llm.evaluation`, and `llm.generation` would mostly create import churn now. Do not reorganize folders for ceremonial cleanliness.

## Evaluation Subsystem

The evaluation subsystem is one of the strongest parts of the current design.

Important pieces:

```text
Evaluator.py
  computes sampled and deterministic full-split losses

EvalResult.py
  stores structured measurement records: loss, perplexity, nTokens,
  nWindows, method, corpus, checkpoint, notes

EvaluationProbe.py
  defines EvalContext and EvaluationProbe protocol

EvaluationMode.py
  provides concrete probe strategies:
  SampledLossEvaluator
  FullSplitEvaluator
  PerBookEvaluator
  CorruptionEvaluator
  BaselineEvaluator
```

This is effectively the Strategy pattern plus a Parameter Object and a structured Result Value Object.

### Full-Split Evaluation Semantics

The most important recent correctness improvement is that full-split evaluation semantics are now pinned down by tests.

The current full-pass evaluator distinguishes:

```text
full_nonoverlap
  stride == blockSize
  evaluates complete non-overlapping windows

full_stride
  stride < blockSize
  evaluates overlapping windows while scoring only newly exposed target positions
```

The intended current behavior is:

```text
nWindows
  number of evaluated block-sized windows

nTokens
  number of scored target tokens, not raw split length

overlapping windows
  do not double-count target positions

trailing bytes
  up to blockSize - 1 bytes may be omitted if they cannot close a complete window
```

The tests in `tests/unit/test_evaluator.py` now cover:

```text
determinism
batch-size invariance
non-overlapping full-pass accounting
stride=1 accounting
non-dividing stride accounting
no duplicate scored targets under overlap
method labels: full_nonoverlap and full_stride
invalid stride rejection
too-small split rejection
training-mode restoration
early-stopping state isolation
```

This is exactly the right kind of correctness work for a learned-structure project. Loss numbers are only useful if the denominator is what the code claims it is. Otherwise we are just making decimals do theater.

### Remaining Evaluation Design Notes

`EvalContext` currently carries several optional fields:

```text
name
split
tokens
raw
corpus
checkpoint
notes
```

Different probes require different subsets:

```text
PerBookEvaluator       requires tokens
CorruptionEvaluator    requires raw
SampledLossEvaluator   requires split/evaluator state
BaselineEvaluator      requires externally supplied loss and optional nTokens
```

This is acceptable for now. The current approach uses validation at evaluator boundaries, for example raising a clear `ValueError` when `tokens` or `raw` is absent.

Do not split `EvalContext` yet. Typed contexts such as `TokenEvalContext` or `RawEvalContext` may become useful later, but right now they would add more names and ceremony than value.

Recommended near-term improvement: keep the existing single `EvalContext`, but document field requirements in evaluator docstrings where needed.

## Training Regression

`src/llm/training_regression.py` is a good system-level safety net.

Its purpose is not to produce useful models. Its purpose is to answer:

```text
Can the configured corpora train from scratch through the real training path?
Can train/validation/test splits load?
Can run directories be created?
Can checkpoints be written?
Can run metadata and metrics be written?
Can best checkpoint test evaluation run?
```

The regression profiles are intentionally small:

```text
profile smoke
  first two corpora

profile all
  all configured corpora plus jane-austen/combined
```

The tiny CPU configuration is appropriate for regression:

```text
vocabSize=256
blockSize=16
nEmbed=32
nHead=2
nLayer=1
batchSize=2
maxSteps=2
cpu
```

### Cleanup Policy

The `--no-clean` option is the right practical response to Windows file-locking behavior.

Current behavior:

```text
default
  clear old regression run directory first

--no-clean
  skip directory deletion before running
```

The deletion helper guards against unsafe paths:

```text
runs/regression/<profile>/<corpus-name>
```

and reports Windows lock problems with an actionable error. Good. This is not glamorous, but neither is losing half an evening to Explorer holding a directory handle hostage.

### Future Split Point

`training_regression.py` currently owns:

```text
profile selection
corpus discovery/planning
config construction
safe cleanup
job execution
artifact assertions
CLI
```

That is acceptable at current size.

If it grows, split later into:

```text
training_regression.py          CLI / orchestration
training_regression_plan.py     profiles, corpus jobs, config construction
training_regression_checks.py   artifact assertions
```

Do not split now. Premature file-splitting is how projects acquire lots of small places to hide bugs.

## Generation Seam

`AutoregressiveGenerator` now exposes useful generation APIs:

```text
generateBytes
generateText
generateCandidate
generateCandidates
```

`GeneratedCandidate` is the right value object for future controlled self-improvement:

```text
prompt
continuation
text
tokenIds
promptTokenCount
generatedTokenCount
seed
temperature
topK
maxNewTokens
```

The continuation is derived from token boundaries:

```text
continuationBytes = tokenIds[promptTokenCount:]
```

That is correct for a byte-level model. String-prefix slicing would be wrong in edge cases because decoded text is not the authoritative representation.

### Generation Refactoring Opportunity

`AutoregressiveGenerator.__init__` still has a compatibility wart:

```python
def __init__(
    self,
    model,
    logger_or_device=None,
    logger=None,
):
```

The device is now derived from the model. Once no caller still passes a device string, simplify this to:

```python
def __init__(
    self,
    model,
    logger=None,
):
```

This is low priority. Keep compatibility until the call sites are known to be clean.

### Batched Candidate Generation

`generateCandidates()` currently loops over `generateCandidate()` and increments the seed:

```text
seed, seed + 1, seed + 2, ...
```

That is deterministic and good enough for Phase 1.

True tensor-batched generation may be useful later for best-of-N candidate generation, but it is not urgent. Keep the public API stable first; optimize later if profiling says it matters.

## Checkpointing and Run Artifacts

Checkpointing is correctly treated as a Memento pattern.

The checkpoint state includes the information needed to resume training meaningfully:

```text
model state
optimizer state
scheduler state
training step
best validation loss
model config
train config
batch RNG state
evaluator RNG state
early-stopping state
```

This is important because a checkpoint is not merely a tensor dump. It is an experimental state.

`RunArtifacts` gives the project durable research output surfaces:

```text
run.json
metrics.jsonl
continuations.jsonl
```

This is the right direction. Research results should accumulate as structured records with provenance, not as terminal output folklore.

## Config and Construction

`ModelConfig` and `TrainConfig` remain solid frozen dataclasses with validation.

`TrainConfig` still mixes several concerns:

```text
optimization settings
runtime/device settings
data paths
checkpoint/run artifact paths
evaluation cadence
```

This is not conceptually pure, but it is acceptable because `TrainConfig` is also a serialization and checkpoint boundary.

Do not split `TrainConfig` right now.

The previous design tension has already been reduced by:

```text
RunPaths
  groups path-related fields as a derived view

OptimizerFactory / SchedulerFactory
  allows optimizer/scheduler construction to be injected without rewriting Trainer
```

That is a good compromise: conceptual grouping without destabilizing checkpoint compatibility.

## Trainer Orchestration

`LMTrainer` remains readable and appropriately orchestrates:

```text
training loop
sampled evaluation
checkpoint save/restore
metrics callbacks
plot control
final test evaluation
```

The callback system is small and useful:

```text
on_eval
on_train_end
```

One possible future consistency improvement remains:

```python
on_test_end(result: EvalResult) -> None
```

Currently, final test metric writing is a one-shot post-training path rather than fully routed through the callback system. This is defensible. Add `on_test_end` only if another consumer needs it.

## Design Patterns Already Present

The code is already using useful patterns without becoming pattern theater.

| Pattern | Current use | Assessment |
|---|---|---|
| Strategy | `EvaluationMode`, corruption functions, optimizer/scheduler factories | Good fit |
| Factory | data module factory, optimizer/scheduler factories | Good fit |
| Memento | checkpoint save/restore | Essential and appropriate |
| Observer / Callback | `TrainingCallback` | Small and useful |
| Parameter Object | `EvalContext`, `RunPaths`, configs | Useful; watch optional-field sprawl |
| Value Object / DTO | `EvalResult`, `GeneratedCandidate`, configs | Good fit |
| Adapter / Compatibility Shim | `scripts/*.py` wrappers | Correctly contained |
| Facade-ish CLI layer | `train_app`, `infer`, `training_regression` | Good enough |

The project should not get a broad design-pattern refactor. The current design is already patterned where patterns help.

## Readiness for Controlled Self-Improvement

Do not start self-improvement yet. The current staged plan is still correct:

```text
1. design stabilization
2. bug/correctness cleanup
3. real-LLM feature-gap review
4. preliminary scaling
5. clean-code review
6. controlled self-improvement
```

That said, the future seams are now visible.

Likely future abstractions:

```python
class RewardFn(Protocol):
    def score(self, candidate: GeneratedCandidate) -> float: ...

@dataclass(frozen=True)
class ScoredCandidate:
    candidate: GeneratedCandidate
    score: float
    rewardName: str

class CandidateSelector(Protocol):
    def select(
        self,
        candidates: list[ScoredCandidate],
        k: int,
    ) -> list[ScoredCandidate]: ...
```

These are probably right later, but they should not be added until the project actually begins the self-improvement phase.

Do not add unused architecture fossils. They look grand at first and then sit there quietly confusing everyone.

## Refactoring Opportunities

### 1. Document `EvalContext` field requirements

Current priority: low-to-medium.

Add concise docstrings or comments explaining which probes require which fields:

```text
PerBookEvaluator requires context.tokens.
CorruptionEvaluator requires context.raw.
BaselineEvaluator uses externally supplied loss and optional nTokens.
```

Do not split the type yet.

### 2. Simplify `AutoregressiveGenerator.__init__` later

Current priority: low.

Remove `logger_or_device` only after confirming all call sites no longer pass a device string.

### 3. Keep an eye on `training_regression.py` size

Current priority: low.

Only split if it grows significantly or gains more profiles/checks.

### 4. Consider a future `on_test_end` callback

Current priority: low.

Useful if final test metrics need the same callback extensibility as eval metrics.

### 5. Review diagnostic utility duplication

Current priority: low.

There may be overlap among:

```text
check_cuda.py
scripts/check_gpu.py
scripts/checkGPU.py
```

Eventually keep one canonical diagnostic path and convert/remove wrappers as appropriate.

### 6. Avoid broad package reorganization

Current priority: explicit non-goal.

Do not move modules into new subpackages right now. The current shape is understandable and the tests pass. Folder shuffling would be mostly architectural confetti.

## Suggested Next Work

### Immediate next task: clean-code review

Now that evaluation token accounting is covered, the next good task is a non-invasive clean-code review.

Scope should be:

```text
find duplicate logic
find stale wrappers
find unclear names
find stale docs
find modules exceeding their responsibility
find dead compatibility paths
```

Constraints should be:

```text
do not change behavior
do not reorganize packages
do not change training math
do not change evaluation semantics
do not start self-improvement
```

### After clean-code review: real-LLM feature-gap review

Potential feature gaps to analyze later:

```text
tokenizer abstraction beyond raw bytes
dataset streaming / memory mapping
larger checkpoint policy
mixed precision
gradient accumulation
batched generation
experiment registry
scaling curves
structured comparison reports
```

### After feature-gap review: preliminary scaling

Scaling experiments should be boring and reproducible:

```text
fixed corpora
fixed seeds
fixed evaluation methods
fixed run artifact schema
clearly labeled blockSize / nEmbed / nLayer grids
regression smoke before and after larger runs
```

## Recommended Priority Table

| Priority | Item | Type | Do now? |
|---|---|---|---|
| 1 | Preserve core/research dependency direction | Architectural invariant | Yes, always |
| 2 | Keep full-split evaluation tests green | Research correctness | Yes |
| 3 | Document `EvalContext` field requirements | Small design cleanup | Soon |
| 4 | Non-invasive clean-code review | Maintenance | Next |
| 5 | Feature-gap review for a real LLM someday | Planning | After cleanup |
| 6 | Preliminary scaling grid | Research engineering | After feature-gap review |
| 7 | Simplify `AutoregressiveGenerator.__init__` | Cleanup | Later |
| 8 | Add `on_test_end` callback | Callback consistency | Later |
| 9 | Add `RewardFn` / `ScoredCandidate` / `CandidateSelector` | Self-improvement seam | Not yet |

## Bottom Line

The architecture is now good enough that the next work should be small, careful, and correctness-oriented.

The major wins are in place:

```text
package-canonical command policy
clean core/research layering
structured evaluation records
EvaluationProbe protocol
full-split evaluation accounting tests
GeneratedCandidate value object
training regression suite
checkpoint memento
run artifact provenance
callback-based training metrics
```

The project should not receive a large architectural reorganization right now.

The next best move is a clean-code review with strict constraints, followed by a feature-gap review for scaling toward a more realistic LLM. Self-improvement should wait until those layers are boringly reliable. Boring reliability is underrated, probably because it does not make dramatic conference slides.
