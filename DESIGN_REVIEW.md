# Design Review — tinyllm

A structural/architectural review (not a line-level bug hunt). Snapshot date:
2026-06-28. Scope: `src/llm` core, `src/llm/research`, config, entry points.

## Summary

The design is strong and notably more disciplined than most research
codebases. Layering is clean and acyclic, the evaluation abstraction is
purpose-built for comparative experiments, and the data layer already contains
the seam the planned self-improvement work needs. The items below are
forward-looking structural observations, not defects.

## Layering and Dependencies (strongest aspect)

The internal dependency graph is acyclic with clear strata:

```
Config / serialization_types / json_utils    (foundation, no internal deps)
        |
Transformer -> Model                          (pure model)
        |
DataModule / EarlyStopping / LRSchedule       (training primitives)
        |
Evaluator -> EvaluationMode                   (evaluation)
        |
Trainer + TrainingCallback                    (orchestration)
        |
train_app / train_cli / infer                 (entry points)
        |
research/*                                    (consumes core; nothing depends back)
```

Two deliberate wins:

- **Core never imports `research/`.** Research tooling is a strict consumer of
  the library, so research scripts can be refactored without touching the model.
- **`Config` sits at the foundation** with only `json_utils` /
  `serialization_types` as dependencies, keeping the most-imported type cheap.

`research/research_eval.py` is now a thin compatibility layer that delegates to
`EvaluationMode` (`PerBookEvaluator`, `book_seed`, `book_generator`,
`evaluator_for_tokens`), which keeps a single source of truth for evaluation
logic and a stable import surface for existing research scripts.

Minor wrinkle: `TrainingCallback` imports `TrainEvalResult` from `Evaluator`,
and `Trainer` imports both — a small diamond, not a cycle. Acceptable; that is
the natural home for the type.

## Evaluation Abstraction (best-designed part)

The `EvaluationMode` hierarchy is five frozen dataclasses
(`SampledLossEvaluator`, `FullSplitEvaluator`, `PerBookEvaluator`,
`CorruptionEvaluator`, `BaselineEvaluator`), each exposing a single
`evaluate()` returning a unified `EvalResult`. They **compose**:
`CorruptionEvaluator` -> `PerBookEvaluator` -> `FullSplitEvaluator` /
`SampledLossEvaluator`. This is exactly the right shape for a project whose
purpose is running many evaluation variants and comparing them. Adding a new
probe (e.g. the planned invariance / structure-preservation probes) means
adding one more dataclass in this family, not touching the trainer or model.

`EvalResult` computing `perplexity` in `__post_init__` and carrying
`method` / `corpus` / `notes` / `checkpoint` provenance makes every measurement
self-describing, which matters when accumulating results across experiments.

Two result types now exist and should stay separate, because they model
different things:

- `TrainEvalResult` (training loop: paired train+val loss, early-stop state)
- `EvalResult` (research: a single labeled, provenance-carrying measurement)

## Data Layer (already has the self-improvement seam)

`SequenceDataModule` accepts **in-memory tensors** (`sequence`,
`validationSequence`); `ByteDataModule` / `TokenDataModule` are file-loading
subclasses on top. This means the "dynamic / generated data path" listed as the
main gap in `SELF_IMPROVEMENT_PLAN.md` is half-built: an expert-iteration loop
can generate candidates, filter them, concatenate into a tensor, and hand it
straight to a `SequenceDataModule` without touching disk.

Remaining gap is narrow: no in-place data update / streaming across iterations,
so each iteration rebuilds the module. For best-of-N that is fine.

`getBatch` requires an explicit generator (raises on `None`). Keep this — it is
a strong reproducibility guarantee and directly serves the "controlled"
requirement.

## Generation Seam

`AutoregressiveGenerator` wraps the model's `generate_autoregressive` with
`temperature` / `topK` / `seed`. For best-of-N it needs one addition: **batched
N-candidate generation** (it currently returns one sequence per call via
`generated[0]`). The `seed` threading already supports deterministic best-of-N.

Cleanup: the constructor's `logger_or_device: Logger | str | None` is a
backward-compat wart; `device` is now derived from the model. Once no caller
passes a device string, drop the union and rename.

## Config Design

`ModelConfig` / `TrainConfig` as frozen dataclasses with thorough
`__post_init__` validation is a solid foundation. `toRunJsonDict` excluding path
fields (so run metadata is machine-independent) and `_trainConfigFromRunJson`
remapping corpora paths on load is a thoughtful round-trip.

Two design notes:

- `runDirectory()` infers the run folder via the string match
  `checkpointDir.name == "checkpoints"` and silently returns `None` otherwise.
  Most brittle line in the config layer; low priority but worth knowing.
  **(Partly addressed 2026-06-28: the logic now lives once in
  `RunPaths.runDirectory()`; `TrainConfig.runDirectory()` delegates to it. The
  string-match fragility itself remains, but there is now a single source of
  truth for it.)**
- `TrainConfig` mixes true hyperparameters (`learningRate`, `batchSize`) with
  environment/IO (`ckptPath`, `dataPath`, `device`). These have different
  lifetimes and audiences. **(Partly addressed 2026-06-28 — see below.)**

### `RunPaths`: path grouping without a serialization split (2026-06-28)

The path/plumbing half of the science-vs-plumbing tension was addressed with a
deliberately conservative choice: a `RunPaths` frozen dataclass that groups the
four path fields (`dataPath`, `validationDataPath`, `testDataPath`, `ckptPath`)
as a **narrowing view** over `TrainConfig`, accessed via `trainConfig.paths()`
and built by `RunPaths.fromTrainConfig`. This mirrors the existing
`DataModuleConfig.fromTrainConfig` pattern.

Why a view rather than physically moving the fields out of `TrainConfig`:

- A blast-radius analysis found 28 files reference `TrainConfig`, but the two
  most-pervasive fields (`device`, 12 files; `seed`, 8 files) do not belong
  cleanly to either a science or a plumbing bucket, so a literal split would not
  decouple most consumers.
- The real risk in a literal split is the serialization layer (`run.json`,
  checkpoints, `fromRunJson` / `toRunJsonDict` / `_trainConfigFromRunJson`),
  which would need a back-compat migration shim and round-trip tests, with a
  genuine correctness risk in checkpoint resume.
- Keeping `TrainConfig` as the serialization boundary and adding `RunPaths` as a
  derived view captures the conceptual win — paths are now a named, grouped,
  reusable concept consumers depend on explicitly (`RunArtifacts`, `train_app`
  preflight) — with **zero serialization-format change and no migration shim**.

This view is also the natural stepping stone if a full `OptimConfig` /
`IOConfig` split is ever wanted: consumers already depend on the grouped
abstraction, so relocating the storage later is a smaller change.

## Trainer Orchestration

`LMTrainer.train()` is readable; callback firing keeps logging, metrics,
checkpointing, and plotting out of the loop body. Resume logic (RNG state
round-trip, early-stopping state restore, `_resumedFromStep` to suppress a
redundant eval, config-drift warnings, LR realignment) handles the genuinely
hard parts correctly. `buildTrainer` dependency-injects evaluator and
runArtifacts and uses a factory dict for data modules (open/closed friendly).

Two notes:

- `evaluateBestCheckpointOnTest` writes its metric via
  `runArtifacts.appendMetric` directly rather than through `MetricsCallback`.
  Defensible (one-shot post-training event) but it is the one place metrics
  writing bypasses the callback system. An `on_test_end` hook would close the
  gap.
- Trainer constructs its own optimizer and LR strategy. If iterated
  self-improvement wants a fresh optimizer per round or a different schedule,
  injecting them (as evaluator/runArtifacts already are) would be consistent.

## Readiness for Self-Improvement

| Plan item | Design reality |
|---|---|
| `RewardFn` protocol | Genuinely new; nothing exists. Net-new module. |
| Dynamic data path | Half-built: `SequenceDataModule` takes in-memory tensors. |
| Best-of-N harness | Needs batched generation; generator is ~90% there. |
| Frozen reference + degeneracy guard | New, but `Checkpoint.load` + a 2nd model gives the frozen ref trivially. |
| Per-iteration structure profiling | Already exists: `EvaluationMode` + `EvalResult` provenance. |
| Determinism controls | Strong base (explicit generators); add `use_deterministic_algorithms` + seed isolation. |

The single most valuable structural addition is the `RewardFn` protocol — the
one genuinely missing abstraction. It should live in a new package that depends
on core only (consistent with the existing one-directional layering), e.g.
`llm.selfimprove` or under `research/`.

## Priorities

| # | Item | Type |
|---|---|---|
| 1 | `RewardFn` protocol in a new core-only-dependent package | New abstraction (highest value) |
| 2 | Batched N-candidate generation in `AutoregressiveGenerator` | Small seam |
| 3 | Decide optimizer/scheduler injection for iterated training | Design decision |
| 4 | `on_test_end` callback so test metrics use `MetricsCallback` | Consistency |
| 5 | ~~Consider splitting `TrainConfig` into science vs. plumbing~~ Path half done via `RunPaths` (2026-06-28); optimizer-injection half pending | Future-proofing |
| 6 | `runDirectory()` string-match fragility (now single-sourced in `RunPaths`) | Minor robustness |
| 7 | Drop `AutoregressiveGenerator` transitional `logger_or_device` param | Cleanup |
