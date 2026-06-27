# Self-Improvement Experiments — Design Notes

Status: planning. Nothing here is built yet. This document captures the goal,
the framing, and the build plan so we can pick it up later.

## The Goal

Run **controlled self-improvement experiments** on the tiny byte-level model,
with a specific research lens: not just "can the model improve a reward," but

> **When a model improves itself, what structure does it acquire, and in what
> order?**

This is the same representation-learning lens used in `LEARNED_STRUCTURE.md`,
now pointed at self-improvement instead of ordinary gradient training.

Chosen method: **expert iteration / best-of-N self-training** (generate N
candidates, score them, keep the best, train toward those). Simpler than
policy-gradient RL but still captures "improve against a signal."

## The Reframed Research Question

The destruction experiments already established a *hierarchy of learned
structure* for normally-trained models:

```
letter identity  >  directionality  >  spelling  >  word shape  >  word order  >  (≈no) names
```

and the model is almost entirely **local (~8 bytes)**.

So the sharper question is:

> Does best-of-N self-training climb that hierarchy differently than gradient
> training does? Does it merely sharpen structure the model already has
> (locality, spelling), or can it push the model *up* the hierarchy toward
> longer-range structure (word order, syntax) that ordinary training barely
> produced?

The destruction-experiment suite is the instrument to answer this: run it on
the base model and after each self-improvement iteration, and watch which
deltas move.

## Why the Reward Is the Independent Variable

Key insight: **the reward determines which part of the hierarchy can possibly
improve.**

- Reward **n-gram likelihood** → rewards local statistics the model is already
  good at → prediction: sharpening of locality/spelling, **no hierarchy climb**.
  This is the deliberate **control**.
- Reward **long-range coherence** (e.g. likelihood under a higher-order n-gram,
  or agreement between distant positions) → prediction: this is the condition
  that *might* push the model to use more context.
- Reward **held-out loss / perplexity** → prediction: general sharpening.

The comparison across rewards *is* the experiment: "best-of-N with reward X
produces structure change Y." The reward is not a detail to settle quickly — it
is the experiment's independent variable.

## What Makes It "Controlled"

Every iteration is profiled with the existing measurement tools, against a
frozen reference model. "Self-improvement" is never just a rising reward
number — it is a measured change in the structure hierarchy. A rising reward
with no structural change (or with degeneration) is a null/negative result, and
we want to be able to see that clearly.

## What the Project Already Has (helps)

- Clean training loop with a callback system to hook into.
- Deterministic evaluation (`estimate_split_full`) + metrics/checkpoint pipeline.
- Working generation path (`AutoregressiveGenerator`) with temperature/top-k/seed.
- Strong measurement discipline: destruction experiments, n-gram baselines,
  context probe, per-book eval.

## What's Missing (must build)

1. **A `RewardFn` protocol** — sequence(s) → scalar score. Pluggable, because
   the reward is the independent variable and will be swapped.
2. **A generation-driven / dynamic data path** — the current data module is
   file-based and static. Self-training needs to train on freshly generated,
   freshly scored sequences (in-memory or dynamically written corpus).
3. **A frozen reference checkpoint** — baseline to measure improvement against
   and to keep the model from drifting into degenerate text.
4. **Tighter determinism controls** — `torch.use_deterministic_algorithms`,
   seed isolation across generate/score/train phases, variance reporting. (Some
   of this was flagged earlier but not yet implemented.)
5. **A degeneracy guard** — tiny models optimizing a reward collapse fast
   (repetition, mode collapse). Need detection + a KL-to-reference or
   distance penalty.

## Build Plan (measurement-first order)

1. `RewardFn` protocol — `(sequence) -> float` (and a batched form).
2. First reward: **n-gram likelihood**, reusing the existing `NgramModel`. It
   is the natural baseline reward *because* we predict it will not climb the
   hierarchy — that's the control.
3. **Best-of-N generation harness** — given a prompt, generate N candidates
   deterministically, score them, keep top-k. Reuses `AutoregressiveGenerator`.
4. **Dynamic data source** — feed kept samples back as training data. (The one
   real gap in current code.)
5. **Frozen reference checkpoint + degeneracy guard** — distinguish "improved"
   from "collapsed," and measure drift (KL / distance to reference).
6. **Experiment harness** — run the destruction suite + n-gram baseline +
   held-out loss on the base model and after each iteration, logging the full
   structure profile each round. This per-iteration structure profile is the
   actual output of the project.

## Recommended First Milestone

Build the full loop with the **n-gram reward as the deliberate control**
(predict: sharpens locality, no hierarchy climb), get per-iteration structure
profiling working end to end, confirm the prediction, and *then* introduce a
**long-range reward** as the experimental condition to see if anything can push
the model up the hierarchy.

First milestone = a complete, measurable loop with a falsifiable prediction,
not an open-ended RL project.

## Open Decisions (to confirm when we start)

1. First milestone: n-gram control first, then long-range reward second?
   (Alternatives: start directly with a long-range reward; or build the
   reward-agnostic harness first and defer the reward choice.)
2. Per-iteration measurement set that defines "what structure it learned":
   full destruction suite + context probe + n-gram baseline each iteration
   (most thorough), vs. context probe + held-out loss only (faster), vs.
   destruction suite only.

## Prerequisite Cleanup (from existing tech debt, relevant here)

- Determinism audit (`torch.use_deterministic_algorithms`, global-random
  isolation) — directly needed for *controlled* experiments.
- `estimate_split_full` sliding-window tail handling — only matters if a
  stride<block_size reward/metric is used (see HANDOFF.md known debt).
