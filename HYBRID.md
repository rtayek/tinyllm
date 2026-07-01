# Hybrid & Efficient Sequence-Mixer Architectures

Planning notes on efficient alternatives to softmax attention, and how to test
them on `tinyllm`. This is the architecture-side companion to research-plan
section 10 in `LEARNED_STRUCTURE.md`. Nothing here is built yet.

## Why this is relevant to tinyllm specifically

The corrected context probe found that the Austen model **uses only ~8 bytes of
context** (loss reaches the full-context baseline by 8 bytes, flat thereafter).
That local-modeling finding is, in miniature, the exact empirical observation
the entire efficient-architecture field is built on: most of what a language
model does is local, and only a small fraction of the work genuinely needs
global attention.

That makes tinyllm an unusually good testbed. The hybrid-vs-pure question the
field spent 2025-2026 working out at enormous cost can be reproduced here, in
miniature, in minutes rather than GPU-months, on a corpus small enough to
iterate on and with a probe suite already built to measure what changed.

## The problem being solved

Softmax attention has two costs:

- **Compute** is quadratic in sequence length, O(N^2). A 10x longer sequence is
  100x more attention compute.
- **Memory**: the KV cache grows linearly with context and dominates memory at
  long context (the usual cause of out-of-memory at inference).

Every alternative below attacks one or both.

## The two families we care about

### Family 1: Cheaper sequence mixers (keep the transformer skeleton)

Replace or reduce the quadratic attention while keeping the overall block
structure (mixer + MLP + residual).

- **Short convolutions.** Mix each token with a fixed *local* window of
  neighbors. Cheap, linear, inherently local. This is what Liquid AI's LFM2
  leans on: gated short convolutions for the majority of layers.
- **State-space models (Mamba / S4 / Mamba-2).** Maintain a fixed-size
  recurrent state that summarizes history. O(1) memory per step, linear
  compute. Mamba matches or exceeds transformer quality while scaling linearly;
  a 10x longer sequence is only 10x more compute, not 100x. The tradeoff: the
  fixed state is a lossy compression of the past, so exact long-range recall is
  weaker than attention.

**The hybrid consensus (2026).** The field converged on *mixing* rather than
*replacing*. Representative points:

- **Liquid AI LFM2** - roughly 1:3 attention-to-convolution ratio (~75% of
  layers are local gated short convolutions, ~25% grouped-query attention),
  found by hardware-in-the-loop architecture search. Up to ~2x faster
  prefill/decode on CPU at matched size.
- **NVIDIA Nemotron-3** - a production hybrid that alternates regular attention
  layers with Mamba-2 state-space layers; one of the best models in its size
  class.
- Recurring finding across years of this work: adding a *small* attention
  component to a primarily-local (conv or SSM) model recovers most of the
  quality gap. A little attention goes a long way; most layers do not need it.

This 1:3-ish ratio is a quantified, expensively-searched version of exactly
what the tinyllm context probe shows for the Austen corpus: the bulk of the
modeling is local.

### Family 2: Attention-free recurrent revivals

Revive RNNs with the parallelizable training that made transformers win in the
first place - train in parallel like a transformer, run in constant memory per
step like an RNN.

- **xLSTM** - an improved LSTM shown to function as an LLM.
- **RWKV / RetNet / gated linear attention (GLA)** - reformulate the sequence
  model as a linear recurrence, dropping the quadratic softmax while keeping
  parallel training.

Families 1 and 2 overlap conceptually: convolutions, state-space recurrences,
and gated linear attention are all *linear input-varying* operators - a linear
mixing whose weights depend on the input - of which softmax attention is the
expensive special case. (Liquid AI formalizes this as the "LIV operator"
framework.)

## Planned tinyllm experiments

The `Transformer` / `Model` split already isolates the mixing layer, so an
alternative mixing block is a drop-in behind the same interface. That makes the
ablations below low-friction.

### Experiment A: three-way mixer ablation (attention vs convolution vs SSM)

Hold everything else fixed (embed dim, layer count, MLP, params) and swap only
the mixer:

1. **Attention** - the current all-attention baseline.
2. **Short causal convolution** - fixed local window (start at width 8 to match
   the measured context).
3. **Minimal selective-state recurrence** - a small Mamba-style linear
   recurrence.

Run the full probe suite (destruction experiments + context probe + n-gram
baseline) on each variant, compared at matched parameter count.

**Prediction (from the locality finding):** convolution and SSM variants should
lose very little val loss versus attention, because the model was not using
long-range attention to begin with.

### Experiment B: attention-to-local ratio sweep (find where attention is load-bearing)

Sweep the ratio of attention layers to local (conv or SSM) layers:

```
4:0   3:1   2:2   1:3   0:4
```

Locate the point where val loss *does* degrade. This empirically pins down, for
*this* corpus, how much attention is genuinely necessary - a sharper,
mechanistic version of what LFM2 / Nemotron found via expensive search. The
prediction is that degradation is small until attention is nearly gone, and
that a 1:3 ratio is roughly free.

### Experiment C: convolution window sweep

Vary the convolution width (4, 8, 16, 32). The context probe predicts width ~8
should suffice and wider windows should add little - a second, independent
confirmation of the ~8-byte locality result, this time from the architecture
side rather than the probe side.

## Why this is worth doing

- It reframes the scaling question. The roadmap says "scale capacity, not
  context." This adds a third axis: **change the mixer.** If the model is local,
  a convolution- or SSM-heavy variant may reach the same loss at substantially
  lower compute and memory.
- It produces a real, publishable-shaped result: the val-loss cost of removing
  attention, as a function of how much is removed, measured cleanly on a small
  corpus with a purpose-built probe suite.
- It validates (or refutes) the field's central efficiency bet at a scale where
  the experiment is cheap and the measurement is legible.

## Method discipline (same as all tinyllm work)

- One architectural variable at a time.
- Matched parameter count across variants (an ablation, not a capacity change).
- Full probe suite on every variant, always compared to the all-attention
  baseline.
- Byte tokens throughout (do not change tokenization and mixer at once - one
  variable at a time).

## Out of scope (for this research lens)

- **Mixture-of-Experts (MoE):** changes *how many* parameters fire per token,
  not *how* tokens mix. Orthogonal to the sequence-mixing question the probes
  are built around, and hard to study meaningfully at tiny scale (the point of
  MoE is many experts). Deprioritized.
- **Neurosymbolic / world-model / continuous-learning architectures:** these
  change what tinyllm *is*, not just how it mixes tokens. Different research
  program.
