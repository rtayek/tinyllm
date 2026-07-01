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

## Recommendation: build a mixer block, do NOT adopt LFM2

There is a tension worth naming up front. The primary tinyllm goal is
**discovering what structure and relationships the model learns**. LFM2 and the
whole efficient-architecture program optimize a *different* objective: **getting
the same quality more cheaply**. These are not the same project, and conflating
them is a trap.

Adopting the actual LFM2 - downloading Liquid's pretrained model, or
reimplementing their gated-convolution + GQA + LIV-operator stack - would make
the research *harder*, not easier:

- **It adds architectural complexity that must then be seen through.** The
  method depends on the model being simple enough that a destruction experiment
  or context probe has a clean interpretation. A hybrid with three mixer types,
  multiplicative gates, and input-varying weights has *more* moving parts to
  attribute structure to, not fewer. Interpretive budget goes to understanding
  the architecture instead of what it learned.
- **A pretrained LFM2 is the wrong object entirely.** It is trained on trillions
  of tokens with a subword tokenizer. Using it forfeits byte-level control,
  small-corpus iteration speed, and known-provenance training data - the exact
  things that make the probes *mean* something. `replace_names` works as a
  control only because we know precisely what went into training; that
  evaporates with a frontier pretrained model.

**The version that serves the goal is the comparison, not the adoption.** The
interesting question is not "is LFM2 good" - it is **"does changing the mixer
change what structure the model learns?"** That is a structure-and-relationships
question, and it is genuinely novel:

- Train an attention tinyllm and a convolution tinyllm on the *same* Austen
  corpus, matched size, everything else identical.
- Run the identical destruction suite and context probe on both.
- The question is not which has lower loss - it is **do they learn the same
  structure hierarchy?** Does the convolution model show the same +2.65
  spelling delta, the same +0.29 word-order signal, the same ~8-byte locality?

Both outcomes are real findings. If the hierarchies are identical, the structure
is a property of the *data and task*, not the architecture - attention was not
doing anything special for this corpus. If they differ, it localizes exactly
what attention contributes that convolution does not. Either way it is a result
about what the model learns, which is the stated goal.

**Concrete recommendation:** do not adopt LFM2. Build a single convolution mixer
block as a drop-in alternative to the attention block, and run the existing
probes on it. This is a few hundred lines against the interface that already
exists (`Transformer` / `Model` already isolate the mixer), it preserves every
property that makes the probes interpretable, and it turns "efficient
architectures" from a *performance* detour back into a *structure-discovery*
experiment.

Suggested order:

1. Finish the self-improvement scaffolding currently mid-stream, or park it
   cleanly at a committed checkpoint.
2. Add one convolution mixer block behind the existing mixer interface.
3. Train conv-tinyllm on Austen, matched params.
4. Run the destruction suite + context probe on both attention and conv models.
5. Compare *hierarchies*, not losses.

That yields the LFM2 *insight* (is attention necessary for this structure?)
without the LFM2 *baggage*.

**A note on focus.** Several rich strands are now open at once: self-improvement,
invariance probes, the mixer ablation, and the arithmetic-corpus idea. Each is
good on its own. The risk is not any single one - it is that opening the next
before closing the last means none reaches the measurement that justified
starting it. If the mixer question is the most compelling right now, it is worth
consciously choosing it *as* the next strand and letting the others wait, rather
than running them in parallel.

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
