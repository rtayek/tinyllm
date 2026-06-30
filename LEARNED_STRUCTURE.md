# What Structure Is This Model Learning?

This note treats `tinyllm` as an experiment in representation learning rather
than primarily as a text-generation project.

## Status of Measurements

### Current checkpoint (Austen, step 9500)

The measurements in the **Current Evidence** section below were taken against
the checkpoint trained on the combined Jane Austen corpus (all six novels,
~3MB of training text) for 9500 steps, at which point early stopping fired.

```text
Checkpoint:  runs/austen-byte/checkpoints/best.pt
Step:        9500
Val loss:    1.4461  (combined Austen validation split)
Train loss:  1.4343
Train/val gap: 0.012  (no overfitting)
```

Per-book validation losses at this checkpoint:

| Book | Val Loss | Perplexity |
|---|---:|---:|
| Pride and Prejudice | 1.4305 | 4.18 |
| Sense and Sensibility | 1.4724 | 4.36 |
| Emma | 1.4220 | 4.15 |
| Mansfield Park | 1.4774 | 4.38 |
| Persuasion | 1.4312 | 4.18 |
| Northanger Abbey | 1.4811 | 4.40 |
| Sherlock Holmes (unseen) | 1.6764 | 5.35 |
| Alice in Wonderland (unseen) | 1.7941 | 6.01 |

The model generalizes well to out-of-domain text. Sherlock Holmes and Alice
in Wonderland were never in training, yet the model assigns reasonable loss
to both — particularly Sherlock, whose formal prose style is closer to
Austen's than Carroll's absurdist fiction.

### Legacy checkpoint (Sherlock, step 3100)

Earlier measurements were taken against a checkpoint at step 3100 trained on
the Adventures of Sherlock Holmes only. That checkpoint is superseded. The
legacy results are preserved in the **Historical Evidence** section below for
comparison but should not be treated as current.

---

## Current Answer (Austen checkpoint, step 9500)

The checkpoint has clearly learned:

- **Letter identity** — which specific letters appear in which positions.
  Replacing letters with random letters is the most destructive intervention
  (average +4.66 nats across all books).
- **Directionality** — English bytes flow left to right. Reversing the byte
  sequence is nearly as catastrophic (average +3.56 nats).
- **Spelling and orthography** — specific letter sequences inside words.
  Full letter shuffle within words costs average +2.65 nats.
- **Word shape** — the model recognises word boundaries and edge letters
  independently of internal spelling. Middle-only letter shuffle costs +1.34
  nats (half the full shuffle signal), cleanly decomposing shape recognition
  from exact internal spelling.
- **Word order** — a modest but real and consistent signal. Shuffling word
  order within sentences costs average +0.29 nats across all eight books,
  with no book below +0.27. Slightly stronger on Austen prose (+0.29–0.34)
  than on Sherlock (+0.28) and Alice (+0.27), consistent with Austen's more
  syntactically structured sentences.

Most importantly, the model is **almost entirely local**:

- **It uses only about 8 bytes of context.** A corrected context probe (see
  below) shows next-byte loss reaches the full-context baseline by 8 bytes of
  history and is flat thereafter. Context beyond ~8 bytes provides no
  measurable benefit. The model is effectively a high-order local model, not
  a long-range one.

It has not demonstrated strong evidence of:

- **Character name identity** — replacing Austen character names with neutral
  placeholders (Alpha, Beta, Gamma…) costs only +0.05–0.09 nats per book.
  The model does not strongly rely on seeing specific names. Sherlock and
  Alice score exactly 0.0 on this experiment (correct — no names replaced),
  confirming the control works.
- Robust syntax or grammar beyond local word-order sensitivity.
- Discourse or story-state tracking.
- Semantic representations beyond surface statistics.
- Substantial passage memorization.

The destruction patterns transfer almost perfectly from Austen (training
domain) to Sherlock and Alice (unseen), with identical rank order and similar
magnitudes. The model learned general English structure, not
Austen-specific patterns.

---

## Current Evidence (Austen checkpoint, step 9500)

All experiments run with `scripts/destruction_experiments.py`,
200 eval iterations, seed 42.

### Baselines (all books)

| Book | Val Loss | Perplexity |
|---|---:|---:|
| Pride and Prejudice | 1.4305 | 4.18 |
| Sense and Sensibility | 1.4724 | 4.36 |
| Emma | 1.4220 | 4.15 |
| Mansfield Park | 1.4774 | 4.38 |
| Persuasion | 1.4312 | 4.18 |
| Northanger Abbey | 1.4811 | 4.40 |
| Sherlock Holmes (unseen) | 1.6764 | 5.35 |
| Alice in Wonderland (unseen) | 1.7941 | 6.01 |
| **Average** | **1.5231** | |

### N-gram Baselines

Not yet reproduced on the canonical Austen split. The transformer val loss
of 1.446 is well below the legacy Sherlock 4-gram baseline of 2.007, but a
direct comparison on the same corpus is required before making a formal claim.
This is the first priority for future work.

### Context Window Probe

Each measurement feeds the model a genuinely shorter sequence of exactly N
bytes of history and records the loss on the single next-byte prediction.
Averaged across all eight books.

(An earlier version of this probe zero-padded the leading bytes of a full
128-byte block instead of shortening the sequence. Because byte 0 is a real,
embedded token, that padding contaminated the small-context measurements and
produced a spurious "long-range context helps" signal. The numbers below use
the corrected method.)

| Context | Avg Loss | Delta | Delta% | Marginal |
|---:|---:|---:|---:|---:|
| 1 | 2.4861 | +0.963 | +63% | — |
| 2 | 2.0446 | +0.522 | +34% | −0.442 |
| 4 | 1.6249 | +0.102 | +7% | −0.420 |
| 8 | 1.5307 | +0.008 | +0.5% | −0.094 |
| 16 | 1.5164 | −0.007 | −0.4% | −0.014 |
| 32 | 1.5146 | −0.009 | −0.6% | −0.002 |
| 64 | 1.4929 | −0.030 | −2.0% | −0.022 |
| 128 | 1.5257 | +0.003 | +0.2% | +0.033 |

Key observations:

- **The model is effectively local: ~8 bytes is enough.** By 8 bytes of
  history the loss (1.531) has already reached the full-context baseline
  (1.523). The first 4 bytes alone recover most of the performance.
- **Context beyond 8 bytes provides no measurable benefit.** From 8 to 128
  bytes the loss is flat within sampling noise (1.49–1.53). The tiny negative
  deltas at 16/32/64 and the tiny positive delta at 128 are all noise around
  the baseline, not signal.
- **There is no long-range structure being used.** The earlier "dip at 16–32
  then recovery at 64–128" pattern was entirely an artifact of the zero-pad
  methodology and does not survive the corrected probe.
- **Implication for scaling:** the model is not using the 128-byte window it
  already has, so increasing `blockSize` will almost certainly not help at
  this model size. The bottleneck is capacity or the nature of byte-level
  local modeling, not context length.

### Structure-Destruction Experiments

`!!` = delta > 1.0 nat (large).  `!` = delta > 0.2 nat (moderate).

**shuffle_letters** — shuffle all characters within each word:

| Book | Baseline | Corrupted | Delta | Delta% |
|---|---:|---:|---:|---:|
| Pride and Prejudice | 1.4305 | 4.1729 | +2.7424 | +192% |
| Sense and Sensibility | 1.4724 | 4.1982 | +2.7258 | +185% |
| Emma | 1.4220 | 4.1845 | +2.7625 | +194% |
| Mansfield Park | 1.4774 | 4.2506 | +2.7733 | +188% |
| Persuasion | 1.4312 | 4.2096 | +2.7784 | +194% |
| Northanger Abbey | 1.4811 | 4.2016 | +2.7205 | +184% |
| Sherlock Holmes | 1.6764 | 4.1490 | +2.4727 | +148% |
| Alice in Wonderland | 1.7941 | 4.0382 | +2.2440 | +125% |
| **Average** | **1.5231** | **4.1756** | **+2.6524** | **+174%** |

**shuffle_middle** — preserve first and last letter; shuffle middle only:

| Book | Baseline | Corrupted | Delta | Delta% |
|---|---:|---:|---:|---:|
| Pride and Prejudice | 1.4305 | 2.8450 | +1.4146 | +99% |
| Sense and Sensibility | 1.4724 | 2.9167 | +1.4443 | +98% |
| Emma | 1.4220 | 2.8434 | +1.4214 | +100% |
| Mansfield Park | 1.4774 | 2.9205 | +1.4431 | +98% |
| Persuasion | 1.4312 | 2.8271 | +1.3959 | +98% |
| Northanger Abbey | 1.4811 | 2.8943 | +1.4132 | +95% |
| Sherlock Holmes | 1.6764 | 2.8646 | +1.1883 | +71% |
| Alice in Wonderland | 1.7941 | 2.8101 | +1.0160 | +57% |
| **Average** | **1.5231** | **2.8652** | **+1.3421** | **+88%** |

**Decomposition:** full shuffle costs +2.65, middle-only costs +1.34 — almost
exactly half. About half the spelling signal comes from word edge recognition
(first/last letter) and half from exact internal letter order. The model has
learned both word shape and internal spelling.

**shuffle_words** — shuffle word order within each sentence:

| Book | Baseline | Corrupted | Delta | Delta% |
|---|---:|---:|---:|---:|
| Pride and Prejudice | 1.4305 | 1.7717 | +0.3412 | +24% |
| Sense and Sensibility | 1.4724 | 1.7359 | +0.2634 | +18% |
| Emma | 1.4220 | 1.7109 | +0.2889 | +20% |
| Mansfield Park | 1.4774 | 1.7590 | +0.2817 | +19% |
| Persuasion | 1.4312 | 1.7474 | +0.3162 | +22% |
| Northanger Abbey | 1.4811 | 1.7933 | +0.3122 | +21% |
| Sherlock Holmes | 1.6764 | 1.9566 | +0.2803 | +17% |
| Alice in Wonderland | 1.7941 | 2.0651 | +0.2710 | +15% |
| **Average** | **1.5231** | **1.8175** | **+0.2944** | **+19%** |

Word order signal is consistent and real across every book — no outliers.
Slightly stronger on Austen prose than on Sherlock or Alice.

**reverse** — reverse the entire byte sequence:

| Book | Baseline | Corrupted | Delta | Delta% |
|---|---:|---:|---:|---:|
| Pride and Prejudice | 1.4305 | 4.9480 | +3.5175 | +246% |
| Sense and Sensibility | 1.4724 | 4.9722 | +3.4998 | +238% |
| Emma | 1.4220 | 5.1009 | +3.6789 | +259% |
| Mansfield Park | 1.4774 | 4.8581 | +3.3807 | +229% |
| Persuasion | 1.4312 | 5.0038 | +3.5726 | +250% |
| Northanger Abbey | 1.4811 | 4.9572 | +3.4761 | +235% |
| Sherlock Holmes | 1.6764 | 5.1068 | +3.4304 | +205% |
| Alice in Wonderland | 1.7941 | 5.7142 | +3.9201 | +219% |
| **Average** | **1.5231** | **5.0826** | **+3.5595** | **+234%** |

Alice scores the highest reverse delta (+3.92) — Carroll's verse, invented
words, and rhythmic constructions are maximally disrupted by reversal.

**random_letters** — replace every ASCII letter with a random letter a-z:

| Book | Baseline | Corrupted | Delta | Delta% |
|---|---:|---:|---:|---:|
| Pride and Prejudice | 1.4305 | 6.1946 | +4.7641 | +333% |
| Sense and Sensibility | 1.4724 | 6.2670 | +4.7945 | +326% |
| Emma | 1.4220 | 6.2130 | +4.7910 | +337% |
| Mansfield Park | 1.4774 | 6.3430 | +4.8657 | +329% |
| Persuasion | 1.4312 | 6.2487 | +4.8175 | +337% |
| Northanger Abbey | 1.4811 | 6.2343 | +4.7532 | +321% |
| Sherlock Holmes | 1.6764 | 6.1508 | +4.4744 | +267% |
| Alice in Wonderland | 1.7941 | 5.8298 | +4.0357 | +225% |
| **Average** | **1.5231** | **6.1851** | **+4.6620** | **+306%** |

**replace_names** — replace Austen character names with neutral placeholders:

| Book | Baseline | Corrupted | Delta | Delta% |
|---|---:|---:|---:|---:|
| Pride and Prejudice | 1.4305 | 1.5174 | +0.0869 | +6% |
| Sense and Sensibility | 1.4724 | 1.5417 | +0.0693 | +5% |
| Emma | 1.4220 | 1.5117 | +0.0897 | +6% |
| Mansfield Park | 1.4774 | 1.5308 | +0.0535 | +4% |
| Persuasion | 1.4312 | 1.4917 | +0.0605 | +4% |
| Northanger Abbey | 1.4811 | 1.5562 | +0.0751 | +5% |
| Sherlock Holmes | 1.6764 | 1.6764 | +0.0000 | +0% ✓ |
| Alice in Wonderland | 1.7941 | 1.7941 | +0.0000 | +0% ✓ |
| **Average** | **1.5231** | **1.5775** | **+0.0544** | **+4%** |

Sherlock and Alice score exactly 0.0 — the control works correctly. The
Austen signal (+0.05–0.09 per book) is real but tiny. The model has not
learned character-specific name distributions in any strong sense. It does
not strongly rely on seeing "Elizabeth" vs "Alpha". Names matter slightly,
probably because they are high-frequency tokens with distinctive
capitalization patterns.

### Experiment Summary

| Experiment | Avg Delta | Avg Delta% | What it means |
|---|---:|---:|---|
| random_letters | +4.66 | +306% | Letter identity — foundation of everything |
| reverse | +3.56 | +234% | Strong left-to-right directionality |
| shuffle_letters | +2.65 | +174% | Spelling and orthography |
| shuffle_middle | +1.34 | +88% | Internal letter order (beyond word edges) |
| shuffle_words | +0.29 | +19% | Word order — real but modest |
| replace_names | +0.05 | +4% | Character names — nearly no signal |

The ratio shuffle_middle / shuffle_letters ≈ 0.51 cleanly decomposes spelling
knowledge into two equal parts: word-edge shape recognition and exact internal
letter ordering.

---

## Historical Evidence (legacy Sherlock checkpoint, step 3100)

Preserved for comparison. All measurements below used the legacy 90/10
contiguous byte split of the Gutenberg Sherlock file, which included ~31%
license boilerplate in the validation region. Do not compare these numbers
directly to the current evidence above.

### Baselines (legacy)

| Model | Validation loss | Perplexity |
|---|---:|---:|
| Unigram | 3.2530 | 25.87 |
| Bigram | 2.5669 | 13.03 |
| Trigram | 2.1477 | 8.57 |
| 4-gram | 2.0065 | 7.44 |
| 5-gram | 2.2697 | 9.68 |
| Transformer | ~2.03 | ~7.6 |

The transformer barely beat the 4-gram at that checkpoint on that corpus.

### Structure-Destruction (legacy)

| Input structure | Loss | Change |
|---|---:|---:|
| Natural text | 2.0278 | — |
| Words shuffled | 2.2709 | +0.2431 |
| Letters shuffled within words | 4.1910 | +2.1632 |
| All bytes shuffled | 4.9638 | +2.9360 |
| Text reversed | 4.2628 | +2.2350 |

### Layer Contributions (legacy)

| Intervention | Loss |
|---|---:|
| No intervention | 2.0046 |
| Remove layer 0 attention | 2.5775 |
| Remove layer 1 attention | 2.0335 |
| Remove layer 2 attention | 2.0220 |
| Remove layer 3 attention | 2.0245 |
| Remove layer 0 MLP | 4.1773 |
| Remove layer 1 MLP | 2.0914 |
| Remove layer 2 MLP | 2.0608 |
| Remove layer 3 MLP | 2.0358 |

Layer 0 MLP dominates. Later layers provide distributed, partly redundant
refinements.

### Memorization (legacy)

With an 80-byte prompt and greedy generation, exact continuation matches:

| Split | Mean exact prefix | Maximum |
|---|---:|---:|
| Train | 1.00 byte | 7 bytes |
| Validation | 0.45 byte | 4 bytes |

No evidence of substantial verbatim memorization.

### Historical Dataset Confounds

The legacy split was a contiguous 90/10 split of one Project Gutenberg file.
~31% of the validation region was Gutenberg license boilerplate. The canonical
split (story-level train/val/test assignment, wrapper-stripped) addresses all
of these issues.

---

## Recommended Research Plan

### 0. Reproduce N-gram Baselines on the Austen Canonical Split (First Priority)

The transformer val loss of 1.446 is well below the legacy Sherlock 4-gram
baseline of 2.007, but a direct comparison on the same corpus is required.
Run unigram through 5-gram models on the combined Austen validation split and
record the crossover point.

### 1. Layer Ablations on the Current Checkpoint

Reproduce the layer-contribution measurements on the Austen checkpoint:

- Is layer 0 MLP still dominant, or does more data distribute the load?
- Do later layers contribute more at val loss 1.45 than they did at 2.03?
- Does ablating attention matter more with the improved checkpoint?

### 2. Scale Model Capacity

The corrected context probe shows the model uses only ~8 bytes of context and
gains nothing from the 128-byte window it already has. **Increasing
`blockSize` is therefore not expected to help at this model size** — the
bottleneck is capacity or the nature of byte-level local modeling, not context
length. Prioritize capacity over context:

- Increase `nEmbed` from 256 to 512.
- Increase `nLayer` from 4 to 6 or 8.
- Only revisit `blockSize` after a larger model, and only if a fresh context
  probe on that model shows it actually using the full window.

Run the full probe suite after each change to measure what capacity buys, and
in particular re-run the context probe — a larger model may begin to use
longer context even though this one does not.

### 3. Preserve Training Trajectories

Save model snapshots at log-spaced intervals and run the destruction
experiments on each one. When is each type of structure acquired?

```text
step 0, 100, 200, 400, 800, 1600, 3200, 6400, 9500
```

### 4. Add Instrumentation

Add an analysis-only forward path returning residual streams, attention
matrices, MLP activations, and per-layer logits without affecting normal
training.

### 5. Use Causal Tests

Prefer interventions over visual inspection: patch head outputs, zero
attention edges, swap residual states between minimal-pair prompts.

### 6. Build Controlled Synthetic Corpora

Train small models on datasets where the required structure is known:
balanced brackets, subject-verb agreement, induction heads, copy tasks,
latent word classes from a small grammar.

### 7. Compare Against Restricted Models

Every experiment should include unigram through n-gram baselines, a
context-restricted model (last 4 or 8 bytes only), an attention-free
baseline, a one-layer transformer, and the full transformer.

### 8. Keep Byte Tokens for Research; Add BPE as a Comparison

Byte tokens force the model to construct letters, word boundaries, spelling,
and larger units from raw symbols. BPE would obscure what the transformer
learned. Add BPE only as a controlled comparison after scaling.

### 9. Structure-Preservation / Invariance Probes

The destruction experiments ask "what happens when I *break* a structure?"
This strand asks the dual question: "what transformations can I apply that the
model treats as *equivalent*?" A transformation the model is invariant to is
one its learned representation preserves.

**Motivation (the categorical lens).** Borrowing loosely from category theory:
a structure-preserving map (a functor) sends objects and their relationships to
new objects while keeping the relationships intact; a diagram "commutes" when
two paths through it agree. Translated to a transformer, the "objects" are
representations in the residual stream and the "morphisms" are the maps the
attention/MLP layers apply to them. Asking whether the model is invariant to a
structure-preserving input transformation is asking whether the corresponding
diagram commutes in representation space — whether the model learned the
*relationship* rather than the surface tokens. This is the measurable,
representation-level version of the idea; category theory is the analysis
framework here, not training data. (Training the model *on* category-theory
text is a separate, and for a tiny local model unpromising, question.)

**This is not hypothetical — we already have two invariance results.** The
existing destruction suite was secretly measuring structure preservation:

- `replace_names` (+0.05 nats, near-zero) is an invariance result. Swapping
  Elizabeth -> Alpha consistently leaves the loss almost unchanged, which means
  the model learned a representation where the name slot is interchangeable —
  a near-commuting diagram / natural-transformation-like property. The
  Sherlock/Alice 0.0 control confirms the measurement is clean.
- `shuffle_middle` vs `shuffle_letters` (the ~0.51 ratio) shows word-shape
  recognition is preserved independently of internal spelling — two separable
  structure-preserving maps.

**Proposed new probes.** Define input transformations that *should* leave a
structure-respecting model's loss invariant, and measure the actual delta. A
small delta means the model learned a representation in which that
transformation commutes; a large delta means it did not. Candidates:

- **Consistent variable/name renaming** — generalize `replace_names`: rename a
  token consistently everywhere and measure invariance. Already partly done;
  formalize and extend to non-name tokens.
- **Consistent character substitution** — swap two characters (e.g. every `e`
  <-> `q`) throughout the text. A model keyed to *relationships* between
  symbols rather than their identities would be partly invariant; a purely
  identity-based model would not. The gap quantifies how identity-bound the
  representation is.
- **Numeric shift** (for a future math/synthetic corpus) — add a constant to
  every number. Invariance would indicate the model learned numeric
  *relationships* rather than memorized digit strings.
- **Case/whitespace-preserving permutations** — transformations that preserve a
  named structural property, to isolate which properties the model encodes.

**Method.** Reuse the destruction-suite machinery exactly: same checkpoint,
same eval, one transformation at a time, report baseline / transformed / delta
per book. The only conceptual difference is the *expected* result — for
destruction we expect a large delta (structure was used), for invariance probes
we expect a *small* delta (structure was preserved). Low delta = the model
learned the invariance; high delta = it did not. Controls (a transformation
applied to a book it cannot affect, mirroring the Sherlock/Alice control in
`replace_names`) keep the measurement honest.

**Why it is worth doing.** It turns a vague intuition ("the model learned a
hierarchy of objects and relationships") into falsifiable measurements using
tooling that already exists, and it gives a principled vocabulary for results
like `replace_names` that we already have but described only informally.

### 10. Efficient Sequence-Mixer Experiments (Convolution vs Attention)

The corrected context probe's headline finding — **the model uses only ~8
bytes of context** — is, in miniature, the exact empirical observation that an
entire line of architecture research is built on: most of what a language model
does is *local*, and only a small fraction of the work genuinely needs global
attention. This strand asks: **if the model is local, does it even need
attention?**

**Background.** Softmax attention scales quadratically with sequence length
(O(N^2) compute) and its KV cache grows linearly with context, which dominates
memory at long context. A family of sub-quadratic alternatives replaces or
reduces attention:

- **State-space models** (Mamba / S4): a fixed-size recurrent state summarizes
  history; O(1) memory per step, linear compute, but lossy long-range recall.
- **Gated linear attention / RetNet**: reformulate attention as a linear
  recurrence, dropping the quadratic softmax.
- **Short convolutions**: mix each token with a fixed *local* window of
  neighbors — cheap, linear, inherently local.
- **Hybrids** (the current consensus, e.g. Liquid AI's LFM2): mostly cheap
  local mixing (gated short convolutions) with a *small* number of attention
  layers sprinkled in to recover precise long-range retrieval. LFM2 uses
  roughly a 1:3 attention-to-convolution ratio — i.e. ~75% of layers are
  local — a ratio found by hardware-in-the-loop architecture search.

**The connection to our finding.** LFM2's 1:3 ratio is a quantified, expensively
searched version of exactly what the context probe shows for this corpus: the
bulk of the modeling is local. If the Austen model isn't using long-range
attention anyway, replacing some of its attention layers with short causal
convolutions should cost almost nothing — and would do so far more cheaply.

**Proposed experiments.**

- **Attention -> convolution ablation.** Replace one or more of the 4 attention
  layers with short causal depthwise convolutions (fixed local window, e.g.
  width 8 to match the measured context). Re-run the full destruction suite +
  context probe. *Prediction, given the locality finding:* a mostly- or
  fully-convolutional tinyllm should lose very little val loss, because the
  model wasn't using long-range attention to begin with.
- **Find where attention becomes load-bearing.** Sweep the
  attention-to-convolution ratio (4:0, 3:1, 2:2, 1:3, 0:4) and locate the
  point where val loss *does* degrade. That empirically pins down, for *this*
  corpus, how much attention is genuinely necessary — a sharper, mechanistic
  version of what LFM2 found via expensive search.
- **Convolution window sweep.** Vary the convolution width (4, 8, 16, 32). The
  context probe predicts width ~8 should suffice and wider windows should add
  little — a second, independent confirmation of the ~8-byte locality result
  from the architecture side rather than the probe side.

**Why it reframes the scaling question.** The current roadmap (section 2) says
"scale capacity, not context." This strand adds a third axis: **change the
mixer.** If the model is local, a convolution-heavy variant may reach the same
loss at substantially lower compute and memory. Measuring exactly that — the
val-loss cost of removing attention, as a function of how much is removed —
would be a legitimate, novel result on a corpus small enough to iterate on in
minutes rather than the GPU-months such ablations cost at frontier scale.

**Method.** Same discipline as everything else: one architectural variable at a
time, the full probe suite (destruction + context probe + n-gram baseline) run
on each variant, every variant compared against the current all-attention
baseline at matched parameter count. The `Transformer`/`Model` split already
isolates the mixing layer, so a convolution block is a drop-in alternative to
the attention block behind the same interface — the architecture is well-placed
for this ablation.
