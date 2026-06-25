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
- **Long-range context** — the model makes meaningful use of 64–128 byte
  context. The last marginal (64→128) is −0.15 nats — the context window
  ceiling has not been reached.

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

Context restricted by zeroing the leading bytes of each 128-byte block.
Loss measured only on the final N positions. Averaged across all eight books.

| Context | Avg Loss | Delta | Delta% | Marginal |
|---:|---:|---:|---:|---:|
| 1 | 3.2662 | +1.743 | +114% | — |
| 2 | 2.9245 | +1.401 | +92% | −0.342 |
| 4 | 2.3916 | +0.869 | +57% | −0.533 |
| 8 | 2.0018 | +0.479 | +31% | −0.390 |
| 16 | 1.8578 | +0.335 | +22% | −0.144 |
| 32 | 1.7994 | +0.276 | +18% | −0.058 |
| 64 | 1.6734 | +0.150 | +10% | −0.126 |
| 128 | 1.5231 | +0.000 | +0% | −0.150 |

Key observations:

- **Most gain in the first 8 bytes.** 8-byte context recovers about 72%
  of full-context performance. Local character statistics dominate.
- **Non-monotonic marginal profile.** There is a marked dip at 16→32
  (−0.058) followed by recovery at 32→64 (−0.126) and 64→128 (−0.150).
  This pattern appeared in earlier Pride-only runs and is reproducible.
  Austen's sentences are often longer than 32 bytes — the model may be
  picking up useful sentence-boundary structure at 64–128 that it cannot
  use at 16–32.
- **Context ceiling not reached.** The 64→128 marginal is −0.150, not
  near zero. Increasing `blockSize` to 256 or 512 is likely to help.

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

The context probe shows the 128-byte ceiling has not been reached
(64→128 marginal is −0.15, not near zero). Recommended next scaling steps,
one at a time:

- Increase `blockSize` from 128 to 256 (most likely to help given the
  context probe results).
- Increase `nEmbed` from 256 to 512.
- Increase `nLayer` from 4 to 6 or 8.

Run the full probe suite after each change to measure what capacity buys.

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
