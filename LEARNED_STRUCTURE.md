# What Structure Is This Model Learning?

This note treats `tinyllm` as an experiment in representation learning rather
than primarily as a text-generation project. Measurements below use the
historical checkpoint at step 3100 and its now-removed legacy Sherlock corpus,
including text-mode newline normalization. They have not yet been reproduced
on the canonical story-level split.

## Current Answer

The checkpoint has clearly learned:

- byte classes such as punctuation and digits,
- spelling and common character transitions,
- word shapes and frequent short phrases,
- some useful context beyond the current word,
- broad late-layer attention patterns.

It has not yet demonstrated strong evidence of:

- robust syntax,
- discourse or story-state tracking,
- semantic representations that generalize beyond the corpus,
- substantial passage memorization,
- a predictive advantage over a well-tuned local n-gram model.

The most important result is that a smoothed 4-gram model reaches validation
loss `2.0065` nats per byte, while the transformer reaches about `2.03`. Most
of the current predictive performance is therefore still explainable by the
previous three bytes.

## Evidence

### Baselines

| Model | Validation loss | Perplexity |
| --- | ---: | ---: |
| Unigram | 3.2530 | 25.87 |
| Bigram | 2.5669 | 13.03 |
| Trigram | 2.1477 | 8.57 |
| 4-gram | 2.0065 | 7.44 |
| 5-gram | 2.2697 | 9.68 |
| Transformer | about 2.03 | about 7.6 |

The unsmoothed structure available in this small corpus makes higher-order
n-grams sparse. The 4-gram result should remain a mandatory baseline for future
experiments.

### Effective Context

Predicting the next byte from progressively longer suffixes gave:

| Context bytes | Loss |
| ---: | ---: |
| 1 | 2.6726 |
| 2 | 2.2430 |
| 4 | 2.0555 |
| 8 | 2.0141 |
| 16 | 2.0115 |
| 32 | 2.0166 |
| 64 | 2.0074 |
| 128 | 2.0406 |

This comparison changes absolute position as context length changes, so it is
not by itself a clean test of context use. A controlled test kept the input at
128 positions and replaced the distant prefix with a natural prefix from
another validation location:

| Correct suffix retained | Loss |
| ---: | ---: |
| 1 | 3.4214 |
| 2 | 2.5224 |
| 4 | 2.2065 |
| 8 | 2.1504 |
| 16 | 2.1253 |
| 32 | 2.1223 |
| 64 | 2.0578 |
| 128 | 2.0619 |

The first four bytes provide most of the gain. Context from 4 to 64 bytes is
still useful, improving loss by about `0.15` nats, but the final 64 bytes add
no measurable benefit in this sample.

### Destroying Different Structures

The model was scored on transformed validation text:

| Input structure | Loss | Change |
| --- | ---: | ---: |
| Natural text | 2.0278 | - |
| Words shuffled | 2.2709 | +0.2431 |
| Letters shuffled within words | 4.1910 | +2.1632 |
| All bytes shuffled | 4.9638 | +2.9360 |
| Text reversed | 4.2628 | +2.2350 |

Destroying spelling is far more damaging than destroying word order. This is
the clearest evidence that the model currently represents orthographic and
local phrase structure much more strongly than syntax.

### Layer Contributions

Removing one component at a time produced these validation losses:

| Intervention | Loss |
| --- | ---: |
| No intervention | 2.0046 |
| Remove layer 0 attention | 2.5775 |
| Remove layer 1 attention | 2.0335 |
| Remove layer 2 attention | 2.0220 |
| Remove layer 3 attention | 2.0245 |
| Remove layer 0 MLP | 4.1773 |
| Remove layer 1 MLP | 2.0914 |
| Remove layer 2 MLP | 2.0608 |
| Remove layer 3 MLP | 2.0358 |

Removing all attention raises loss to about `2.85`; removing all MLPs raises it
to about `3.98`. Removing blocks 1 through 3 together raises loss to about
`2.60`, even though each later component has a small individual effect. The
first block is dominant, while later blocks appear to provide distributed or
partly redundant refinements.

Zeroing positional embeddings raises loss from about `2.04` to `2.73`.
Position information is causally important, but this does not prove that the
model has learned linguistic position. It may partly reflect reliance on the
training architecture's absolute-position convention.

### Attention Patterns

First-layer heads attend locally, with expected look-back distances around
8-16 bytes and 18-35% average weight on the immediately previous byte. The
last layer is much broader, with expected distances around 34-43 bytes and
high attention entropy.

This resembles a local-to-broad progression, but broad attention is not
automatically meaningful attention. The late heads are diffuse, and individual
late attention ablations barely change loss. Attention weights should be
treated as hypotheses for causal tests, not explanations by themselves.

### Embeddings

Nearest-neighbor structure appears for several byte classes:

- `.` is close to `?`, `!`, `;`, `,`, and `:`.
- `0` is close to other digits.
- lowercase `a` is close to uppercase `A` and other vowels.
- space is related to newline and formatting characters.

Unused or rare byte values also appear among neighbors because many of the 256
possible bytes are poorly trained. Analyses should distinguish frequent ASCII
bytes from rare UTF-8 continuation bytes and unused values.

### Memorization

With an 80-byte prompt and greedy generation, exact continuation matches were
short:

| Split | Mean exact prefix | Maximum |
| --- | ---: | ---: |
| Train | 1.00 byte | 7 bytes |
| Validation | 0.45 byte | 4 bytes |

This does not support substantial verbatim passage memorization. It also does
not rule out distributed memorization of names, phrases, or corpus statistics.

## Dataset Confounds

The current split is a contiguous 90/10 split of one Project Gutenberg file.
It is not a clean train/test design for studying abstraction.

- Validation starts partway through "The Copper Beeches".
- The story ends at about 96.9% of the file.
- Roughly 31% of the validation region is Project Gutenberg license and
  website boilerplate rather than Sherlock Holmes prose.
- Adjacent train and validation regions share author, characters, formatting,
  and document-level artifacts.

For structural experiments, remove the Gutenberg header/footer and split by
story, not by byte position. A stronger test is to train on some stories and
evaluate on complete held-out stories.

## Recommended Research Plan

### 1. Preserve Training Trajectories

The project currently keeps only the best checkpoint. Save lightweight model
snapshots at fixed steps, for example:

```text
0, 50, 100, 200, 400, 800, 1600, 3200
```

Run the same probes on every snapshot. The interesting question is not only
what exists at the end, but when spelling, word boundaries, induction, and
longer context become useful.

### 2. Add Instrumentation

Add an analysis-only forward path that can return:

- residual stream after each block,
- attention matrices per layer and head,
- MLP activations,
- logits after each intermediate layer.

Keep the normal training API unchanged. Instrumentation should be optional so
it does not retain large tensors during ordinary training.

### 3. Use Causal Tests

Prefer interventions over visual inspection:

- patch one head's output from a clean example into a corrupted example,
- zero or replace selected attention edges,
- swap residual states between minimal-pair prompts,
- restrict attention to the last `k` bytes,
- compare original and corrupted contexts with positions held fixed.

A linear probe can show that information is present. A causal intervention is
needed to show that the model uses it.

### 4. Build Controlled Synthetic Corpora

Natural prose contains too many correlated features. Train small models on
datasets where the required structure is known:

- balanced brackets with varied nesting depth,
- subject-verb agreement across distractors,
- repeated-token induction such as `A B ... A -> B`,
- copy and reverse tasks,
- quotation or delimiter closure,
- latent word classes generated by a small grammar,
- names assigned temporary properties that must be recalled later.

Vary one factor at a time: context length, number of layers, number of heads,
data volume, and tokenizer.

### 5. Compare Against Restricted Models

Every experiment should include:

- unigram through n-gram baselines,
- a model restricted to the last 4 or 8 bytes,
- an attention-free baseline,
- a one-layer transformer,
- the full transformer.

Claim longer-range learning only when the full model beats both the n-gram and
restricted-context controls on held-out structures.

### 6. Keep Byte Tokens Initially

Byte tokens are an advantage for this research goal. They force the model to
construct letters, word boundaries, spelling, and larger units from raw
symbols. A BPE tokenizer would improve efficiency but would build some of that
structure into preprocessing.

Use BPE later as a controlled comparison:

- train byte and BPE models on the same text,
- approximately match parameter count and training compute,
- compare when each representation acquires syntax and long-range behavior.

## Next Concrete Experiment

The highest-value next experiment is:

1. Use the cleaned corpora generated by `tinyllm-prepare-corpora`.
2. Use the generated 8/2/2 Sherlock story split.
3. Train checkpoints from several random seeds.
4. Save model snapshots throughout training.
5. Track transformer loss, 4-gram loss, and context-restriction loss.
6. Run word-order, spelling, and natural-prefix-swap interventions at every
   snapshot.

That experiment would show whether the transformer eventually moves beyond
local byte statistics, when that transition occurs, and whether it generalizes
to a story it never saw.
