# Efficiency, Structure, and Compression

This research theme asks how much useful learned structure TinyLLM can retain
while reducing full-attention use, active parameter count, numerical precision,
and training-data redundancy.

The motivating observation is deliberately modest: many recent language-model
improvements appear to combine recognizable transformer components with better
routing, local or linear mixing, data selection, quantization, distillation, and
hardware-aware execution. TinyLLM cannot reproduce frontier-scale results, nor
should it pretend to. It can test the underlying claims in a controlled setting
where architecture, data, compute, and learned structure are directly measured.

## Research Questions

1. How many full-attention layers are necessary for the structural behavior
   observed by the existing probes?
2. Can causal local mixers replace some or all full-attention layers without
   losing the model's learned-structure fingerprint?
3. Do sparse feed-forward layers preserve useful structure at lower active
   parameter counts?
4. Which learned capabilities degrade first under INT8, INT4, ternary, and
   binary weight constraints?
5. Can a sparse teacher transfer useful structure into a smaller dense student?
6. Does curated training data outperform larger noisy datasets under equal-token
   and equal-compute budgets?
7. Which apparent efficiency gains survive measurement of actual memory,
   throughput, wall-clock time, and energy-relevant compute proxies?

## Architecture Track

The first comparison keeps the current transformer baseline and varies the
sequence of full-attention and causal local-mixer blocks.

| Schedule | Meaning |
|---|---|
| `AAAA` | Full-attention baseline |
| `LLAA` | Two local mixers followed by two attention blocks |
| `LALA` | Alternating local mixer and attention blocks |
| `LLLL` | Local-mixer-only model |

The initial local mixer should be simple, causal, and easy to interpret. A
small depthwise causal convolution or similarly constrained mixer is preferable
to importing a large architecture whose behavior would be harder to isolate.

Later architecture experiments may add:

- sliding-window attention,
- dilated local attention,
- occasional global-attention layers,
- recurrent or state-space-style mixers,
- sparse feed-forward layers with top-k routing.

Every architecture comparison should control parameter count where possible and
report both total and active parameters.

## Sparse Feed-Forward Track

A minimal mixture-of-experts experiment can compare:

1. a dense feed-forward model,
2. a sparse model with larger total capacity but matched active parameters,
3. a smaller dense student trained from the sparse teacher.

A suitable first implementation is four feed-forward experts with top-1 routing.
The purpose is not to claim frontier-scale mixture-of-experts behavior. The
purpose is to determine whether routing produces measurable specialization,
changes the learned-structure fingerprint, or merely adds complexity at this
scale.

Useful routing measurements include:

- expert utilization,
- load imbalance,
- token-to-expert stability,
- specialization by byte, word shape, punctuation, or corpus,
- quality at matched active compute.

## Quantization Track

Quantization should proceed from conventional baselines toward more speculative
weight constraints:

1. FP32 baseline,
2. BF16 or FP16 inference,
3. INT8 post-training quantization,
4. INT4 weight-only quantization,
5. ternary-weight experimental layers,
6. binary-weight experiments only after ternary behavior is understood.

For every precision level, record:

- validation and test loss,
- checkpoint size,
- peak memory,
- inference tokens per second,
- output degradation,
- structural-probe degradation.

The central question is not only whether perplexity survives compression. It is
which kinds of learned structure disappear first as precision falls.

## Structure Under Compression

Run the existing learned-structure measurements across architecture and
precision combinations:

- natural-text validation loss,
- shuffled-letter loss,
- middle-letter shuffle loss,
- shuffled-word loss,
- reversed-text loss,
- random-letter loss,
- character-name replacement loss,
- context-window sensitivity,
- n-gram advantage,
- Jacobian-style or perturbation sensitivity when available.

A result table should resemble:

| Architecture | Precision | Natural Text | Word Shuffle | Letter Shuffle | Context Sensitivity |
|---|---:|---:|---:|---:|---:|
| `AAAA` | FP32 | ... | ... | ... | ... |
| `LALA` | FP32 | ... | ... | ... | ... |
| `LALA` | INT8 | ... | ... | ... | ... |
| `LALA` | INT4 | ... | ... | ... | ... |
| `LALA` | ternary | ... | ... | ... | ... |

This may reveal whether compression first removes memorized detail, local
orthography, word order, long-range dependencies, or several forms of structure
at different rates.

## Data-Quality Track

Create controlled training corpora that differ in quality while preserving
reproducibility:

- clean literary prose,
- mixed literary prose,
- duplicated passages,
- boilerplate-heavy text,
- noisy OCR-like text,
- random web-style fragments,
- curated high-structure subsets.

Compare both equal-token and equal-compute training. Data-quality claims should
not be inferred from unequal training budgets.

The main question is:

> Does better-curated data improve learned structure more than simply adding
> more tokens?

## Distillation Track

TinyLLM can test small-scale structural distillation without claiming that the
result generalizes automatically to frontier models.

Possible comparisons:

- dense teacher to smaller dense student,
- sparse teacher to smaller dense student,
- logits-only distillation,
- hidden-state or feature matching,
- ordinary supervised training at the same student compute budget.

Evaluate not only student loss, but also whether the student's destruction and
context fingerprints converge toward the teacher's.

## Hardware-Aware Reporting

Every experiment should report enough information to distinguish architectural,
memory, numerical, and runtime efficiency:

- total parameters,
- active parameters per token,
- checkpoint size,
- training tokens,
- estimated FLOPs or a clearly defined compute proxy,
- wall-clock training time,
- peak GPU memory,
- inference tokens per second,
- hardware and software versions,
- random seed and corpus hashes.

An optimization is not considered successful merely because one metric improves.
The tradeoff among quality, structure, storage, memory, and speed must be visible.

## Initial Experiment Sequence

1. Complete the `AAAA`, `LLAA`, `LALA`, and `LLLL` comparison.
2. Run the existing structure-destruction and context probes for every schedule.
3. Add INT8 and INT4 checkpoint evaluation.
4. Compare structural fingerprints across architecture and precision.
5. Implement a minimal top-1 sparse feed-forward layer.
6. Compare dense and sparse models at matched active parameter counts.
7. Add controlled clean, duplicated, boilerplate, and noisy-data experiments.
8. Add a small teacher-student distillation experiment.
9. Introduce ternary weights only after the conventional quantization baselines
   are reproducible.

## Evidence Standard

Claims from industry discussions, model announcements, podcasts, and benchmark
leaderboards are treated as hypotheses, not project assumptions.

TinyLLM should prefer conclusions that survive:

- controlled baselines,
- matched budgets,
- repeated seeds,
- out-of-domain evaluation,
- learned-structure probes,
- direct runtime measurement.

The project succeeds when it explains what was preserved, what was lost, and
what the efficiency gain actually cost. Producing a smaller checkpoint without
that explanation would be compression, but not much of an experiment.
