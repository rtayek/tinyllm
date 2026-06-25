These results are excellent. They show your tinyllm checkpoint is learning real byte/character structure, spelling, word forms, and a modest amount of word-order structure. It is not merely memorizing Austen.

The headline:

```text
Strong signal:     spelling, orthography, left-to-right byte transitions
Moderate signal:   word order / syntax
Weak signal:       named entities
Useful surprise:   out-of-domain books still behave coherently
```

## 1. Baseline: the Austen model generalizes reasonably

Your Austen books cluster tightly:

```text
Austen validation loss range:
  best:  Emma                 1.4220
  worst: Northanger Abbey     1.4811
```

That is a narrow band. Good.

The out-of-domain books are worse:

```text
Sherlock Holmes       1.6764
Alice in Wonderland   1.7941
```

That is exactly what you would hope to see. The model learned general English structure, but it is better on Austen-style prose.

Approximate interpretation:

```text
Austen perplexity:   ~4.15–4.40
Sherlock perplexity:  5.35
Alice perplexity:     6.01
```

So the model is not just learning raw byte frequencies. It is sensitive to domain/style.

## 2. Context window probe: longer context matters more than expected

This is one of the most important findings.

```text
context_1     loss 3.2662
context_2     loss 2.9245
context_4     loss 2.3916
context_8     loss 2.0018
context_16    loss 1.8578
context_32    loss 1.7994
context_64    loss 1.6734
context_128   loss 1.5231
```

The old Sherlock result you mentioned said most gain came from the first 8 bytes. Your Austen checkpoint is different.

You still get meaningful improvement all the way to 128 bytes:

```text
64 → 128 improvement: 0.1502 nats
32 → 64 improvement:  0.1260 nats
16 → 32 improvement:  0.0585 nats
```

That means the model is using context beyond short spelling windows.

My interpretation:

> Your model is not only learning local character transitions. It is using phrase-level and sentence-level context, at least up to 128 bytes.

That strongly supports trying a larger context experiment later:

```text
block_size 128 → 256
```

But I would not jump immediately. First verify that your training and eval are clean, then run the block-size experiment.

## 3. Letter shuffling: huge orthography signal

Full letter shuffle gives:

```text
average delta: +2.6524 nats
average delta%: +174.1%
```

That is a very large effect.

Middle-only shuffle gives:

```text
average delta: +1.3421 nats
average delta%: +88.1%
```

This tells us something useful:

```text
Full word spelling matters a lot.
First/last letters and word shape also matter.
Internal letter order matters substantially.
```

Because middle-only shuffle is about half as damaging as full letter shuffle, the model is using both:

```text
word boundary / word shape
and
exact internal spelling
```

That is a good sign for a byte-level model.

## 4. Word-order shuffle: modest but real syntax signal

Word shuffling gives:

```text
average delta: +0.2944 nats
average delta%: +19.3%
```

This is not huge, but it is definitely not zero.

My interpretation:

> The model has learned some word-order, phrase, and syntax structure, but spelling/orthography dominates its predictive power.

That is typical for a small byte-level model. At byte level, many next-byte predictions are inside words. So spelling naturally contributes more to loss than syntax.

This result also says that improving syntax/semantics may require:

```text
larger context
larger model
more training
possibly subword/word-level experiments later
```

But the model is already beyond pure character statistics.

## 5. Reverse text: very strong directionality signal

Reverse text gives:

```text
average delta: +3.5595 nats
average delta%: +233.7%
```

That is a massive jump.

This means the model has strongly learned left-to-right byte transitions:

```text
th, he, ing, tion, ed, ly, Mr., Mrs., ...
```

Reversing byte order destroys those transitions. This result is expected, but it is a good sanity check that the model’s loss is meaningful.

## 6. Random letters: total destruction, as expected

Random letters gives:

```text
average delta: +4.6620 nats
average delta%: +306.1%
```

This is your “the model is absolutely not just reading spaces and punctuation” test.

Since punctuation and whitespace remain, but loss explodes, the model is heavily dependent on real word identity and English orthography.

## 7. Name replacement: weak but real Austen signal

Name replacement gives:

```text
average delta: +0.0544 nats
average delta%: +3.6%
```

For Austen books only, the signal is around:

```text
+0.0535 to +0.0897
```

Sherlock and Alice are unchanged because presumably your replacement list did not target those books.

This is a small signal, but still meaningful. Names matter, but they do not dominate byte-level loss. That makes sense: names are sparse compared with all the ordinary prose.

A bigger name experiment might show more if you measure only windows near names.

For example:

```text
Evaluate only text spans containing Elizabeth, Darcy, Emma, Knightley, etc.
```

Then compare baseline vs replaced names. The global average dilutes the effect.

## Main conclusion

Your checkpoint has learned a hierarchy like this:

```text
1. Strongest: valid English/Austen character transitions
2. Very strong: spelling and word identity
3. Strong: directionality
4. Moderate: word boundary and word-shape structure
5. Modest: word order / phrase syntax
6. Weak globally: named entity identity
```

That is a very good place to be for controlled experiments.

## What I would do next

### Experiment 1: train with block_size 256

Your context probe justifies this.

Keep everything else as constant as possible:

```text
current baseline:
  block_size = 128

experiment:
  block_size = 256
```

Compare:

```text
global val loss
per-book val loss
Sherlock/Alice out-of-domain loss
context probe again
word-order shuffle delta
sample quality
tokens/sec
```

The key question:

> Does block_size 256 lower baseline loss and increase the word-order-shuffle delta?

If yes, the larger context is helping the model learn higher-level structure.

### Experiment 2: increase model capacity modestly

After block size, try a modest capacity increase.

For example:

```text
baseline:
  n_layer = current
  n_head  = current
  n_embd  = current

experiment:
  n_layer + 1 or +2
  n_embd  modestly larger
```

The goal is not just lower loss. The goal is specifically:

```text
Does word_order_shuffle delta increase?
Does context_128 vs context_256 matter?
Does out-of-domain loss improve or only Austen loss?
```

### Experiment 3: span-local name replacement

Since global name replacement is diluted, make a targeted probe:

```text
1. Find validation windows containing character names.
2. Evaluate those windows normally.
3. Replace names.
4. Re-evaluate only those windows.
```

Expected result: much larger than +0.0544.

This tells you whether the model has learned local character-name contexts, such as:

```text
Mr. Darcy
Miss Bennet
Emma said
Elizabeth replied
```

### Experiment 4: sentence-boundary shuffle

Your current word-order shuffle is probably within sentence, which is good. Add another one:

```text
shuffle sentences within paragraphs
```

This tests paragraph-level coherence separately from word-level syntax.

Expected: probably smaller than word-order shuffle, unless the model is using longer discourse context.

## The most useful metric to watch

For your next model-improvement runs, I would not only track lower validation loss. Track this:

```text
structure sensitivity profile
```

Something like:

```text
baseline loss
context_8 delta
context_32 delta
context_128 delta
shuffle_letters delta
shuffle_words delta
reverse delta
random_letters delta
replace_names delta
```

Then each checkpoint has a fingerprint.

Example:

```text
Checkpoint A:
  val loss:             1.5231
  shuffle_words delta: +0.2944

Checkpoint B:
  val loss:             1.4800
  shuffle_words delta: +0.4100
```

Checkpoint B is not merely better at next-byte prediction. It is using more word-order structure.

## My read on readiness

Your project is now ready for the next stage:

```text
controlled model-improvement experiments
```

Not quite self-improvement yet, but you now have the right instrumentation to begin moving in that direction.

A good near-term milestone would be:

> Produce two checkpoints where the second has lower validation loss and a stronger word-order/context sensitivity profile than the first.

That would mean tinyllm is not just getting better at spelling; it is learning more structure.
