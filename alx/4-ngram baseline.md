The transformer beats every n-gram model decisively. This is the key result.

**The transformer (1.523 avg) beats the 4-gram (1.827 avg) by 0.304 nats** — that's a 17% improvement. On the Sherlock corpus it beats the 4-gram by 0.317 nats despite never training on Sherlock. This is unambiguous — the transformer has learned structure that n-gram statistics cannot capture.

A few observations:

**The 4-gram is the best n-gram** — same conclusion as the legacy Sherlock run. The 5-gram gets worse (1.873 avg) because sparse high-order contexts hurt more than the extra precision helps, even with Laplace smoothing on 3MB of training data. The transformer sidesteps this entirely by learning dense representations rather than counting.

**The transformer's advantage is largest on out-of-domain text.** On Austen validation it beats the 4-gram by ~0.29 nats. On Sherlock it beats it by 0.32 nats. On Alice it beats it by 0.36 nats. The transformer generalised better than any n-gram model to text it never saw.

**The 3-gram crossover is informative.** The transformer beats the 3-gram by ~0.46 nats on Austen — meaning it's doing substantially better than trigram statistics. Given the context probe showed most gain in the first 8 bytes, the transformer has essentially learned a very efficient compressed n-gram model plus something extra in the longer context.

This closes the core research question. The transformer is clearly in post-n-gram territory. Update `LEARNED_STRUCTURE.md` with these results?