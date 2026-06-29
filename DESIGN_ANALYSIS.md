# Deep Design Analysis — tinyllm

Snapshot date: 2026-06-29. This goes beyond the issue catalogue in
`DESIGN_REVIEW.md` to examine whether the fundamental structure is sound, where
the conceptual load concentrates, and whether the design will hold as the
self-improvement work lands on top of it.

## 1. The architecture has a clear and correct center of gravity

Fan-in (how many internal modules import each module):

```
17  Config
11  Model
 7  tensor_utils
 6  json_utils
 5  cli_utils / Evaluator / DataModule
 4  TextGenerator / EvalResult / Checkpoint
```

`Config` (17) and `Model` (11) are the hubs, which is exactly right: the
description of an experiment and the network itself. Everything else has
fan-in <= 7, and the highest utility (`tensor_utils`) is genuinely leaf-level.
No accidental god-module, no utility grab-bag.

The critical property: the dependency graph is **acyclic and stratified**, and
`Config` sits at the bottom depending only on two trivial leaves
(`serialization_types`, `json_utils`). Because the most-depended-on module is
also the cheapest to import, the whole graph stays loose. This is the
structural reason the codebase has absorbed repeated refactors (the
`EvalResult` rename, `RunPaths`, `OptimizerFactory`) without cascading
breakage.

## 2. The deepest design decision — byte-level tokens — is load-bearing and consistent

Using raw UTF-8 bytes as the vocabulary (256 tokens) is the most consequential
choice, and it is honored end to end without leaking special-casing:

- `Model` / `Transformer` are tokenization-agnostic (just integer indices).
- `corruptions.py` operates on `bytes`, decoding/encoding at the boundary.
- `GeneratedCandidate` carries both `text` and `tokenIds`.

Design tension: **the byte assumption is implicit, not encoded in a type.**
`vocabSize=256` is just an int; the byte-ness is assumed in scattered places
(`bytes(raw_list)`, `prompt.encode("utf-8")`, `bytearray(...)`). This is fine
today and over-abstracting would be a mistake. But it is the assumption most
likely to fracture if BPE is ever added, because the fracture points are spread
across `TextGenerator`, `corruptions`, and the research scripts rather than
localized behind one tokenizer seam. If BPE becomes real, the first refactor
should be to make "how tokens become bytes/text" a single interface, *before*
writing BPE itself.

## 3. The evaluation subsystem is the high point — and reveals the one real structural seam

The `EvaluationMode` hierarchy is well-architected. The composition is
deliberate: `CorruptionEvaluator` -> `PerBookEvaluator` ->
(`FullSplitEvaluator` | `SampledLossEvaluator`), all converging on the
immutable, self-describing `EvalResult`. The planned invariance probes slot in
as one more evaluator.

The deep observation: **the family shares a concept but not a contract.** The
`evaluate()` signatures diverge:

- `SampledLossEvaluator.evaluate()` — no args
- `FullSplitEvaluator.evaluate()` — no args
- `PerBookEvaluator.evaluate(book_name, tokens, corpus)`
- `CorruptionEvaluator.evaluate(name, raw, corpus)`
- `BaselineEvaluator.evaluate(name, split, loss, n_tokens, corpus, notes)`

There is no `Protocol` unifying them because they cannot be unified as written:
the split-bound evaluators capture inputs at *construction* (dataflow via the
constructor); the per-item evaluators take inputs at *call time* (dataflow via
`evaluate`). Two different dataflow philosophies wearing the same method name.

This seam will be stressed first by the self-improvement loop, which wants to
"run this *list* of configured probes and collect their `EvalResult`s." It
cannot today — no common type to hold them in a list, no uniform invocation.
The fix is a small **parameter object** (an `EvalContext` carrying
`name`/`tokens`/`raw`/`corpus`) so every evaluator becomes
`evaluate(ctx) -> EvalResult` and a `Protocol` can describe the family. Defer
until the invariance probes are written, but this is the one structural change
the evaluation layer needs, and self-improvement is what will force it.

## 4. The serialization story is coherent but carries a quiet inconsistency

Six `toDict`/`fromDict` pairs now exist (`ModelConfig`, `TrainConfig`,
`RunConfig`, `EvalResult`, `Checkpoint`, `GeneratedCandidate`), unified on the
read side by the `Serializable` protocol (correctly scoped to `toDict` only).
The *deserialization* side diverges:

- `ModelConfig.fromDict` — `cls(**data)`, trusting
- `TrainConfig.fromDict` — filters to valid fields (tolerant of extra keys)
- `EvalResult.fromDict` / `GeneratedCandidate.fromDict` — field-by-field
  coercion with shared helpers from `serialization_types.py`
- `Checkpoint.fromDict` — `@staticmethod`, different convention

Four deserialization philosophies remain: trusting, filtering, coercing, and a
different method kind. The small optional coercion helpers now live in
`serialization_types.py` alongside the protocol they support, avoiding local
duplicates without introducing a serialization framework.

## 5. GeneratedCandidate shows the self-improvement design starting well — with one early smell

`GeneratedCandidate` is a good first move: an immutable, serializable record of
a generation with full provenance (seed, temperature, topK, token counts). It
mirrors `EvalResult`'s philosophy — measurements and generations are both
self-describing immutable records, built in the same idiom as the rest.

Early smell: `generateCandidates` produces N candidates by looping
`generateCandidate` N times, each a separate forward pass with `seed + index`.
Functionally correct and deterministic, but it forgoes batching — N sequences
could generate in one batched forward pass. Best-of-N is the inner loop of the
entire self-improvement project, so the place performance matters most is the
place currently leaving it on the table. Worth deciding before the loop is built
around the per-candidate call.

The prompt/continuation split is now token-boundary based:
`GeneratedCandidate.continuation` is decoded from
`tokenIds[promptTokenCount:]`, while `text` remains the decoded full generated
sequence. This avoids the earlier UTF-8 / `errors="replace"` fragility from
string-prefix slicing.

## 6. Where the structure will bend under self-improvement — synthesis

The self-improvement loop will press on four points, in order of resistance:

1. **Evaluation family has no common contract (Section 3)** — bites first, when
   the loop wants to run a list of probes per iteration. Needs the `EvalContext`
   parameter object.
2. **`RewardFn` has no home yet** — the one genuinely missing abstraction.
   `GeneratedCandidate` is its natural input type, so the groundwork is being
   laid in the right order.
3. **Optimizer lifecycle (already solved)** — factory injection means iterated
   training can get a fresh optimizer per round. Pre-empted.
4. **Dynamic data path (mostly solved)** — `SequenceDataModule` takes in-memory
   tensors, so generated data feeds back without disk. Remaining gap (no
   in-place update) is fine for rebuild-per-round.

Two of the four were already addressed proactively — the design is being shaped
*ahead* of the feature, the opposite of bolting the loop on and retrofitting
abstractions.

## Overall assessment

A well-architected system whose structure reflects its concepts: an acyclic
graph centered on the two things that deserve to be central, an evaluation
subsystem built for the comparative experiments the project exists to run, and
immutable self-describing records (`EvalResult`, `GeneratedCandidate`) as the
consistent currency.

Two things to watch, in priority:

1. **Unify the evaluation family behind a parameter object + protocol** when the
   invariance probes land. The real structural seam; self-improvement forces it.
2. **Decide true tensor-batched generation for best-of-N** before the loop is
   optimized around per-candidate calls. Phase 1 candidate records exist, but
   generation still loops internally.

None is urgent and none is a defect. They are the places the next phase will
apply force, and the structure should take that force well.

---

## Addendum — 2026-06-29 (later): EvaluationProbe introduced but not yet adopted

The live tree now contains a new module `EvaluationProbe.py` that did not exist
in the snapshot analyzed above. It is exactly the parameter-object + protocol
recommended in Section 3 and watch-item 1:

```python
@dataclass(frozen=True)
class EvalContext:
    name: str
    split: str = "validation"
    tokens: torch.Tensor | None = None
    raw: bytes | None = None
    corpus: str | None = None
    checkpoint: str | None = None
    notes: str | None = None

class EvaluationProbe(Protocol):
    def evaluate(self, context: EvalContext) -> EvalResult: ...
```

Both names are exported from `__init__`. This is the right design and the right
shape: a single optional-field context that can carry the inputs each evaluator
flavor needs (split-bound evaluators ignore `tokens`/`raw`; per-item evaluators
use them), and a `Protocol` that finally gives the family a common type.

**The key structural finding for this snapshot: the abstraction is defined and
exported, but not yet broadly adopted.** `EvaluationMode.py` is unchanged — its five
evaluators (`SampledLossEvaluator`, `FullSplitEvaluator`, `PerBookEvaluator`,
`CorruptionEvaluator`, `BaselineEvaluator`) still have their original divergent
`evaluate()` signatures and none implements `EvaluationProbe`. A small unit
test exercises the protocol with a fake probe, but core and research evaluators
do not consume `EvalContext` yet.

This is a normal and reasonable intermediate state — the target type was
introduced first, ahead of the migration that will conform the evaluators to
it. But it has two implications worth recording:

1. **The seam from Section 3 is half-closed.** The destination exists; the
   evaluators have not moved to it. Until they implement
   `evaluate(context: EvalContext) -> EvalResult`, the family still cannot be
   held in a `list[EvaluationProbe]` and run uniformly, which is the capability
   the self-improvement loop needs. The recommendation upgrades from "add the
   parameter object" to "complete the migration: make each `EvaluationMode`
   evaluator implement `EvaluationProbe`, moving call-time inputs
   (`book_name`/`tokens`/`raw`) out of `evaluate()`'s parameters and into the
   `EvalContext` it receives."

2. **Transitional risk: two ways to do the same thing.** While both the old
   signatures and the new protocol coexist, new code could be written against
   either. The migration should be finished (or the old signatures removed)
   before the self-improvement loop is built on top, so the loop targets the
   uniform `EvaluationProbe.evaluate(context)` contract rather than the
   divergent ones.

One small naming note: `EvalContext.split` defaults to `"validation"`, whereas
the existing evaluators default `split` to `"val"`. When the migration happens,
these must be reconciled (the `SequenceDataModule`/`Evaluator` split keys must
match the context's), or a probe will request a split that does not exist.
This is exactly the kind of mismatch the migration needs to catch.

Net: the project moved in the direction this analysis recommended, and did so
in the right order (target type first). The remaining work is conforming the
five evaluators to the protocol and reconciling the `split` default before the
self-improvement loop depends on the uniform contract.
