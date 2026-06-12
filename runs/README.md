# Training Runs

Each training experiment should have its own directory:

```text
runs/<run-id>/
  run.json
  checkpoints/
    best.pt
    latest.pt
    step-000800.pt
  metrics.jsonl
  samples/
  plots/
  analysis/
```

`run.json` should record:

- source corpus manifest path and SHA-256,
- Git commit,
- tokenizer and model configuration,
- training configuration and random seed,
- parent checkpoint when resuming,
- selected model artifact, if any.

Schema version 2 stores corpus paths and hashes only under `corpora`; they are
not duplicated in the `training` configuration. Full training configuration,
including corpus paths, remains in checkpoints for resume and drift detection.

Training writes `run.json` and appends validation and final test records to
`metrics.jsonl`. The test split is evaluated from `best.pt` after training and
is not used for checkpoint selection or early stopping.

Run contents are ignored by Git because checkpoints and repeated experiment
outputs can become large. Promote intentionally retained inference artifacts
to `models/`.

Training uses `best.pt` for the lowest validation loss and `latest.pt` for
recovery. Periodic `step-NNNNNN.pt` snapshots are pruned to the configured
retention count. Checkpoints include training and evaluation RNG state plus
early-stopping progress.
