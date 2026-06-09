# Retained Models

This directory describes intentionally retained inference models. Use one
directory per promoted model:

```text
models/<model-id>/
  README.md
  model_config.json
  tokenizer.json
  corpus_manifest.json
  provenance.json
  model.pt
```

`provenance.json` should record the source run, checkpoint step, Git commit,
corpus manifest hash, and model-file SHA-256.

PyTorch model files are ignored by Git. Store retained weights with Git LFS,
release artifacts, or external object storage, while committing their metadata
and documentation here.
