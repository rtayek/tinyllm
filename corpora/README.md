# Corpora

Canonical fiction corpora use:

```text
corpora/<author>/<work>/
  manifest.json
  raw/
  clean/
    full.txt
  units/
    chapters/ or stories/
  splits/
    train.txt
    validation.txt
    test.txt
```

`manifest.json` records provenance, SHA-256 hashes, logical units, and split
membership. Rebuild all corpora with:

```sh
tinyllm-prepare-corpora
```

Use `--download-pride` only when the stored Pride and Prejudice source is
missing or intentionally being refreshed.

The removed `fixtureData/` layout belonged to the historical checkpoint. New
experiments use these canonical corpus paths and should not resume that
checkpoint without deliberately accepting the data change.
