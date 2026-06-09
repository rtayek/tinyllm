#!/bin/sh
set -e

export PYTHONPATH=src

RUN_DIR="${RUN_DIR:-runs/sherlock-byte-default}"
CHECKPOINT="$RUN_DIR/checkpoints/best.pt"

mkdir -p "$RUN_DIR/checkpoints"
rm -f "$CHECKPOINT"

python -m llm.Main \
  --corpus corpora/arthur-conan-doyle/adventures-of-sherlock-holmes/splits/train.txt \
  --validation-corpus corpora/arthur-conan-doyle/adventures-of-sherlock-holmes/splits/validation.txt \
  --test-corpus corpora/arthur-conan-doyle/adventures-of-sherlock-holmes/splits/test.txt \
  --checkpoint "$CHECKPOINT"
