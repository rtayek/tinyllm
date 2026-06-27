#!/bin/sh
set -e

export PYTHONPATH=src

RUN_DIR="${RUN_DIR:-runs/austen-byte}"
MAX_STEPS="${MAX_STEPS:-10000}"
RESUME="${RESUME:-0}"
CHECKPOINT="$RUN_DIR/checkpoints/best.pt"

mkdir -p "$RUN_DIR/checkpoints"
if [ "$RESUME" != "1" ]; then
  rm -f "$RUN_DIR"/checkpoints/*.pt
  rm -f "$RUN_DIR"/metrics.jsonl "$RUN_DIR"/run.json
  rm -f "$RUN_DIR"/plots/* "$RUN_DIR"/samples/*
fi

python -m llm.train_app \
  --corpus corpora/jane-austen/combined/splits/train.txt \
  --validation-corpus corpora/jane-austen/combined/splits/validation.txt \
  --test-corpus corpora/jane-austen/combined/splits/test.txt \
  --checkpoint "$CHECKPOINT" \
  --max-steps "$MAX_STEPS" \
  --plot \
  "$@"
