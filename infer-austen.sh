#!/bin/sh
set -e

export PYTHONPATH=src

RUN_DIR="${RUN_DIR:-runs/austen-byte}"
CHECKPOINT="$RUN_DIR/checkpoints/best.pt"

python -m llm.infer --checkpoint "$CHECKPOINT" "$@"
