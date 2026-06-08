#!/bin/sh
set -e
export PYTHONPATH=src
python -m llm.infer "$@"
