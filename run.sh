#!/bin/sh
set -e
export PYTHONPATH=src
rm -f plots/*
rm -f checkpoints/*
python -m llm.Main
