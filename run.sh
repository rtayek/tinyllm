#!/bin/sh
set -e
rm -f plots/*
rm -f checkpoints/*
python src/Main.py
