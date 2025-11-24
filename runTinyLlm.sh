#!/usr/bin/env bash
# Startup script for the CamelCase Tiny LLM project.
# Runs Main.py with normal (line-buffered) output.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

python Main.py
