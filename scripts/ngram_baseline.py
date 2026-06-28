"""Compatibility shim: delegates to llm.research.ngram_baseline.

The canonical entry point is ``tinyllm-ngram-baseline`` (registered
in pyproject.toml) or ``python -m llm.research.ngram_baseline``.
This file exists so that ``python scripts/ngram_baseline.py`` still
works for users who have the old command in their shell history.
"""
from __future__ import annotations

from llm.research.ngram_baseline import *  # noqa: F403
from llm.research.ngram_baseline import main


if __name__ == "__main__":
    main()
