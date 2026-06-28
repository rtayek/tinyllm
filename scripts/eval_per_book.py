"""Compatibility shim: delegates to llm.research.eval_per_book.

The canonical entry point is ``tinyllm-eval-per-book`` (registered
in pyproject.toml) or ``python -m llm.research.eval_per_book``.
This file exists so that ``python scripts/eval_per_book.py`` still
works for users who have the old command in their shell history.
"""
from __future__ import annotations

from llm.research.eval_per_book import *  # noqa: F403
from llm.research.eval_per_book import main


if __name__ == "__main__":
    main()
