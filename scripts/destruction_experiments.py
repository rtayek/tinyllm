"""Compatibility shim: delegates to llm.research.destruction_experiments.

The canonical entry point is ``tinyllm-destruction-experiments`` (registered
in pyproject.toml) or ``python -m llm.research.destruction_experiments``.
This file exists so that ``python scripts/destruction_experiments.py`` still
works for users who have the old command in their shell history.
"""
from __future__ import annotations

from llm.research.destruction_experiments import *  # noqa: F403
from llm.research.destruction_experiments import main


if __name__ == "__main__":
    main()
