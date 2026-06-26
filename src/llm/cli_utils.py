from __future__ import annotations

import argparse
import logging


LOG_LEVELS: dict[str, int] = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}


def require_positive(parser: argparse.ArgumentParser, flag: str, value: int) -> None:
    if value <= 0:
        parser.error(f"{flag} must be greater than zero")


def require_non_negative(
    parser: argparse.ArgumentParser,
    flag: str,
    value: int | float,
) -> None:
    if value < 0:
        parser.error(f"{flag} must be non-negative")


def parse_log_level(
    parser: argparse.ArgumentParser,
    value: str,
    default: int = logging.INFO,
) -> int:
    level_name = value.upper()
    if level_name not in LOG_LEVELS:
        valid = ", ".join(LOG_LEVELS)
        parser.error(f"--log-level must be one of: {valid}")
    return LOG_LEVELS.get(level_name, default)
