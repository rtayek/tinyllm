from __future__ import annotations

import argparse
from typing import Sequence

import torch


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print CUDA/GPU availability.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    parse_args(argv)
    print("CUDA available:", torch.cuda.is_available())
    print("Device count:", torch.cuda.device_count())
    print("Current device:", torch.cuda.current_device() if torch.cuda.is_available() else None)
    print("Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)


if __name__ == "__main__":
    main()
