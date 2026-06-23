from typing import cast
import logging
import torch


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"

def resolve_device(requested_device: str, logger: logging.Logger | None = None) -> str:
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        if logger is not None:
            logger.warning("CUDA requested, but not available; falling back to cpu")
        return "cpu"
    return requested_device

def tensor_to_int_list(tensor: torch.Tensor) -> list[int]:
    """
    Flatten a tensor and return a list of ints.
    Centralizes the cast to keep type checkers happy.
    """
    flat = tensor.view(-1).to(dtype=torch.long)
    flat_list = cast(list[int], flat.tolist())  # type: ignore[reportUnknownMemberType]
    return [int(v) for v in flat_list]
