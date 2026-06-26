from typing import cast
import logging
import torch


def resolve_device(requested_device: str, logger: logging.Logger | None = None) -> str:
    if requested_device.startswith("cuda"):
        if not torch.cuda.is_available():
            if logger is not None:
                logger.warning("CUDA requested, but not available; falling back to cpu")
            return "cpu"
        if ":" in requested_device:
            try:
                index = int(requested_device.split(":", 1)[1])
            except ValueError:
                if logger is not None:
                    logger.warning("Invalid CUDA device '%s'; falling back to cpu", requested_device)
                return "cpu"
            if index < 0 or index >= torch.cuda.device_count():
                if logger is not None:
                    logger.warning("CUDA device '%s' is not available; falling back to cpu", requested_device)
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
