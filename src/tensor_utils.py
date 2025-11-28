from __future__ import annotations

from typing import List, cast

import torch


def tensor_to_int_list(tensor: torch.Tensor) -> List[int]:
    """
    Flatten a tensor and return a list of ints.
    Centralizes the cast to keep type checkers happy.
    """
    flat = tensor.view(-1).to(dtype=torch.long)
    flat_list: List[int] = cast(List[int], flat.tolist())  # pyright: ignore[reportUnknownMemberType]
    return [int(v) for v in flat_list]
