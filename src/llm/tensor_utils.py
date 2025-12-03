from typing import Callable, cast
import random
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    manual_seed: Callable[[int], torch.Generator] = torch.manual_seed  # type: ignore[reportUnknownMemberType]
    manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"

def get_master_process() -> bool:
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0

def get_num_gpus() -> int:
    if not torch.cuda.is_available():
        return 0
    return torch.cuda.device_count()

def get_ddp_free_model(model: nn.Module) -> nn.Module:
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        return cast(nn.Module, model.module)  # type: ignore[reportUnknownMemberType]
    return model

def tensor_to_int_list(tensor: torch.Tensor) -> list[int]:
    """
    Flatten a tensor and return a list of ints.
    Centralizes the cast to keep type checkers happy.
    """
    flat = tensor.view(-1).to(dtype=torch.long)
    flat_list = cast(list[int], flat.tolist())  # type: ignore[reportUnknownMemberType]
    return [int(v) for v in flat_list]
