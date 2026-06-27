from __future__ import annotations

import os

import torch


def test_cpu_thread_environment_is_pinned_for_tests() -> None:
    assert os.environ["OMP_NUM_THREADS"] == "1"
    assert os.environ["MKL_NUM_THREADS"] == "1"
    assert os.environ["OPENBLAS_NUM_THREADS"] == "1"
    assert torch.get_num_threads() == 1
