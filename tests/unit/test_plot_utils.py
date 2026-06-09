from pathlib import Path
from typing import Any, cast

import pytest
import matplotlib.pyplot as plt  # type: ignore[import]

from llm.Config import ModelConfig, TrainConfig
from llm.plot_utils import plot_training_curve


def test_plot_training_curve_closes_figure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    plt_mod = cast(Any, plt)
    figures_before = set(plt_mod.get_fignums())

    plot_path, config_path = plot_training_curve(
        [(0, 2.0, 2.1), (1, 1.9, 2.0)],
        ModelConfig(),
        TrainConfig(device="cpu"),
    )

    assert Path(plot_path).exists()
    assert Path(config_path).exists()
    assert set(plt_mod.get_fignums()) == figures_before


def test_plot_training_curve_uses_run_directory(tmp_path: Path) -> None:
    checkpoint = tmp_path / "runs" / "experiment" / "checkpoints" / "best.pt"

    plot_path, config_path = plot_training_curve(
        [(0, 2.0, 2.1)],
        ModelConfig(),
        TrainConfig(ckptPath=str(checkpoint), device="cpu"),
    )

    assert Path(plot_path).parent == checkpoint.parent.parent / "plots"
    assert Path(config_path).parent == checkpoint.parent.parent / "plots"
