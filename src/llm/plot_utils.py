from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Any, cast

import matplotlib  # type: ignore[import]

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # type: ignore[import]

from .Config import ModelConfig, TrainConfig


def plot_training_curve(training_curve: List[Tuple[int, float, float]], modelConfig: ModelConfig, trainConfig: TrainConfig) -> Tuple[str, str]:
    """
    Plot and save the training/validation loss curves.
    """
    steps = [x[0] for x in training_curve]
    trainLosses = [x[1] for x in training_curve]
    valueLosses = [x[2] for x in training_curve]

    runDirectory = trainConfig.runDirectory()
    outputDirectory = (
        runDirectory / "plots" if runDirectory is not None else Path("plots")
    )
    outputDirectory.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"plot_lr{trainConfig.learningRate}_wd{trainConfig.weightDecay}_bs{trainConfig.batchSize}_{timestamp}.png"
    filepath = outputDirectory / filename

    config_dump_path = outputDirectory / f"config_{timestamp}.txt"
    with open(config_dump_path, "w", encoding="utf-8") as f:
        f.write("MODEL CONFIGURATION:\n")
        for field, value in vars(modelConfig).items():
            f.write(f"{field} = {value}\n")
        f.write("\nTRAINING CONFIGURATION:\n")
        for field, value in vars(trainConfig).items():
            f.write(f"{field} = {value}\n")

    plt_mod: Any = cast(Any, plt)
    figure, axes = plt_mod.subplots(figsize=(10, 5))
    try:
        axes.plot(steps, trainLosses, label="train loss")
        axes.plot(steps, valueLosses, label="val loss")
        axes.set_xlabel("step")
        axes.set_ylabel("loss")
        axes.set_title("Training Curve")
        axes.legend()
        axes.grid(True)
        figure.savefig(filepath, dpi=150)
    finally:
        plt_mod.close(figure)
    return str(filepath), str(config_dump_path)
