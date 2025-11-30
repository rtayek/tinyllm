from __future__ import annotations

import os
from datetime import datetime
from typing import List, Tuple, Any, cast

import matplotlib.pyplot as plt  # type: ignore[import]

from Config import ModelConfig, TrainConfig


def plot_training_curve(
    training_curve: List[Tuple[int, float, float]],
    modelConfig: ModelConfig,
    trainConfig: TrainConfig,
) -> Tuple[str, str]:
    """
    Plot and save the training/validation loss curves.
    """
    steps = [x[0] for x in training_curve]
    trainLosses = [x[1] for x in training_curve]
    valueLosses = [x[2] for x in training_curve]

    dir = "plots"
    os.makedirs(dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"plot_lr{trainConfig.learningRate}_wd{trainConfig.weightDecay}_bs{trainConfig.batchSize}_{timestamp}.png"
    filepath = os.path.join(dir, filename)

    config_dump_path = os.path.join(dir, f"config_{timestamp}.txt")
    with open(config_dump_path, "w", encoding="utf-8") as f:
        f.write("MODEL CONFIGURATION:\n")
        for field, value in vars(modelConfig).items():
            f.write(f"{field} = {value}\n")
        f.write("\nTRAINING CONFIGURATION:\n")
        for field, value in vars(trainConfig).items():
            f.write(f"{field} = {value}\n")

    plt_mod: Any = cast(Any, plt)
    plt_mod.figure(figsize=(10, 5))
    plt_mod.plot(steps, trainLosses, label="train loss")
    plt_mod.plot(steps, valueLosses, label="val loss")
    plt_mod.xlabel("step")
    plt_mod.ylabel("loss")
    plt_mod.title("Training Curve")
    plt_mod.legend()
    plt_mod.grid(True)
    plt_mod.savefig(filepath, dpi=150)
    return filepath, config_dump_path
