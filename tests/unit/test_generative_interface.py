# tests/unit/test_generative_interface.py

import logging
from pathlib import Path

from llm.Config import ModelConfig, TrainConfig, RunConfig
from llm.Main import buildTrainer
from llm.generative import TinyGptGenerative


def test_tiny_gpt_generative_produces_text(tmp_path: Path) -> None:
    # Use a tiny synthetic corpus and force CPU for a quick, deterministic smoke run.
    data_path = tmp_path / "input.txt"
    data_path.write_bytes(b"hello tiny llm\n" * 10)

    model_cfg = ModelConfig(blockSize=8, nEmbed=16, nHead=2, nLayer=1, dropout=0.0)
    train_cfg = TrainConfig(
        batchSize=2,
        dataPath=str(data_path),
        device="cpu",
        dataModule="byte",
    )
    run_cfg: RunConfig = RunConfig(modelConfig=model_cfg, trainConfig=train_cfg)

    trainer = buildTrainer(run_cfg, log=logging.getLogger("test"))
    lm = TinyGptGenerative(trainer.model, trainer.trainConfig, trainer.logger)

    text = lm.generate_text(max_new_tokens=8)

    assert isinstance(text, str)
    assert len(text) > 0
