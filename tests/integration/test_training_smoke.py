import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for candidate in (SRC, ROOT, ROOT.parent):
    path_str = str(candidate)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from Config import ModelConfig, TrainConfig
from DataModule import ByteDataModule
from Model import TinyGpt
from Trainer import Trainer


def test_training_smoke(tmp_path: Path) -> None:
    # Synthetic dataset
    dataPath = tmp_path / "input.txt"
    dataPath.write_bytes(b"hello tiny llm\n" * 200)

    ckptPath = tmp_path / "ckpt.pt"

    modelCfg = ModelConfig(
        blockSize=8,
        vocabSize=256,
        nEmbed=16,
        nHead=2,
        nLayer=1,
        dropout=0.0,
    )
    trainCfg = TrainConfig(
        batchSize=2,
        learningRate=1e-3,
        warmupFrac=0.1,
        maxSteps=5,
        evalInterval=1,
        evalIters=2,
        weightDecay=0.0,
        plotCurve=False,
        ckptPath=str(ckptPath),
        dataPath=str(dataPath),
        device="cpu",
    )

    torch.manual_seed(42)  # pyright: ignore[reportUnknownMemberType]
    dataModule = ByteDataModule(modelCfg, trainCfg)
    model = TinyGpt(modelCfg).to(trainCfg.device)
    trainer = Trainer(modelCfg, trainCfg, model, dataModule)

    trainer.loadCheckpointIfExists()
    trainer.train()

    assert trainer.trainingCurve, "Training curve should not be empty after training"
    assert trainer.bestValLoss is not None
    assert ckptPath.exists(), "Checkpoint file should be written"
