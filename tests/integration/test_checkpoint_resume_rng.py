from pathlib import Path
import torch

from llm.Config import ModelConfig, TrainConfig
from llm.DataModule import ByteDataModule
from llm.Model import TinyGpt
from llm.Trainer import Trainer


def test_checkpoint_restores_generator_state(tmp_path: Path) -> None:
    dataPath = tmp_path / "input.txt"
    dataPath.write_bytes(b"hello tiny llm\n" * 50)

    ckptPath = tmp_path / "ckpt.pt"

    modelCfg = ModelConfig(blockSize=8, vocabSize=256, nEmbed=16, nHead=2, nLayer=1, dropout=0.0)
    trainCfg = TrainConfig(
        batchSize=2,
        learningRate=1e-3,
        warmupFrac=0.1,
        maxSteps=3,
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

    assert ckptPath.exists(), "Checkpoint file should be written"

    checkpoint = torch.load(ckptPath, map_location=trainCfg.device, weights_only=False)  # pyright: ignore[reportUnknownMemberType]
    generatorState = checkpoint.get("generatorState", None)
    assert generatorState is not None

    dataModuleTwo = ByteDataModule(modelCfg, trainCfg)
    modelTwo = TinyGpt(modelCfg).to(trainCfg.device)
    trainerTwo = Trainer(modelCfg, trainCfg, modelTwo, dataModuleTwo)
    trainerTwo.loadCheckpointIfExists()

    assert torch.equal(trainerTwo.generator.get_state(), generatorState)

    genCopy = torch.Generator()
    genCopy.set_state(generatorState)
    batchXExpected, _ = dataModuleTwo.getBatch("train", genCopy)
    batchXActual, _ = dataModuleTwo.getBatch("train", trainerTwo.generator)
    assert torch.equal(batchXExpected, batchXActual)
