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
from EarlyStopping import EarlyStopping
from Model import TinyGpt
from Checkpoint import CheckpointManager


def test_data_module_batch_shapes(tmp_path: Path) -> None:
    print("foobar")
    dataPath: Path = tmp_path / "input.txt"
    dataPath.write_bytes(b"abcdefghijklmnopqrstuvwxyz")

    modelConfig = ModelConfig(blockSize=4, vocabSize=256)
    trainConfig = TrainConfig(batchSize=2, dataPath=str(dataPath))

    dataModule = ByteDataModule(modelConfig, trainConfig)
    batchX, batchY = dataModule.getBatch("train")

    assert batchX.shape == (trainConfig.batchSize, modelConfig.blockSize)
    assert batchY.shape == (trainConfig.batchSize, modelConfig.blockSize)
    assert batchX.device.type == trainConfig.device


def test_model_forward_shapes() -> None:
    modelConfig = ModelConfig(blockSize=4, vocabSize=32, nEmbed=16, nHead=4, nLayer=2, dropout=0.0)
    model = TinyGpt(modelConfig)
    indices = torch.randint(0, modelConfig.vocabSize, (2, modelConfig.blockSize))

    logits, loss = model(indices, indices)

    assert logits.shape == (2, modelConfig.blockSize, modelConfig.vocabSize)
    assert loss is not None
    assert torch.isfinite(loss)


def test_early_stopping_logic() -> None:
    stopper = EarlyStopping(patience=2, delta=0.1)

    improved, _, shouldStop, _ = stopper.check(None, 1.0)
    assert improved is True and shouldStop is False

    stopper.check(1.0, 1.05)
    improved, _, shouldStop, count = stopper.check(1.0, 1.05)
    assert improved is False and count == 2 and shouldStop is True

    stopper.reset()
    improved, _, shouldStop, count = stopper.check(1.0, 0.8)
    assert improved is True and shouldStop is False and count == 0


def test_checkpoint_roundtrip(tmp_path: Path) -> None:
    trainCkptPath: Path = tmp_path / "ckpt.pt"
    modelConfig = ModelConfig(blockSize=4, vocabSize=32, nEmbed=16, nHead=4, nLayer=2, dropout=0.0)
    trainConfig = TrainConfig(batchSize=2, ckptPath=str(trainCkptPath))

    model = TinyGpt(modelConfig)
    optimizer = torch.optim.AdamW(model.parameters(), lr=trainConfig.learningRate, weight_decay=trainConfig.weightDecay)
    generator = torch.Generator()
    generator.manual_seed(123)

    manager = CheckpointManager(modelConfig, trainConfig)
    manager.saveCheckpoint(
        model,
        optimizer,
        lrStrategyState=None,
        step=10,
        bestValLoss=0.5,
        generatorState=generator.get_state(),
    )

    newModel = TinyGpt(modelConfig)
    newOptimizer = torch.optim.AdamW(newModel.parameters(), lr=trainConfig.learningRate, weight_decay=trainConfig.weightDecay)

    (
        step,
        bestValLoss,
        lrRestored,
        _version,
        versionMatches,
        drift,
        generator_state,
    ) = manager.loadCheckpoint(newModel, newOptimizer, lrStrategy=None)

    assert step == 10
    assert bestValLoss == 0.5
    assert lrRestored is False
    assert versionMatches is True
    assert drift["model"] == {} and drift["train"] == {}
    assert generator_state is not None
    for pOld, pNew in zip(model.parameters(), newModel.parameters()):
        assert torch.equal(pOld, pNew)
