import torch
from pathlib import Path

from llm.Config import ModelConfig, TrainConfig
from llm.Model import TinyGPTLanguageModel
from llm.Checkpoint import CheckpointManager


def test_checkpoint_export_and_load(tmp_path: Path) -> None:
    dataPath = tmp_path / "input.txt"
    dataPath.write_bytes(b"hello tiny llm\n" * 20)

    trainCkpt = tmp_path / "train_ckpt.pt"
    modelExport = tmp_path / "model_export.pt"

    modelConfig = ModelConfig(blockSize=8, vocabSize=256, nEmbed=16, nHead=2, nLayer=1, dropout=0.0)
    trainConfig = TrainConfig(
        batchSize=2,
        learningRate=1e-3,
        warmupFrac=0.1,
        maxSteps=2,
        evalInterval=1,
        evalIters=1,
        weightDecay=0.0,
        plotCurve=False,
        ckptPath=str(trainCkpt),
        dataPath=str(dataPath),
        device="cpu",
    )

    torch.manual_seed(0)  # pyright: ignore[reportUnknownMemberType]
    model = TinyGPTLanguageModel(modelConfig)
    optimizer = torch.optim.AdamW(model.parameters(), lr=trainConfig.learningRate, weight_decay=trainConfig.weightDecay)
    manager = CheckpointManager(modelConfig, trainConfig)
    manager.saveCheckpoint(model, optimizer, lrStrategyState=None, step=1, bestValLoss=0.5, generatorState=None)

    manager.saveModel(str(modelExport))
    assert modelExport.exists()

    loadedModel = TinyGPTLanguageModel(modelConfig)
    loader = CheckpointManager(modelConfig, trainConfig, modelCkptPath=str(modelExport))
    loader.loadModel(loadedModel, str(modelExport))

    for pSaved, pLoaded in zip(model.parameters(), loadedModel.parameters()):
        assert torch.equal(pSaved, pLoaded)
