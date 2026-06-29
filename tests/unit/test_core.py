from pathlib import Path
from typing import Callable
import logging
import torch
from unittest.mock import MagicMock

from llm.Config import ModelConfig, TrainConfig
from llm.DataModule import ByteDataModule, DataModuleConfig, SequenceDataModule
from llm.EarlyStopping import EarlyStopping
from llm.Model import TinyGPTLanguageModel
from llm.Checkpoint import CheckpointManager
from llm.Evaluator import Evaluator


class RecordingHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


def test_data_module_batch_shapes(tmp_path: Path) -> None:
    dataPath: Path = tmp_path / "input.txt"
    dataPath.write_bytes(b"abcdefghijklmnopqrstuvwxyz")

    modelConfig = ModelConfig(blockSize=4, vocabSize=256)
    trainConfig = TrainConfig(batchSize=2, dataPath=str(dataPath), device="cpu")

    dataModule = ByteDataModule(modelConfig, trainConfig)
    generator = torch.Generator()
    generator.manual_seed(0)
    batchX, batchY = dataModule.getBatch("train", generator)

    assert batchX.shape == (trainConfig.batchSize, modelConfig.blockSize)
    assert batchY.shape == (trainConfig.batchSize, modelConfig.blockSize)
    assert batchX.device.type == trainConfig.device


def test_model_forward_shapes() -> None:
    modelConfig = ModelConfig(blockSize=4, vocabSize=32, nEmbed=16, nHead=4, nLayer=2, dropout=0.0)
    model = TinyGPTLanguageModel(modelConfig)
    indices = torch.randint(0, modelConfig.vocabSize, (2, modelConfig.blockSize))

    logits, loss, _ = model(indices, indices)

    assert logits.shape == (2, modelConfig.blockSize, modelConfig.vocabSize)
    assert loss is not None
    assert torch.isfinite(loss)


def test_cached_logits_match_uncached_logits() -> None:
    modelConfig = ModelConfig(
        blockSize=8,
        vocabSize=32,
        nEmbed=16,
        nHead=4,
        nLayer=2,
        dropout=0.0,
        use_cache=True,
    )
    model = TinyGPTLanguageModel(modelConfig).eval()
    indices = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)

    full_logits, _, _ = model(indices)
    _, _, cache = model(indices[:, :3], use_cache=True)
    cached_logits, _, _ = model(
        indices[:, 3:],
        past_key_values=cache,
        use_cache=True,
    )

    assert torch.allclose(
        cached_logits[:, -1],
        full_logits[:, -1],
        atol=1e-6,
    )


def test_cached_generation_sanity() -> None:
    modelConfig = ModelConfig(
        blockSize=8,
        vocabSize=32,
        nEmbed=16,
        nHead=4,
        nLayer=1,
        dropout=0.0,
        use_cache=True,
    )
    model = TinyGPTLanguageModel(modelConfig)
    prompt = torch.tensor([[1, 2, 3]], dtype=torch.long)

    generated = model.generate_autoregressive(prompt, maxNewTokens=4)

    assert generated.shape == (1, 7)
    assert torch.equal(generated[:, : prompt.size(1)], prompt)
    assert torch.all((generated >= 0) & (generated < modelConfig.vocabSize))


def test_generation_zero_tokens_preserves_prompt() -> None:
    modelConfig = ModelConfig(
        blockSize=8,
        vocabSize=32,
        nEmbed=16,
        nHead=4,
        nLayer=1,
        dropout=0.0,
    )
    model = TinyGPTLanguageModel(modelConfig)
    prompt = torch.tensor([[1, 2, 3]], dtype=torch.long)

    generated = model.generate_autoregressive(prompt, maxNewTokens=0)

    assert torch.equal(generated, prompt)


def test_generation_rejects_negative_max_new_tokens() -> None:
    modelConfig = ModelConfig(
        blockSize=8,
        vocabSize=32,
        nEmbed=16,
        nHead=4,
        nLayer=1,
        dropout=0.0,
    )
    model = TinyGPTLanguageModel(modelConfig)
    prompt = torch.tensor([[1, 2, 3]], dtype=torch.long)

    try:
        model.generate_autoregressive(prompt, maxNewTokens=-1)
    except ValueError as exc:
        assert "maxNewTokens must be non-negative" in str(exc)
    else:
        raise AssertionError("expected ValueError for negative maxNewTokens")


def test_generation_seed_is_reproducible() -> None:
    modelConfig = ModelConfig(
        blockSize=8,
        vocabSize=32,
        nEmbed=16,
        nHead=4,
        nLayer=1,
        dropout=0.0,
    )
    model = TinyGPTLanguageModel(modelConfig)
    prompt = torch.tensor([[1, 2, 3]], dtype=torch.long)

    first = model.generate_autoregressive(
        prompt,
        maxNewTokens=8,
        temperature=0.8,
        topK=10,
        seed=123,
    )
    second = model.generate_autoregressive(
        prompt,
        maxNewTokens=8,
        temperature=0.8,
        topK=10,
        seed=123,
    )

    assert torch.equal(first, second)


def test_top_k_one_uses_greedy_token() -> None:
    modelConfig = ModelConfig(
        blockSize=8,
        vocabSize=32,
        nEmbed=16,
        nHead=4,
        nLayer=1,
        dropout=0.0,
    )
    model = TinyGPTLanguageModel(modelConfig).eval()
    prompt = torch.tensor([[1, 2, 3]], dtype=torch.long)
    logits, _, _ = model(prompt)
    expected = torch.argmax(logits[:, -1], dim=-1, keepdim=True)

    generated = model.generate_autoregressive(
        prompt,
        maxNewTokens=1,
        topK=1,
        seed=123,
    )

    assert torch.equal(generated[:, -1:], expected)


def test_cached_generation_matches_uncached_across_context_boundary() -> None:
    baseConfig = ModelConfig(
        blockSize=4,
        vocabSize=32,
        nEmbed=16,
        nHead=4,
        nLayer=2,
        dropout=0.0,
        use_cache=False,
    )
    cachedConfig = ModelConfig(
        blockSize=4,
        vocabSize=32,
        nEmbed=16,
        nHead=4,
        nLayer=2,
        dropout=0.0,
        use_cache=True,
    )
    uncachedModel = TinyGPTLanguageModel(baseConfig)
    cachedModel = TinyGPTLanguageModel(cachedConfig)
    cachedModel.load_state_dict(uncachedModel.state_dict())
    prompt = torch.tensor([[1, 2, 3]], dtype=torch.long)
    manual_seed: Callable[[int], torch.Generator] = torch.manual_seed  # type: ignore[reportUnknownMemberType]

    manual_seed(123)
    uncached = uncachedModel.generate_autoregressive(prompt, maxNewTokens=6)
    manual_seed(123)
    cached = cachedModel.generate_autoregressive(prompt, maxNewTokens=6)

    assert torch.equal(cached, uncached)


def test_cached_generation_logs_context_rebuild() -> None:
    modelConfig = ModelConfig(
        blockSize=4,
        vocabSize=32,
        nEmbed=16,
        nHead=4,
        nLayer=1,
        dropout=0.0,
        use_cache=True,
    )
    model = TinyGPTLanguageModel(modelConfig)
    prompt = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    model_logger = logging.getLogger("llm.Model")
    original_level = model_logger.level
    handler = RecordingHandler()
    model_logger.addHandler(handler)
    model_logger.setLevel(logging.DEBUG)

    try:
        model.generate_autoregressive(prompt, maxNewTokens=2)
    finally:
        model_logger.removeHandler(handler)
        model_logger.setLevel(original_level)

    assert any("KV cache reached blockSize=4" in message for message in handler.messages)


def test_early_stopping_logic() -> None:
    stopper = EarlyStopping(patience=2, delta=0.1)

    r = stopper.check(None, 1.0)
    assert r.improved is True and r.should_stop is False

    stopper.check(1.0, 1.05)
    r = stopper.check(1.0, 1.05)
    assert r.improved is False and r.no_improve_evals == 2 and r.should_stop is True

    stopper.reset()
    r = stopper.check(1.0, 0.8)
    assert r.improved is True and r.should_stop is False and r.no_improve_evals == 0


def test_early_stopping_keeps_significant_improvement_reference() -> None:
    stopper = EarlyStopping(patience=2, delta=0.003)

    stopper.check(None, 1.8025)
    r = stopper.check(1.8025, 1.7989)

    assert r.improved is False
    assert r.frac_improvement is not None and 0 < r.frac_improvement < stopper.delta
    assert r.should_stop is False
    assert r.no_improve_evals == 1
    assert stopper.referenceLoss == 1.8025


def test_checkpoint_roundtrip(tmp_path: Path) -> None:
    trainCkptPath: Path = tmp_path / "ckpt.pt"
    modelConfig = ModelConfig(blockSize=4, vocabSize=32, nEmbed=16, nHead=4, nLayer=2, dropout=0.0)
    trainConfig = TrainConfig(batchSize=2, ckptPath=str(trainCkptPath))

    model = TinyGPTLanguageModel(modelConfig)
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

    newModel = TinyGPTLanguageModel(modelConfig)
    newOptimizer = torch.optim.AdamW(newModel.parameters(), lr=trainConfig.learningRate, weight_decay=trainConfig.weightDecay)

    result = manager.restoreCheckpoint(newModel, newOptimizer, lrStrategy=None)

    assert result.step == 10
    assert result.bestValLoss == 0.5
    assert result.lrStateRestored is False
    assert result.versionMatches is True
    assert result.configDrift["model"] == {} and result.configDrift["train"] == {}
    assert result.generatorState is not None
    assert result.evaluatorGeneratorState is None
    assert result.earlyStoppingState is None
    for pOld, pNew in zip(model.parameters(), newModel.parameters()):
        assert torch.equal(pOld, pNew)


def test_checkpoint_allows_non_shape_model_config_drift(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "ckpt.pt"
    saved_config = ModelConfig(
        blockSize=4,
        vocabSize=32,
        nEmbed=16,
        nHead=4,
        nLayer=2,
        dropout=0.0,
        use_cache=False,
    )
    requested_config = ModelConfig(
        blockSize=4,
        vocabSize=32,
        nEmbed=16,
        nHead=4,
        nLayer=2,
        dropout=0.25,
        use_cache=True,
    )
    trainConfig = TrainConfig(batchSize=2, ckptPath=str(checkpoint_path))

    saved_model = TinyGPTLanguageModel(saved_config)
    saved_optimizer = torch.optim.AdamW(saved_model.parameters())
    manager = CheckpointManager(saved_config, trainConfig)
    manager.saveCheckpoint(
        saved_model,
        saved_optimizer,
        lrStrategyState=None,
        step=3,
        bestValLoss=1.0,
    )

    requested_model = TinyGPTLanguageModel(requested_config)
    requested_optimizer = torch.optim.AdamW(requested_model.parameters())
    requested_manager = CheckpointManager(requested_config, trainConfig)

    result = requested_manager.restoreCheckpoint(requested_model, requested_optimizer)

    assert result.step == 3
    assert result.configDrift["model"] == {"dropout": 0.0, "use_cache": False}
    for saved_param, requested_param in zip(
        saved_model.parameters(),
        requested_model.parameters(),
    ):
        assert torch.equal(saved_param, requested_param)


def test_model_generate_restores_training_state() -> None:
    modelConfig = ModelConfig(blockSize=8, vocabSize=32, nEmbed=16, nHead=4, nLayer=2, dropout=0.0)
    model = TinyGPTLanguageModel(modelConfig)
    # Put model into training mode
    model.train()
    assert model.training is True

    # Generate text
    initial_indices = torch.randint(0, modelConfig.vocabSize, (1, 4)) # Initial length 4
    _ = model.generate_autoregressive(initial_indices, maxNewTokens=4) # maxNewTokens = blockSize - initial_indices_length = 8 - 4 = 4

    # Assert that the model is back in training mode
    assert model.training is True


def test_evaluator_estimate_loss_restores_training_state() -> None:
    modelConfig = ModelConfig(blockSize=4, vocabSize=32, nEmbed=16, nHead=4, nLayer=2, dropout=0.0)
    trainConfig = TrainConfig(batchSize=2)

    # Mock DataModule to return dummy batches
    mock_data_module = MagicMock(spec=SequenceDataModule)
    mock_data_module.getBatch.return_value = (
        torch.randint(0, modelConfig.vocabSize, (trainConfig.batchSize, modelConfig.blockSize)),
        torch.randint(0, modelConfig.vocabSize, (trainConfig.batchSize, modelConfig.blockSize)),
    )

    # Mock EarlyStopping
    mock_early_stopping = MagicMock(spec=EarlyStopping)

    model = TinyGPTLanguageModel(modelConfig)
    # Put model into training mode
    model.train()
    assert model.training is True

    # Mock logger
    mock_logger = MagicMock(spec=logging.Logger)
    # Create a generator
    generator = torch.Generator()
    generator.manual_seed(1337)

    evaluator = Evaluator(
        model=model,
        data_module=mock_data_module,
        trainConfig=trainConfig,
        early_stopping=mock_early_stopping,
        generator=generator,
        logger=mock_logger,
    )

    # Estimate loss
    _ = evaluator.estimate_loss()

    # Assert that the model is back in training mode
    assert model.training is True


def test_evaluator_full_split_matches_all_window_loss() -> None:
    modelConfig = ModelConfig(
        blockSize=2,
        vocabSize=8,
        nEmbed=8,
        nHead=2,
        nLayer=1,
        dropout=0.0,
    )
    trainConfig = TrainConfig(batchSize=2, device="cpu")
    tokens = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.long)
    dataModule = SequenceDataModule(
        modelConfig,
        DataModuleConfig.fromTrainConfig(trainConfig),
        sequence=tokens,
        validationSequence=tokens,
    )
    model = TinyGPTLanguageModel(modelConfig)
    evaluator = Evaluator(
        model,
        dataModule,
        trainConfig,
        EarlyStopping(patience=1, delta=0.0),
    )

    full_loss = evaluator.estimate_split_full("val", batch_size=2)

    # Reference: non-overlapping windows, stride = blockSize, each target
    # scored exactly once.
    block_size = modelConfig.blockSize
    last_start = tokens.size(0) - block_size - 1
    starts = list(range(0, last_start + 1, block_size))
    losses: list[torch.Tensor] = []
    with torch.no_grad():
        for start in starts:
            batch_x = tokens[start : start + block_size].unsqueeze(0)
            batch_y = tokens[start + 1 : start + block_size + 1].unsqueeze(0)
            logits, _, _ = model(batch_x)
            losses.append(
                torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    batch_y.reshape(-1),
                    reduction="sum",
                )
            )
    expected = sum(float(loss.item()) for loss in losses) / (
        len(losses) * block_size
    )

    assert abs(full_loss - expected) < 1e-6
