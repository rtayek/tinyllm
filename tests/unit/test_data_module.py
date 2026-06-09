import torch

from llm.Config import ModelConfig, TrainConfig
from llm.DataModule import SequenceDataModule


def test_get_batch_accepts_exactly_one_valid_window() -> None:
    model_config = ModelConfig(blockSize=4)
    train_config = TrainConfig(batchSize=1, device="cpu")
    data_module = SequenceDataModule(
        model_config,
        train_config,
        torch.arange(50),
    )
    data_module.trainSequence = torch.tensor([10, 11, 12, 13, 14])

    batch_x, batch_y = data_module.getBatch("train")

    assert torch.equal(batch_x, torch.tensor([[10, 11, 12, 13]]))
    assert torch.equal(batch_y, torch.tensor([[11, 12, 13, 14]]))


def test_get_batch_matches_expected_windows() -> None:
    model_config = ModelConfig(blockSize=3)
    train_config = TrainConfig(batchSize=2, device="cpu")
    data_module = SequenceDataModule(
        model_config,
        train_config,
        torch.arange(50),
    )
    data_module.trainSequence = torch.arange(10, 20)
    expected_generator = torch.Generator().manual_seed(42)
    starts = torch.randint(
        low=0,
        high=7,
        size=(2,),
        generator=expected_generator,
    )
    generator = torch.Generator().manual_seed(42)

    batch_x, batch_y = data_module.getBatch("train", generator)

    offsets = torch.arange(3)
    positions = starts.unsqueeze(1) + offsets.unsqueeze(0)
    assert torch.equal(batch_x, data_module.trainSequence[positions])
    assert torch.equal(batch_y, data_module.trainSequence[positions + 1])
