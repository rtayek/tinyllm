"""The Serializable protocol should be satisfied by every round-trippable type."""
from __future__ import annotations

from llm.Checkpoint import Checkpoint
from llm.Config import ModelConfig, RunConfig, TrainConfig
from llm.EvalResult import EvalResult
from llm.serialization_types import Serializable


def _assert_serializable(obj: Serializable) -> None:
    # Static typing already enforces this at the call sites below; the runtime
    # check and the toDict() call confirm the contract holds in practice.
    assert isinstance(obj, Serializable)
    assert isinstance(obj.toDict(), dict)


def test_model_config_is_serializable() -> None:
    _assert_serializable(ModelConfig())


def test_train_config_is_serializable() -> None:
    _assert_serializable(TrainConfig(device="cpu"))


def test_run_config_is_serializable() -> None:
    _assert_serializable(RunConfig())


def test_eval_result_is_serializable() -> None:
    _assert_serializable(EvalResult(name="book", split="val", loss=1.5))


def test_checkpoint_is_serializable() -> None:
    checkpoint = Checkpoint(
        version=1,
        modelState={},
        optimizerState={},
        step=0,
        bestValLoss=None,
        modelConfig={},
        trainConfig={},
    )
    _assert_serializable(checkpoint)
