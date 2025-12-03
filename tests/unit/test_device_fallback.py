import torch
import logging
from llm.Config import RunConfig, TrainConfig, ModelConfig
from llm.Main import buildTrainer

def test_buildTrainer_falls_back_to_cpu_when_cuda_unavailable():
    model_cfg = ModelConfig()
    train_cfg = TrainConfig(device="cuda")  # intentionally request CUDA
    run_cfg = RunConfig(modelConfig=model_cfg, trainConfig=train_cfg)
    if torch.cuda.is_available():
        return
    trainer = buildTrainer(run_cfg, log=logging.getLogger("test"))
    assert trainer.trainConfig.device == "cpu"
