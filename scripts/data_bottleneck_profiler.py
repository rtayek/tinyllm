from __future__ import annotations

import argparse
import time
from typing import Sequence

import torch


class MockModelConfig:
    """Mock configuration for the model."""

    dModel: int = 768


class MockTrainConfig:
    """Mock configuration for training."""

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batchSize: int = 64
    blockSize: int = 512
    mockLoadTimeMs: int = 10


class MockSequenceDataModule:
    """Return synthetic batches and simulate data loading time."""

    def __init__(self, trainCfg: MockTrainConfig, modelCfg: MockModelConfig):
        self.trainCfg = trainCfg
        self.modelCfg = modelCfg
        self.vocabSize = 10000

    def getBatch(
        self,
        split: str,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.trainCfg.device == "cpu":
            time.sleep(self.trainCfg.mockLoadTimeMs / 1000.0)

        x = torch.randint(
            0,
            self.vocabSize,
            (self.trainCfg.batchSize, self.trainCfg.blockSize),
            dtype=torch.long,
        )
        y = torch.randint(
            0,
            self.vocabSize,
            (self.trainCfg.batchSize, self.trainCfg.blockSize),
            dtype=torch.long,
        )

        if self.trainCfg.device == "cuda":
            x = x.pin_memory()
            y = y.pin_memory()

        return x, y


def runBottleneckTest(numSteps: int = 100, warmupSteps: int = 10) -> None:
    """Compare synthetic data loading time against processing time."""

    modelCfg = MockModelConfig()
    trainCfg = MockTrainConfig()
    dataModule = MockSequenceDataModule(trainCfg, modelCfg)

    device = trainCfg.device
    print(f"--- Running Bottleneck Test on Device: {device} ---")

    mockWeights: torch.Tensor | None = None
    if device == "cuda":
        mockWeights = torch.randn(
            modelCfg.dModel,
            modelCfg.dModel,
            device=device,
            dtype=torch.float32,
        )
        torch.cuda.synchronize()

    loadTimes: list[float] = []
    gpuTimes: list[float] = []

    print(f"Batch Size: {trainCfg.batchSize}, Block Size: {trainCfg.blockSize}")
    print(f"Simulated Data Load Time (per batch): {trainCfg.mockLoadTimeMs}ms")
    print(f"Starting benchmark (Warmup: {warmupSteps} steps, Test: {numSteps} steps)...")

    for step in range(numSteps + warmupSteps):
        startLoadTime = time.perf_counter()
        batchX_cpu, _batchY_cpu = dataModule.getBatch("train")
        endLoadTime = time.perf_counter()

        startGpuTime = time.perf_counter()
        batchX_gpu = batchX_cpu.to(device, non_blocking=True)

        if device == "cuda":
            mockEmbedding = torch.randn(
                trainCfg.batchSize * trainCfg.blockSize,
                modelCfg.dModel,
                device=device,
                dtype=torch.float32,
            )
            if mockWeights is None:
                raise RuntimeError("Expected mockWeights to be set on CUDA")
            _ = torch.matmul(mockEmbedding, mockWeights)
            torch.cuda.synchronize()
        else:
            _ = batchX_gpu.float() * 2

        endGpuTime = time.perf_counter()

        if step >= warmupSteps:
            loadTimes.append((endLoadTime - startLoadTime) * 1000)
            gpuTimes.append((endGpuTime - startGpuTime) * 1000)

    if not loadTimes or not gpuTimes:
        print("Test failed to collect data.")
        return

    avgLoadTime = sum(loadTimes) / len(loadTimes)
    avgGpuTime = sum(gpuTimes) / len(gpuTimes)

    print(f"\n--- TEST RESULTS (Averages over {numSteps} steps) ---")
    print(f"Average Data Loading Time (CPU/IO): {avgLoadTime:.2f} ms")
    print(f"Average Processing Time (Transfer + Compute): {avgGpuTime:.2f} ms")

    if avgLoadTime > avgGpuTime * 1.5:
        print("\n*** BOTTLENECK DETECTED ***")
        print(
            f"Data loading ({avgLoadTime:.2f}ms) is significantly slower "
            f"than processing ({avgGpuTime:.2f}ms)."
        )
        print("The DataModule or its underlying DataLoader/Dataset is the bottleneck.")
    elif avgLoadTime > avgGpuTime * 0.8:
        print("\n*** POTENTIAL BOTTLENECK ***")
        print(
            f"Data loading ({avgLoadTime:.2f}ms) is nearly as slow as "
            f"processing ({avgGpuTime:.2f}ms)."
        )
        print("Consider optimizing the DataModule or increasing DataLoader workers.")
    else:
        print("\n*** NO MAJOR BOTTLENECK ***")
        print("Data loading is fast enough for this synthetic benchmark.")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a synthetic data bottleneck profiler.")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--warmup-steps", type=int, default=10)
    args = parser.parse_args(argv)
    if args.steps < 1:
        parser.error("--steps must be greater than zero")
    if args.warmup_steps < 0:
        parser.error("--warmup-steps must be zero or greater")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if not torch.cuda.is_available():
        print(
            "WARNING: CUDA not available. Running CPU-only test. "
            "Results may not reflect real GPU bottlenecks."
        )
    runBottleneckTest(numSteps=args.steps, warmupSteps=args.warmup_steps)


if __name__ == "__main__":
    main()
