from __future__ import annotations
import time
from typing import List, Optional, Tuple
import torch

# --- Mock Classes for Simulation ---

# Use your preferred camelCase naming convention
class MockModelConfig:
    """Mock configuration for the model."""
    dModel: int = 768

class MockTrainConfig:
    """Mock configuration for training."""
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    batchSize: int = 64
    blockSize: int = 512
    # NOTE: Set this higher (e.g., 50) if your real data loading is very slow
    mockLoadTimeMs: int = 10 
    
class MockSequenceDataModule:
    """
    Mocks your DataModule to return synthetic data and simulate loading time.
    
    The 'mockLoadTimeMs' simulates the time taken for IO, deserialization,
    and copying to the DataLoader's queue (if using one).
    """
    def __init__(self, trainCfg: MockTrainConfig, modelCfg: MockModelConfig):
        self.trainCfg = trainCfg
        self.modelCfg = modelCfg
        self.vocabSize = 10000

    def getBatch(self, split: str, generator: Optional[torch.Generator] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Simulates fetching one batch, including artificial delay."""
        
        # Simulate CPU work/IO time
        if self.trainCfg.device == 'cpu':
            time.sleep(self.trainCfg.mockLoadTimeMs / 1000.0)
        
        # Create synthetic data
        x = torch.randint(0, self.vocabSize, (self.trainCfg.batchSize, self.trainCfg.blockSize), dtype=torch.long)
        y = torch.randint(0, self.vocabSize, (self.trainCfg.batchSize, self.trainCfg.blockSize), dtype=torch.long)
        
        # Crucial: Pin memory if device is CUDA (optimizes transfer to GPU)
        if self.trainCfg.device == 'cuda':
            x = x.pin_memory()
            y = y.pin_memory()

        return x, y

# --- Bottleneck Test Logic ---

def runBottleneckTest(
    numSteps: int = 100, 
    warmupSteps: int = 10
) -> None:
    """
    Runs a test to compare data loading time vs. GPU processing time.
    """
    
    modelCfg = MockModelConfig()
    trainCfg = MockTrainConfig()
    dataModule = MockSequenceDataModule(trainCfg, modelCfg)
    
    device = trainCfg.device
    print(f"--- Running Bottleneck Test on Device: {device} ---")
    
    mockWeights: Optional[torch.Tensor] = None
    if device == 'cuda':
        # Need to create a mock tensor/module to force GPU activity
        mockWeights = torch.randn(modelCfg.dModel, modelCfg.dModel, device=device, dtype=torch.float32)
        torch.cuda.synchronize()
        
    loadTimes: List[float] = []
    gpuTimes: List[float] = []

    print(f"Batch Size: {trainCfg.batchSize}, Block Size: {trainCfg.blockSize}")
    print(f"Simulated Data Load Time (per batch): {trainCfg.mockLoadTimeMs}ms")
    print(f"Starting benchmark (Warmup: {warmupSteps} steps, Test: {numSteps} steps)...")
    
    for step in range(numSteps + warmupSteps):
        # 1. Start timer for DATA LOADING (CPU)
        startLoadTime = time.perf_counter()
        
        batchX_cpu, _batchY_cpu = dataModule.getBatch("train")
        
        endLoadTime = time.perf_counter()
        
        # 2. Start timer for GPU TRANSFER and PROCESSING
        startGpuTime = time.perf_counter()
        
        # Transfer to device (the 'pin_memory' above helps this)
        batchX_gpu = batchX_cpu.to(device, non_blocking=True)
        # We don't necessarily need batchY_gpu for this test, but include for completeness
        # batchY_gpu = batchY_cpu.to(device, non_blocking=True)

        # Simulate a typical Transformer/GPT operation:
        # Embedding lookup (or a quick matmul) + loss calculation.
        # This represents the actual forward/backward pass time.
        if device == 'cuda':
            # Create a mock embedding tensor
            mockEmbedding = torch.randn(
                trainCfg.batchSize * trainCfg.blockSize, 
                modelCfg.dModel, 
                device=device, 
                dtype=torch.float32
            )
            # Simulate a multi-head attention matrix multiplication
            if mockWeights is None:
                raise RuntimeError("Expected mockWeights to be set on CUDA")
            _ = torch.matmul(mockEmbedding, mockWeights) 
            
            # Use synchronize to wait for all GPU work to finish
            torch.cuda.synchronize() 
        else:
            # Simulate CPU processing time for comparison
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

    print("\n--- TEST RESULTS (Averages over 100 steps) ---")
    print(f"Average Data Loading Time (CPU/IO): {avgLoadTime:.2f} ms")
    print(f"Average GPU Processing Time (Transfer + Compute): {avgGpuTime:.2f} ms")
    
    # 3. Analysis and Conclusion
    if avgLoadTime > avgGpuTime * 1.5:
        print("\n*** ?? BOTTLENECK DETECTED ?? ***")
        print(f"Data Loading ({avgLoadTime:.2f}ms) is significantly slower than GPU Processing ({avgGpuTime:.2f}ms).")
        print("Your GPU is starving. The DataModule or its underlying DataLoader/Dataset is the bottleneck.")
        print("\nAction Plan: Increase num_workers in your DataLoader or optimize data reading.")
    elif avgLoadTime > avgGpuTime * 0.8:
        print("\n*** ?? POTENTIAL BOTTLENECK ?? ***")
        print(f"Data Loading ({avgLoadTime:.2f}ms) is nearly as slow as GPU Processing ({avgGpuTime:.2f}ms).")
        print("This is inefficient. You should optimize your DataModule or increase DataLoader workers.")
    else:
        print("\n*** ? NO MAJOR BOTTLENECK ? ***")
        print("Your data loading is fast enough to keep your GPU busy. This is the desired state.")

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available. Running CPU-only test. Results may not reflect real GPU bottlenecks.")
    runBottleneckTest()
