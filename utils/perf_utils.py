"""
Performance Measurement Utilities for Model Profiling

This module provides utilities to measure model efficiency metrics including:
- GFLOPs (Giga Floating Point Operations)
- MMACs (Million Multiply-Accumulate Operations)
- Model Size (in MB)
- Energy Consumption (using Zeus Monitor)
- Latency (inference time from energy measurement)

"""

import torch
import torch.nn as nn
from fvcore.nn import FlopCountAnalysis
from zeus.monitor import ZeusMonitor


def get_model_size_mb(model):
    """
    Calculate model size in megabytes.

    Args:
        model: PyTorch model

    Returns:
        float: Model size in MB
    """
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    size_mb = (param_size + buffer_size) / (1024 ** 2)
    return size_mb


def measure_flops_macs(model, dataloader, device):
    """
    Measure GFLOPs and MMACs using real data from dataloader.

    Args:
        model: PyTorch model in eval mode
        dataloader: DataLoader providing real input batches
        device: Device to run the model on

    Returns:
        tuple: (gflops, mmacs) - GigaFLOPs and Million MACs
    """
    model.eval()

    # Get first real batch
    for x, y in dataloader:
        x = x.to(device).float()
        break

    # Wrapper to handle model output format
    class ModelWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            return self.model(x)

    try:
        wrapped_model = ModelWrapper(model)
        flops_analyzer = FlopCountAnalysis(wrapped_model, x)
        total_flops = flops_analyzer.total()

        # Convert to GFLOPs and MMACs
        gflops = total_flops / 1e9
        mmacs = total_flops / 2e6  # 1 MAC = 2 FLOPs

        return gflops, mmacs

    except Exception as e:
        print(f"Error measuring FLOPs/MACs: {e}")
        return None, None


@torch.no_grad()
def measure_energy(model, dataloader, device, n_batches=50):
    """
    Measure energy consumption per batch using Zeus Monitor.

    Args:
        model: PyTorch model in eval mode
        dataloader: DataLoader providing real input batches
        device: CUDA device to monitor (e.g., 'cuda:0')
        n_batches: Number of batches to measure (default: 50)

    Returns:
        tuple: (energy_per_batch, time_per_batch) in Joules and milliseconds
               Returns (None, None) if not on CUDA or if error occurs
    """
    model.eval()

    # Check if device is CUDA
    if 'cuda' not in str(device):
        print("Energy measurement only available on CUDA devices")
        return None, None

    # Extract GPU index from device string
    gpu_idx = int(str(device).split(':')[1]) if ':' in str(device) else 0

    try:
        monitor = ZeusMonitor(gpu_indices=[gpu_idx])

        # Warmup phase (10 batches)
        print("Warmup phase...")
        warmup_count = 0
        for x, y in dataloader:
            if warmup_count >= 10:
                break
            x = x.to(device).float()
            _ = model(x)
            warmup_count += 1

        torch.cuda.synchronize()

        # Measurement phase
        print(f"Measuring energy over {n_batches} batches...")
        monitor.begin_window("inference")

        batch_count = 0
        for x, y in dataloader:
            if batch_count >= n_batches:
                break
            x = x.to(device).float()
            _ = model(x)
            batch_count += 1

        torch.cuda.synchronize()
        measurement = monitor.end_window("inference")

        # Calculate per-batch metrics
        energy_per_batch = measurement.total_energy / batch_count
        time_per_batch_ms = (measurement.time / batch_count) * 1000

        print(f"Total energy: {measurement.total_energy:.4f}J over {batch_count} batches")
        print(f"Energy per batch: {energy_per_batch:.4f}J")

        return energy_per_batch, time_per_batch_ms

    except Exception as e:
        print(f"Error measuring energy: {e}")
        return None, None




def collect_all_metrics(model, dataloader, device, num_batches_energy=50):
    """
    Collect all performance metrics for a model.

    Args:
        model: PyTorch model in eval mode
        dataloader: DataLoader providing real input batches
        device: Device to run the model on
        num_batches_energy: Number of batches for energy measurement

    Returns:
        dict: Dictionary containing all performance metrics:
            - 'params_millions': Number of parameters in millions
            - 'model_size_mb': Model size in megabytes
            - 'gflops': Giga floating point operations
            - 'mmacs': Million multiply-accumulate operations
            - 'energy_per_batch_j': Energy per batch in Joules (None if unavailable)
            - 'latency_ms': Average latency in milliseconds (from energy measurement)
    """
    from utils.helper_function import count_model_parameters

    print("\n" + "="*80)
    print("Collecting Performance Metrics")
    print("="*80)

    # Number of parameters
    print("\n1. Counting parameters...")
    params = count_model_parameters(model)
    params_millions = params / 1e6
    print(f"   Parameters: {params_millions:.3f}M")

    # Model size
    print("\n2. Calculating model size...")
    model_size = get_model_size_mb(model)
    print(f"   Model Size: {model_size:.3f}MB")

    # FLOPs and MACs
    print("\n3. Measuring FLOPs and MACs...")
    gflops, mmacs = measure_flops_macs(model, dataloader, device)
    if gflops is not None:
        print(f"   GFLOPs: {gflops:.3f}")
        print(f"   MMACs: {mmacs:.3f}")
    else:
        print("   Failed to measure FLOPs/MACs")

    # Energy and Latency (measured together)
    print(f"\n4. Measuring energy and latency ({num_batches_energy} batches)...")
    energy, latency = measure_energy(model, dataloader, device, num_batches_energy)
    if energy is not None:
        print(f"   Energy per batch: {energy:.6f}J")
        print(f"   Latency per batch: {latency:.3f}ms")
    else:
        print("   Energy and latency measurement unavailable")
        latency = None

    print("\n" + "="*80)
    print("Performance Metrics Collection Complete")
    print("="*80 + "\n")

    metrics = {
        'params_millions': float(params_millions),
        'model_size_mb': float(model_size),
        'gflops': float(gflops) if gflops is not None else None,
        'mmacs': float(mmacs) if mmacs is not None else None,
        'energy_per_batch_j': float(energy) if energy is not None else None,
        'latency_ms': float(latency) if latency is not None else None
    }

    return metrics
