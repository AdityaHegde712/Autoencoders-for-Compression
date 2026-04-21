"""Pick accelerator and sync for timing. Prefer MPS (Apple Silicon), then CUDA, then CPU."""

from __future__ import annotations

import torch
import torch.nn as nn


def get_device() -> torch.device:
    """
    Best device for convolution-heavy PyTorch models.
    Order: MPS (Apple Silicon) > CUDA > CPU.
    """
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
        return torch.device("cuda")
    return torch.device("cpu")


def maybe_compile(module: nn.Module, device: torch.device) -> nn.Module:
    """
    torch.compile usually helps on CUDA; on MPS/CPU it often adds compile latency
    or is unsupported for parts of the graph, so keep modules eager there.
    """
    if device.type != "cuda":
        return module
    return torch.compile(module)


def synchronize(device: torch.device) -> None:
    """Wait for queued GPU/MPS work to finish (useful for benchmarks)."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()
