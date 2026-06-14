"""PyTorch adapter for resolved compute plans."""

from __future__ import annotations

from typing import Any

from pegasus.compute.devices import ComputeDevicePlan
from pegasus.core.exceptions import ComputeBackendError


def torch_runtime(plan: ComputeDevicePlan) -> tuple[Any, Any, Any]:
    try:
        import torch
    except ImportError as exc:
        raise ComputeBackendError("PyTorch numerical backend is unavailable") from exc
    device = torch.device(plan.device)
    dtype = torch.float32 if plan.dtype == "float32" else torch.float64
    return torch, device, dtype
