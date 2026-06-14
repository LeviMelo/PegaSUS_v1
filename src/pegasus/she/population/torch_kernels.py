"""Torch helpers for population workloads routed through the compute backend."""

from __future__ import annotations

from typing import Any

from pegasus.compute.devices import ComputeDevicePlan, resolve_torch_device
from pegasus.compute.kernels import tensor_nbytes
from pegasus.compute.torch_backend import torch_runtime


def population_tensor_runtime(
    shape: tuple[int, int, int, int, int],
    *,
    task_id: str = "population_sparse_solver",
    cuda_required: bool = False,
    seed: int = 20260613,
) -> tuple[Any, Any, Any, ComputeDevicePlan]:
    estimated = tensor_nbytes(shape, copies=6)
    plan = resolve_torch_device(
        task_id, cuda_required=cuda_required, prefer_cuda=True,
        seed=seed, estimated_bytes=estimated,
    )
    torch, device, dtype = torch_runtime(plan)
    return torch, device, dtype, plan


def project_nonnegative(torch: Any, values: Any) -> Any:
    return torch.clamp(values, min=0.0)


def squared_residual_sum(torch: Any, observed: Any, expected: Any, weight: float = 1.0) -> Any:
    if weight < 0:
        raise ValueError("population kernel weight must be nonnegative")
    residual = observed - expected
    return float(weight) * torch.sum(residual * residual)
