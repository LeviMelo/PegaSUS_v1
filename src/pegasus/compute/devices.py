"""Central compute device, dtype, seed, and memory policy."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from pegasus.compute.memory import MemoryPreflight, preflight_memory
from pegasus.compute.random import seed_everything
from pegasus.core.config import load_yaml
from pegasus.core.exceptions import ComputeBackendError, CUDARequiredError


@dataclass(frozen=True)
class ComputeDevicePlan:
    task_id: str
    device: str
    device_type: str
    dtype: str
    cuda_required: bool
    cuda_available: bool
    cpu_fallback_allowed: bool
    seed: int
    memory_preflight: MemoryPreflight | None

    def as_manifest(self) -> dict[str, Any]:
        payload = dict(vars(self))
        payload["memory_preflight"] = self.memory_preflight.as_manifest() if self.memory_preflight else None
        return payload


def _compute_config(config: Mapping[str, Any] | str | Path | None) -> dict[str, Any]:
    if config is None:
        payload = load_yaml("config/compute.yaml")
    elif isinstance(config, (str, Path)):
        payload = load_yaml(config)
    else:
        payload = dict(config)
    compute = payload.get("compute", payload)
    if not isinstance(compute, dict):
        raise ComputeBackendError("compute configuration must be a mapping")
    return compute


def resolve_torch_device(
    task_id: str,
    config: Mapping[str, Any] | str | Path | None = None,
    cuda_required: bool | None = None,
    *,
    prefer_cuda: bool = False,
    seed: int = 1729,
    estimated_bytes: int | None = None,
    cuda_available_override: bool | None = None,
) -> ComputeDevicePlan:
    try:
        import torch
    except ImportError as exc:
        raise ComputeBackendError("PyTorch numerical backend is unavailable") from exc
    compute = _compute_config(config)
    cuda = compute.get("cuda", {}) if isinstance(compute.get("cuda", {}), dict) else {}
    enabled_for = {str(value) for value in cuda.get("enabled_for", [])}
    required = bool(cuda_required) if cuda_required is not None else False
    available = bool(torch.cuda.is_available()) if cuda_available_override is None else bool(cuda_available_override)
    if required and not available:
        raise CUDARequiredError(f"CUDA is required for task {task_id!r} but unavailable")
    use_cuda = available and (required or (prefer_cuda and task_id in enabled_for))
    dtype_name = str(cuda.get("dtype", "float32") if use_cuda else compute.get("cpu_dtype", "float64"))
    if dtype_name not in {"float32", "float64"}:
        raise ComputeBackendError(f"Unsupported compute dtype: {dtype_name}")
    memory = None
    if estimated_bytes is not None:
        available_bytes = None
        memory_kind = "vram" if use_cuda else "ram"
        if use_cuda:
            free_bytes, _ = torch.cuda.mem_get_info()
            available_bytes = int(free_bytes)
        memory = preflight_memory(
            estimated_bytes,
            available_bytes=available_bytes,
            max_fraction=float(cuda.get("max_vram_fraction", 0.8)) if use_cuda else float(compute.get("max_ram_fraction", 0.8)),
            memory_kind=memory_kind,
        )
    seed_everything(seed, torch_module=torch)
    return ComputeDevicePlan(
        task_id=task_id,
        device="cuda" if use_cuda else "cpu",
        device_type="cuda" if use_cuda else "cpu",
        dtype=dtype_name,
        cuda_required=required,
        cuda_available=available,
        cpu_fallback_allowed=not required,
        seed=seed,
        memory_preflight=memory,
    )
