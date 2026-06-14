from __future__ import annotations

import random

import pytest

from pegasus.compute.devices import resolve_torch_device
from pegasus.compute.kernels import tensor_nbytes
from pegasus.compute.memory import preflight_memory
from pegasus.compute.random import seed_everything
from pegasus.core.exceptions import CUDARequiredError, MemoryPreflightError


COMPUTE_CONFIG = {
    "compute": {
        "cpu_dtype": "float64",
        "max_ram_fraction": 0.8,
        "cuda": {
            "enabled_for": ["stdfm_solver", "pirs_hsic_exact"],
            "dtype": "float32",
            "max_vram_fraction": 0.8,
        },
    }
}


def test_compute_backend_cpu_policy_and_dtype() -> None:
    plan = resolve_torch_device(
        "stdfm_solver",
        COMPUTE_CONFIG,
        cuda_required=False,
        prefer_cuda=True,
        cuda_available_override=False,
        seed=7,
        estimated_bytes=1024,
    )
    assert plan.device == "cpu"
    assert plan.dtype == "float64"
    assert plan.cpu_fallback_allowed is True
    assert plan.memory_preflight is not None and plan.memory_preflight.ok


def test_compute_backend_cuda_required_is_typed_abort() -> None:
    with pytest.raises(CUDARequiredError):
        resolve_torch_device(
            "pirs_hsic_exact",
            COMPUTE_CONFIG,
            cuda_required=True,
            cuda_available_override=False,
        )


def test_seed_policy_is_reproducible() -> None:
    seed_everything(123)
    first = [random.random() for _ in range(3)]
    seed_everything(123)
    assert [random.random() for _ in range(3)] == first


def test_memory_preflight_and_kernel_size() -> None:
    assert tensor_nbytes((10, 20), dtype="float32", copies=2) == 1600
    with pytest.raises(MemoryPreflightError):
        preflight_memory(81, available_bytes=100, max_fraction=0.8)
