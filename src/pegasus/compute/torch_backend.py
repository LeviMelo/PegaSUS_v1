from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pegasus.core.exceptions import ConfigurationError


@dataclass(frozen=True)
class TorchStatus:
    installed: bool
    cuda_available: bool
    torch_version: str | None
    cuda_version: str | None
    device_name: str | None
    error: str | None = None


def inspect_torch() -> TorchStatus:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - depends on local env
        return TorchStatus(
            installed=False,
            cuda_available=False,
            torch_version=None,
            cuda_version=None,
            device_name=None,
            error=str(exc),
        )

    cuda_available = bool(torch.cuda.is_available())
    device_name = torch.cuda.get_device_name(0) if cuda_available else None

    return TorchStatus(
        installed=True,
        cuda_available=cuda_available,
        torch_version=torch.__version__,
        cuda_version=torch.version.cuda,
        device_name=device_name,
        error=None,
    )


class TorchBackend:
    def __init__(
        self,
        require_cuda: bool = False,
        dtype: Literal["float32", "float64"] = "float32",
    ) -> None:
        import torch

        self.torch = torch
        self.dtype_name = dtype
        self.dtype = torch.float32 if dtype == "float32" else torch.float64

        if require_cuda and not torch.cuda.is_available():
            raise ConfigurationError("CUDA is required for this module but is unavailable.")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def tensor(self, data):
        return self.torch.as_tensor(data, dtype=self.dtype, device=self.device)

    def to_numpy(self, tensor):
        return tensor.detach().cpu().numpy()

    def memory_report(self) -> dict[str, int | str]:
        if self.device.type != "cuda":
            return {"device": "cpu"}

        return {
            "device": str(self.device),
            "allocated": int(self.torch.cuda.memory_allocated()),
            "reserved": int(self.torch.cuda.memory_reserved()),
            "max_allocated": int(self.torch.cuda.max_memory_allocated()),
        }