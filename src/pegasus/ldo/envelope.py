"""LDO compute-envelope guard (MSD-II §II.10, MII-SCALE-01).

Any operation whose dense form exceeds the compute envelope MUST use the
structured (multi-resolution/tiled) form or **refuse** — never silently
subsample. This estimates the LDO's dominant dense allocations and enforces the
``compute.yaml`` envelope (``float32``, ``max_vram_fraction`` of device memory).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

_DTYPE_BYTES = 4  # float32 per compute.yaml
_DEFAULT_DEVICE_BYTES = 6 * 1024**3  # RTX 4050 6 GB fallback


class ScaleExceedsEnvelopeError(RuntimeError):
    """Raised when an LDO run cannot fit the compute envelope (scale_exceeds_compute_envelope)."""


@dataclass(frozen=True)
class ComputeEnvelope:
    max_bytes: int
    max_vram_fraction: float
    device_bytes: int


def load_compute_envelope(config_path: str | Path = "config/compute.yaml", *, device_bytes: int | None = None) -> ComputeEnvelope:
    fraction = 0.80
    try:
        doc = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
        fraction = float(((doc.get("compute") or {}).get("cuda") or {}).get("max_vram_fraction", 0.80))
    except Exception:
        pass
    total = device_bytes if device_bytes is not None else _detect_device_bytes()
    return ComputeEnvelope(max_bytes=int(total * fraction), max_vram_fraction=fraction, device_bytes=total)


def _detect_device_bytes() -> int:
    try:
        import torch

        if torch.cuda.is_available():
            return int(torch.cuda.get_device_properties(0).total_memory)
    except Exception:
        pass
    return _DEFAULT_DEVICE_BYTES


def estimate_ldo_bytes(*, p: int, S: int, T: int, K: int) -> int:
    """Peak dense allocation estimate for one LDO fit.

    Dominant terms: the S×S spatial operators (Q, Q_half, eigenvectors ≈ 3 S²),
    the lag-extended sample matrix (S·T × p(K+1)), and the extended covariance /
    precision (p(K+1))². All in float32.
    """
    pk = p * (K + 1)
    spatial = 3 * S * S
    samples = max(0, S * (T - K)) * pk
    cov = 3 * pk * pk
    return int((spatial + samples + cov) * _DTYPE_BYTES)


def assert_within_envelope(*, p: int, S: int, T: int, K: int, envelope: ComputeEnvelope | None = None) -> int:
    """Return the estimated bytes, or refuse if the dense form exceeds the envelope."""
    envelope = envelope or load_compute_envelope()
    estimate = estimate_ldo_bytes(p=p, S=S, T=T, K=K)
    if estimate > envelope.max_bytes:
        raise ScaleExceedsEnvelopeError(
            "scale_exceeds_compute_envelope: estimated "
            f"{estimate / 1024**3:.2f} GB dense LDO allocation exceeds the "
            f"{envelope.max_bytes / 1024**3:.2f} GB envelope "
            f"(p={p}, S={S}, T={T}, K={K}); use multi-resolution/tiling (§II.7)."
        )
    return estimate


__all__ = [
    "ComputeEnvelope",
    "ScaleExceedsEnvelopeError",
    "load_compute_envelope",
    "estimate_ldo_bytes",
    "assert_within_envelope",
]
