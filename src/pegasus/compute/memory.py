"""RAM/VRAM preflight for numerical tasks."""

from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass
from typing import Any

from pegasus.core.exceptions import MemoryPreflightError


@dataclass(frozen=True)
class MemoryPreflight:
    estimated_bytes: int
    available_bytes: int | None
    permitted_bytes: int | None
    fraction: float
    ok: bool
    memory_kind: str

    def as_manifest(self) -> dict[str, Any]:
        return dict(vars(self))


def available_ram_bytes() -> int | None:
    try:
        import psutil

        return int(psutil.virtual_memory().available)
    except ImportError:
        pass
    if os.name == "nt":
        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("length", ctypes.c_ulong),
                ("memory_load", ctypes.c_ulong),
                ("total_phys", ctypes.c_ulonglong),
                ("avail_phys", ctypes.c_ulonglong),
                ("total_page_file", ctypes.c_ulonglong),
                ("avail_page_file", ctypes.c_ulonglong),
                ("total_virtual", ctypes.c_ulonglong),
                ("avail_virtual", ctypes.c_ulonglong),
                ("avail_extended_virtual", ctypes.c_ulonglong),
            ]
        status = MemoryStatus()
        status.length = ctypes.sizeof(MemoryStatus)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.avail_phys)
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        pages = os.sysconf("SC_AVPHYS_PAGES")
        return int(page_size * pages)
    except (AttributeError, OSError, ValueError):
        return None


def preflight_memory(
    estimated_bytes: int,
    *,
    available_bytes: int | None = None,
    max_fraction: float = 0.8,
    memory_kind: str = "ram",
    raise_on_failure: bool = True,
) -> MemoryPreflight:
    if estimated_bytes < 0:
        raise ValueError("estimated_bytes must be nonnegative")
    if not 0 < max_fraction <= 1:
        raise ValueError("max_fraction must be in (0, 1]")
    available = available_ram_bytes() if available_bytes is None and memory_kind == "ram" else available_bytes
    permitted = None if available is None else int(available * max_fraction)
    ok = permitted is None or estimated_bytes <= permitted
    result = MemoryPreflight(estimated_bytes, available, permitted, max_fraction, ok, memory_kind)
    if not ok and raise_on_failure:
        raise MemoryPreflightError(
            f"{memory_kind} preflight failed: estimated={estimated_bytes} permitted={permitted}"
        )
    return result
