"""Live hardware headroom → a memory-budgeted size (CLAUDE.md §V: auto-determine from the data/problem
and the machine, not a static constant).

This is the ONE place that reads available RAM/VRAM and turns it into a chunk size or worker count, so
every stage sizes to the SAME safe-headroom policy instead of its own ad-hoc constant (the normalize's
row-batch parallelism, the population tensor's locality-block, a future compile/LDO batch). It centralizes
the *sizing policy* deliberately — NOT the chunking mechanisms, which are legitimately domain-specific (a
row-batch decode and a locality-block solve stay their own code; forcing a common interface would be
worse). Built on the maintained primitives (`psutil` for RAM, `torch.cuda.mem_get_info` for VRAM) rather
than a heavyweight framework — the workload (R subprocess + polars + torch) doesn't fit Dask/Ray without a
migration whose cost dwarfs the payoff.

Everything is FLOORED (a loaded box still works — never sizes below a safe default) and CAPPED (an
over-estimate can't request an absurd size); and it reads LIVE state each call, so a re-run on a busy
machine backs off automatically. It sizes only — it does not gate: a stage that over-sizes anyway must
still degrade gracefully (fallback / spill), because these estimates are approximate by construction.
"""

from __future__ import annotations

_GB = 1024 ** 3


def available_ram_bytes() -> int:
    """System-available RAM (reclaimable cache counts as available). Conservative fallback if psutil is
    absent so a size is always computable."""
    try:
        import psutil

        return int(psutil.virtual_memory().available)
    except Exception:
        return 8 * _GB


def available_vram_bytes() -> int:
    """Free CUDA VRAM on the default device, or 0 if there is no usable CUDA device (CPU path). Never
    raises — a device probe failure reads as 'no VRAM budget', which the caller treats as CPU-only."""
    try:
        import torch

        if torch.cuda.is_available():
            free, _total = torch.cuda.mem_get_info()
            return int(free)
    except Exception:
        pass
    return 0


def memory_budgeted_count(
    *,
    ram_bytes_per_unit: float,
    floor: int,
    cap: int,
    headroom_frac: float = 0.5,
    reserve_bytes: int = 4 * _GB,
    vram_bytes_per_unit: float = 0.0,
) -> int:
    """Largest unit count whose working set fits a safe fraction of the LIVE available RAM — and free
    VRAM too, when ``vram_bytes_per_unit > 0`` (the unit lives on-device).

    ``ram_bytes_per_unit`` is the estimated RAM working set of ONE unit (one row-batch, one block's
    cells, one parallel worker). ``headroom_frac`` is the fraction of usable memory to actually spend
    (the rest is slack for other allocations + estimate error). ``reserve_bytes`` is held back off the
    top (OS + the process's fixed footprint). The result is clamped to ``[floor, cap]`` — ``floor`` is
    the historical safe default (so this can only ever *widen* from it), ``cap`` bounds an over-estimate.
    """
    usable_ram = max(0, available_ram_bytes() - reserve_bytes)
    count = int(usable_ram * headroom_frac / max(1.0, ram_bytes_per_unit))
    if vram_bytes_per_unit > 0:
        vram = available_vram_bytes()
        if vram > 0:
            count = min(count, int(vram * headroom_frac / max(1.0, vram_bytes_per_unit)))
    return max(floor, min(cap, count))


__all__ = ["available_ram_bytes", "available_vram_bytes", "memory_budgeted_count"]
