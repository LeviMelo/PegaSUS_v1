"""Shared numerical kernel sizing helpers."""

from __future__ import annotations


def tensor_nbytes(shape: tuple[int, ...], *, dtype: str = "float64", copies: int = 1) -> int:
    if any(dimension < 0 for dimension in shape) or copies < 1:
        raise ValueError("tensor dimensions must be nonnegative and copies positive")
    itemsize = {"float32": 4, "float64": 8}.get(dtype)
    if itemsize is None:
        raise ValueError(f"unsupported tensor dtype: {dtype}")
    elements = 1
    for dimension in shape:
        elements *= dimension
    return elements * itemsize * copies
