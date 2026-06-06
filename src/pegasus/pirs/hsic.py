"""
Slice 0 scaffold module: pirs/hsic.py

This module intentionally contains no domain logic. Future implementation slices
must replace blocked stubs through typed contracts.
"""

from pegasus.core.exceptions import BlockedModuleError


def blocked(*, module: str = "pirs/hsic.py", reason: str = "slice0_scaffold_only") -> None:
    raise BlockedModuleError(module=module, reason=reason)
