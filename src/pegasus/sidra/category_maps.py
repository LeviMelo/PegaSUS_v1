"""
Slice 0 scaffold module: sidra/category_maps.py

This module intentionally contains no domain logic. Future implementation slices
must replace blocked stubs through typed contracts.
"""

from pegasus.core.exceptions import BlockedModuleError


def blocked(*, module: str = "sidra/category_maps.py", reason: str = "slice0_scaffold_only") -> None:
    raise BlockedModuleError(module=module, reason=reason)
