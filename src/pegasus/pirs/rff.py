from __future__ import annotations

from pegasus.core.exceptions import BlockedModuleError


def blocked(*, module: str = "pirs/rff.py", reason: str = "slice9_hsic_not_active_in_slice8a") -> None:
    raise BlockedModuleError(module=module, reason=reason)
