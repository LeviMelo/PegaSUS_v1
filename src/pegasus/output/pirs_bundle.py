from __future__ import annotations

class RetiredLegacyBundleError(RuntimeError):
    pass


def _retired(*args, **kwargs):
    raise RetiredLegacyBundleError(
        "This legacy first-class output writer is retired. "
        "Use autonomous EFG/PIRS/HSIC execution plus OutputBundleManager Phase-E serialization."
    )


build_pirs_output_bundle = _retired
write_pirs_bundle = _retired
attach_pirs_to_run = _retired

__all__ = ['build_pirs_output_bundle', 'write_pirs_bundle', 'attach_pirs_to_run', 'RetiredLegacyBundleError']
