from __future__ import annotations

class RetiredLegacyBundleError(RuntimeError):
    pass


def _retired(*args, **kwargs):
    raise RetiredLegacyBundleError(
        "This legacy first-class output writer is retired. "
        "Use autonomous EFG/PIRS/HSIC execution plus OutputBundleManager Phase-E serialization."
    )


build_sidra_stdfm_output_bundle = _retired
write_sidra_stdfm_bundle = _retired
attach_sidra_stdfm_to_run = _retired

__all__ = ['build_sidra_stdfm_output_bundle', 'write_sidra_stdfm_bundle', 'attach_sidra_stdfm_to_run', 'RetiredLegacyBundleError']
