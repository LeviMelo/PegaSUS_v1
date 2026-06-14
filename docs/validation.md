# Validation

Run unit and integration tests with `python -m pytest tests/unit tests/integration`. Validate configuration and registries with `python scripts/dev/audits/audit_config_and_registries.py` and run the CBL contract command defined by the repository audit scripts.

Use `pegasus acceptance level3 --run <run>` for the final gate. It validates the exact bundle, autonomous authority, registry hashes, EFG evidence, stage proofs, requested numerical artifacts, source reality, and dashboard read-only behavior.
