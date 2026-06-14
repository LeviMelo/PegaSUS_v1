# Production Boundaries

PegaSUS production code must use explicit compiler boundaries rather than
private one-off table or device helpers.

Storage writes in production output, EFG, PIRS, acceptance, and dashboard code
should route through `pegasus.output.table_io`, which delegates physical parquet
operations to `pegasus.storage`.

Numerical device, seed, dtype, and memory policy should route through
`pegasus.compute`.

The EFG core seed and bridge modules are metadata-first. They classify seeds and
bridge opportunities but do not materialize tensors or silently execute numeric
fallbacks.

Legacy fixture bundle writers remain quarantined compatibility surfaces until
parity and deletion are explicitly executed.
## Slice 28Y semantic manifest evidence

The autonomous EFG build boundary records two metadata-only summaries on the
`EFGResult` manifest: `core_seed_summary` and `bridge_plan_summary`. These
summaries classify admitted fields into V_core-like seed roles and identify
cross-field bridge opportunities without materializing tensors or adding new
first-class output-bundle keys.

Slice 28ZA extends the storage-boundary adoption to the SIDRA denominator anchor. The module may still use row-level Python transformations, but Parquet reads/writes are routed through output.table_io and pegasus.storage.

## Slice 28ZB — compute RNG boundary closure

HSIC random-feature and Nyström generators must be created through
`pegasus.compute.random.torch_generator`. Domain modules may pass the
returned generator to PyTorch operations, but may not construct and seed
`torch.Generator` objects locally. The `audit_slice28zb_compute_rng_boundary.py`
gate scans production code for direct generator seeding outside the
central compute boundary.

## Slice 28ZC — boundary-audit hardening

Slice 28ZC converts the transitional Slice 28X boundary inventory into a strict regression gate. Direct Parquet/Polars storage I/O is allowed only inside the storage adapter layer or the canonical `output.table_io` row/table boundary. Direct seeded `torch.Generator.manual_seed()` use is allowed only through `compute.random`. The aggregate `audit_boundary_closure.py` audit composes the 28X/28Z/28ZA/28ZB gates and must remain clean before further runtime-authority work proceeds.

## Slice 28ZC repair — scoped and terse boundary closure

The Slice 28ZC hardening gate is intentionally scoped to production modules that were explicitly migrated to the canonical `output.table_io` and `compute.random` boundaries in Slices 28Z through 28ZB. It is not a repository-wide ban on Polars/Arrow inside unfinished SHE/adaptor surfaces. Audit output must remain bounded: violations are summarized and truncated instead of dumping the full scanner result into the terminal.

## Slice 28ZC repair — audit import API compatibility

`audit_slice28x_production_boundaries.py` exposes `run_audit()` as its stable import API. CLI execution must call this function rather than duplicating payload construction inside `main()`. This preserves compatibility with earlier Slice 28X integration tests while keeping the stricter scoped boundary policy introduced in Slice 28ZC.
