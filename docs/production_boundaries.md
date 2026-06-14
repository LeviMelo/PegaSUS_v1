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
