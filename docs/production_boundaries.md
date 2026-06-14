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
