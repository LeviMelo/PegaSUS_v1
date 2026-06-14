# Level 3 Acceptance

`pegasus acceptance level3 --run <run>` returns `failed`, `fixture_validated`, or `production_candidate`. The manifest contains 23 named checks, blocking reasons, warnings, stage statuses, and artifact evidence.

Fixture validation proves structural and numerical integrity without claiming external provenance. Production candidacy additionally requires a validated `materialized_external` source manifest, registry-backed substrate, autonomous EFG evidence, complete stage proofs, and every requested population, ST-DFM, PIRS, and HSIC artifact or an explicit failed/blocked terminal state.
