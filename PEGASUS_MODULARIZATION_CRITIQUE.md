# PegaSUS Modularization / Consolidation / API-Boundary Critique

Objective 2: the codebase has repeatedly proven to carry redundant, orphaned, or badly-architected code — a symptom of scattered logic with no centralized computation, no clear module/API boundaries, and no enforced data contracts, even where the mathematical architecture implies clean, bounded steps. This ledger captures the structural problems (lead's direct survey first; the `modularization` workflow appends full findings). **Full granularity — nothing compacted.**

---

## Part A — Lead's direct structural survey

### M1 · `MOD-DUP-MORAN` — HIGH — Moran's I / spatial autocorrelation is computed SIX different ways
- **detail:** Spatial autocorrelation (Moran's I) — a single well-defined statistic — is reimplemented across the tree: `q_tensor._moran_contiguity` (1-D cell-order chain, the garbage proxy of A20), `_moran_i_adjacency`, `_moran_i_for_group`, `_moran_diagnostic`, `_moran_proxy`, plus `_moran_corrected_n_eff`. At least 4 distinct estimators of the SAME quantity, some geography-aware and some not, with no shared implementation or contract. This is the clearest single example of scattered, uncentralized mathematical logic, and it directly produces a validity bug (A20: the effective-n correction uses the non-geographic one).
- **evidence:** `grep -roE "def _moran[a-z_]*"` → `_moran_proxy, _moran_i_for_group, _moran_i_adjacency, _moran_diagnostic, _moran_corrected_n_eff, _moran_contiguity`.
- **recommendation:** ONE `pegasus.geo.spatial_autocorrelation` module: `moran_i(values, weight_graph)` taking the canonical `SpatialWeightGraph`, plus `effective_n(values, graph)`. Every consumer (q-tensor, LDO SE, diagnostics) calls it. Deletes 5 near-duplicates and fixes A20/A21.

### M2 · `MOD-DUP-CLEAN` — MEDIUM — eight string-cleaning / hashing / table-reading helpers reimplemented
- **detail:** `_clean` ×4 (+ `_clean_text`, `_clean_str`, `_clean_or_none`, `_clean_code`), `_digits` ×4, `_stable_hash` ×4, `_read_table` ×5 — the same primitive utilities copy-pasted across `datasus/normalize/*`, `sidra/*`, and others. Each divergent copy is a latent silent-failure site (one `_clean` strips differently than another → inconsistent decode). The earlier modules report (T2.5) flagged this at the normalizer level; it is broader.
- **evidence:** duplication scan above.
- **recommendation:** a single `pegasus.core.text` (clean/digits/stable_hash) + `pegasus.core.io` (`read_table`) that all modules import; delete the copies. Pin one canonical behavior with a contract test.

### M3 · `MOD-LAYER-GEO` — MEDIUM — geo (foundational) imports denominators (higher layer): a layering inversion
- **detail:** `geo/migration_affinity.py:33` imports `denominators.population.migration.MigrationFlowReconstruction`. Geometry/spatial primitives should be a FOUNDATION that denominators depend on — not the reverse. This inverts the dependency layering and risks an import cycle (denominators already depends on geo).
- **evidence:** `geo/migration_affinity.py:33`.
- **recommendation:** move `migration_affinity` into `denominators` (it is a denominator-domain artifact) or invert so the migration reconstruction depends on a geo-level affinity primitive; enforce a layering rule (geo/core/measurement never import efg/ldo/denominators/workflows).

### M4 · `MOD-DUP-ILR` — LOW/MEDIUM — ILR composition transform reimplemented (`_ilr_np`, `_ilr_coords`, `_ilr_basis`, `_ilr_like_residual`)
- **detail:** The isometric-log-ratio transform for compositional data (race shares) has multiple implementations across reconstruction/measurement. `_ilr_like_residual` (a separate "like" variant) suggests divergent behavior.
- **recommendation:** one `pegasus.measurement.composition` (ILR basis + coords + residual); single source for the race-composition math.

### M5 · `MOD-DUP-TOF64` — LOW — `_to_f64_array` duplicated across `denominators/reconstruction` and `she/reconstruction`
- **detail:** Two parallel reconstruction schema modules (denominators vs she) each define the array-normalization + problem dataclass. The earlier notes flagged `she/reconstruction` vs `denominators/reconstruction` as parallel; this is a duplicated core contract (and M2's float32 change had to be reasoned about in both).
- **recommendation:** determine which reconstruction package is canonical and collapse; a single `PopulationTensorProblem` contract.

### M6 · `MOD-CONTRACT-RELIABILITY` — HIGH — no data contract binding "uncertainty produced" (EFG/§3.12) to "uncertainty consumed" (LDO)
- **detail:** The systemic finding from the math ledger (A1/A13/A20/A22): the EFG layer emits a rich per-field/per-cell reliability state and the LDO ignores it, because there is no typed, enforced interface requiring the estimator to consume it. This is the archetypal missing module boundary — two subsystems that MUST share a contract, communicating through an ignored tensor.
- **recommendation:** a first-class `ObservationReliability` contract (per-cell weight + per-field effective-n + fragility) that `assemble_ldo_tensor` produces and the covariance/SE estimators are REQUIRED to consume, with a contract test asserting a low-reliability cell measurably changes the fit. Both a validity fix (math ledger) and the exemplar of the modularization redesign.

*(Direct survey continues; the `modularization` workflow's full findings — orphan scan, megazord decomposition, API-surface audit — append to Part B.)*

## Part B — `modularization` workflow findings (full, verbatim)
*(pending)*
