# PegaSUS data lifecycle & object model (crystallized 2026-07-09)

The design law for how data enters, is built into objects, and is queried — enforced in code, not
convention. Written after a three-front reconnaissance of the live code + filesystem (the national run
would stress exactly these seams). Authority for roles/GC is `pegasus.core.data_lifecycle`; this doc
is the human-readable companion.

## The object model — three tiers + a query layer

```
CANONICAL INPUT LAKE          DERIVED (rebuildable)            VERSIONED ASSET            RUN BUNDLE (per-run)
raw/datasus (FTP→R,raw) ──▶ processed/datasus (raw cols) ──▶                             runs/{run_id}/ (17-key)
                            normalized/datasus (SHE cols) ─┐                             ├─ identity: UserIntent +
SIDRA cache/metadata ─────▶ processed/sidra/facts ─────────┼─▶ assets/population_tensor/ │   ReproducibilityManifest
                                                           │   v{year}.{seq}             ├─ Tables/efg_tensors/
                                                           │   build-once, scope-guarded │   {field_id}.parquet
                                                           │   lazy slice-on-query ──────┼─▶  (materialized fields)
                                                           └──────────────────────────────┘
   discovery: RunCatalog (output/run_catalog)      query: output/query — edge ✓ · count/raw_field ✓ · rate ✗ (scoped)
```

- **Canonical input** — externally fetched; never auto-GC (re-fetch is costly).
- **Derived** — rebuildable from canonical; safe to drop. (`processed` = R-bridge output, still raw
  DATASUS columns; `normalized` = SHE-canonical decode. The names are legacy; behavior is correct.)
- **Asset** — versioned, immutable, build-once foundational tensors, **sliced** (never rebuilt) per
  query. Complete + scope-guarded in code; **empty on disk only because no national compile has
  completed** — the national run is its first exercise.
- **Run bundle** — the 17-key per-run output under `data/runs` (the *only* production run root;
  `actual_state_panels`/`actual_smokes` are dev/test snapshots, never written by `src`).

## The persistence contract (`core.data_lifecycle.DATA_LIFECYCLE`)

| Root | Role | GC policy | TTL |
|---|---|---|---|
| `data/raw` | canonical_input | never | — |
| `data/processed`, `data/normalized`, `data/sidra` | derived | rebuild_if_missing | — |
| `data/assets` | asset | user_gated | — |
| `data/cache` | cache | drop_by_mtime | 30d |
| `data/diagnostics` | metadata | drop_by_mtime | 90d |
| `data/metadata`, `data/manifests` | metadata | never | — |
| `data/runs` | durable_output | user_gated | — |
| `data/intermediate` | ephemeral | delete_always | — |
| `data/actual_state_panels`, `data/actual_smokes` | test_fixture | user_gated | — |

`classify(path)` answers any path's role; `is_safe_to_delete` is True only for cache/ephemeral;
`storage_gc.gc_by_contract` enforces exactly those two (never touches inputs, assets, outputs,
metadata, fixtures). Ungoverned paths (e.g. the stray `data/_peryear_probe`) classify to `None`.

## Honestly-ranked remaining gaps (national-scale leverage)

1. **`rate`/`standardized_rate` queries** — the field reader + denominator registry are wired, but
   mapping a denominator **id → its materialized bundle field** + cell-key-matched division (and
   age-standardization weights) is the correctness-critical next piece. Currently a typed refusal
   rather than a divide-by-guess. **This is the genuine remaining query-object work.**
2. **Strays** — `data/_peryear_probe` (ungoverned), `data/sidra` legacy strata (superseded by assets).

### Refuted by measurement (a plausible reconnaissance claim that did not survive a probe)

- **"Bundle built fully in-RAM is a national-scale RAM cliff" — FALSE.** Measured on a real bundle:
  `Q_tensor`/`V_fields`/`VariableDictionary`/`Hypotheses` are **one row per FIELD/EDGE** (~73–86 rows,
  0.30 MB total), not per cell. `Q_tensor` carries per-field aggregate diagnostics (n_events, n_eff,
  cov_S…), so it scales with the field count (bounded even nationally), not the cell count. The
  genuinely large per-cell data lives in `Tables/efg_tensors/` and is **already streamed** separately
  (the materialization path + lazy `slice_population_tensor`), never accumulated in `bundle_manager`'s
  `list[dict]`. The eager edge-query load was likewise O(edges) ≈ O(fields²); made lazy anyway as
  cheap hygiene. **No streaming-flush rewrite is needed** — the in-RAM accumulation is O(n_fields).
- (`schema_seed` vs `validate` drift — guarded with a consistency test, currently consistent.)

## What was crystallized/robustified (2026-07-09)

- `core/data_lifecycle.py` — the contract above, in code; `paths.py` reconciled to it.
- `storage_gc.gc_by_contract` — policy-driven cache-TTL + ephemeral reclamation (dry-run default).
- `output/run_catalog.py` — discover runs by scope/time/seed/stage (fixes "no run index").
- `output/query/field_tensor.py` + engine — `count`/`raw_field` queries now serve real values,
  lazily; typed refusals for unknown/ambiguous quantities and missing filter columns.
