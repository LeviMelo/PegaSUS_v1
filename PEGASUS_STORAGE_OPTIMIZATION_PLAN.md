# PegaSUS Storage Optimization Plan

> ## ✅ IMPLEMENTED (2026-07-05)
> Tiers 1–2 + the manifest fix are shipped and verified. **`data/` 53 GB → 26 GB**; all 308 unit/contract tests green; national C25 compile + national pop-tensor re-validated OK post-migration.
> - **DATASUS v4 bridge** (commit `83914ac`): drops `raw.rds`, gates the microdatasus sidecar off (also skips the super-linear `process_*` CPU), writes `processed.parquet` with ZSTD; cache-hit rekeyed onto `processed.parquet` + `manifest.json` (accepts legacy v3); ancillary pruned on success. GC reclaimed **30.2 GiB** (`data/raw/datasus` 26 GB → 592 MB).
> - **SIDRA slim dumps** (commit `6245664`): the duplicated response payload is no longer re-stored (it lives in the client cache); facts written ZSTD; GC reclaimed **2.2 GiB** (`data/sidra` 2.3 GB → 30 MB).
> - **Run-bundle manifest** (commit `61746bb`): `tensor_values`/`migration_values` no longer inlined into `ReproducibilityManifest` — **430 MB → 150 KB** at national scale (~2,900×).
> - **Column fidelity preserved throughout** — every reclaim is a redundant *serialization* or duplicate archive; no column or row of source data was dropped.
> - Remaining (optional): Tier-3 Hive-partitioned lazy views (W7); stale dev-output GC (`actual_state_panels` 1.7 GB, `diagnostics` 0.44 GB — user's run outputs, left for their call).

**Status:** ✅ core implemented & verified (original analysis below)
**Trigger:** `data/` reached 53 GB on a single national all-source acquisition and was still growing linearly. Before scaling PegaSUS further, the storage layer must be made compact and bounded.
**Bottom line:** ~**70–80 % of `data/` is redundant or reclaimable**. Target end state ~**8–12 GB** for the same analytical content, with per-`(system, UF, year)` footprint bounded so national scaling stays linear.

---

## 1. Measured current state

Top-level `data/` (measured 2026-07-05):

| Dir | Size | What it is |
|---|---:|---|
| `raw/datasus/` | **26 GB** | Per-chunk `raw.rds` + `microdatasus_processed.parquet` |
| `processed/datasus/` | **14 GB** | Per-chunk `processed.parquet` (the consumed layer) |
| `sidra/` | 2.3 GB | SIDRA facts + raw JSON responses |
| `cache/sidra/` | 2.1 GB | SIDRA response cache (duplicates `sidra/`) |
| `actual_state_panels/` | 1.7 GB | Stale Alagoas dev-run outputs |
| `normalized/` | 1.4 GB | Canonical + national-combined parquet |
| `runs/` | 0.7 GB | Compile bundles (incl. 450 MB manifest bug) |
| `diagnostics/` | 0.44 GB | Dev diagnostic dumps |
| `cache/datasus/`, `manifests/`, `metadata/` | ~0.2 GB | Request manifests, metadata |

Per-system DATASUS (raw + processed): **SIH-RD 32 GB**, SINASC 5 GB, SIM-DO 4 GB, CNES ~0. SIH (monthly hospitalization microdata, ~250–300 M records over 2000–2024) dominates and is the least relevant system to the pancreatic-cancer *mortality* win-condition (that comes from SIM).

### Per-chunk anatomy (the core problem)

One SIH chunk — `SIH-RD/uf=SP/period=2000_01` — stores the **same ~250 K rows three times**:

| File | Size | Format | Consumed by anything? |
|---|---:|---|---|
| `raw/…/raw.rds` | 6.15 MB | R-native RDS | **No** — `readRDS` appears nowhere in the codebase |
| `raw/…/microdatasus_processed.parquet` | 6.43 MB | parquet (microdatasus semantic labels) | Legacy path being retired (in-house codebook is now authoritative) |
| `processed/…/processed.parquet` | 6.37 MB | parquet (raw-coded, UTF-8) | **Yes** — the only artifact the normalizer reads |
| `manifest.json` | 6.3 KB | json | cache-hit + provenance |
| `heartbeat.json`, `stdout.log`, `stderr.log` | ~1 KB | debug | ephemeral |

Source: `src/pegasus/datasus/r_scripts/fetch_process_microdatasus.R` lines 347 (`saveRDS`), 352 (`processed.parquet`), 369 (`microdatasus_processed.parquet`). The downloaded `.dbc` is already deleted after parsing (line 221), so there is **no** original-source archive being kept — the 26 GB "raw" is two *re-serializations* of data that already exists as `processed.parquet`.

### Formats / codecs

- `processed.parquet` uses **SNAPPY** and keeps **all** DBF columns (114 SIH / 88 SIM). **Keeping all columns is correct** — this is the full-fidelity, raw-coded archival copy the in-house codebook translates; the canonical layer selects the ~62 it currently binds, but the source must retain everything so a future concept binding never forces a re-fetch. The only inefficiency here is the **codec**, not the column count.
- The national-combined canonical I write uses **ZSTD** (polars default) — already ~2× denser per row than the SNAPPY chunks, at the *same* columns.
- **No `write_parquet` call in the codebase sets a compression codec or level** — every layer rides on library defaults (R/arrow → SNAPPY; polars → ZSTD-default). Nothing is tuned. Recompressing SNAPPY→ZSTD is lossless and drops nothing.

### File-count overhead

`raw/datasus` alone holds **11,162 `heartbeat.json` + 22,410 `stdout/stderr` logs + 10,701 `manifest.json` ≈ 44 K files**. At a 4 KB NTFS cluster, that's ~180 MB of pure slack before content, plus real inode/scan cost. Growing ~6 files per chunk.

---

## 2. Root inefficiencies

| # | Inefficiency | Evidence | Reclaimable |
|---|---|---|---:|
| **I1** | `raw.rds` written every chunk, never read back | no `readRDS` in repo; only existence-checked | **~13 GB** |
| **I2** | `microdatasus_processed.parquet` written every chunk; legacy semantic path superseded by in-house codebook | R line 369; memory `datasus-codebook-registry` (retiring microdatasus) | **~13 GB** |
| **I3** | `processed.parquet` uses the **SNAPPY** codec (not ZSTD) | untuned codec (R/arrow default) | **~4–5 GB** (of 14, codec only — columns kept) |
| **I4** | Downstream re-materialization: `processed → datasus_combined → normalized/canonical → normalized/national` — each a full copy | `pipeline.py` L265/269/983 | grows per run |
| **I5** | 44 K ancillary debug files kept after success | `stdout.log`/`stderr.log`/`heartbeat.json` per chunk | file-count + slack |
| **I6** | SIDRA response cache duplicates facts | `cache/sidra` 2.1 GB vs `sidra/` 2.3 GB | **~2 GB** |
| **I7** | Stale dev outputs | `actual_state_panels` 1.7 GB + `diagnostics` 0.44 GB | **~2 GB** |
| **I8** | Run-bundle `ReproducibilityManifest` inlines tensors as raw floats (`migration_values`, 22.5 M floats) instead of referencing a parquet path | 450 MB manifest; `migration_flows_path: null` unused | **~0.45 GB/run**; ~11 GB on full-window |

---

## 3. Target storage architecture

**Principle: one compact, portable, content-addressed copy per `(system, UF, year[, month])`; everything else is a view or a hash.**

1. **Raw is ephemeral.** `.dbc` → parse → delete (already done). **Stop writing `raw.rds`.** Provenance is preserved by recording `raw_sha256` in the manifest at fetch time — keep the *hash*, drop the *bytes*.
2. **One consumed artifact per chunk:** `processed.parquet` at **full raw fidelity — all DBF columns, raw codes**. It is the archival source-of-truth the in-house codebook translates; **columns are never pruned** (a future concept binding must not require a re-fetch). The only change is the **codec: ZSTD + dictionary encoding** (lossless). Retire `microdatasus_processed.parquet` (the microdatasus semantic copy — not consumed by the in-house normalizer; see §7).
3. **Downstream layers are lazy, not materialized.** `datasus_combined` / `canonical` / `national` become `scan_parquet` views over a **Hive-partitioned dataset** (`system/uf/year/…`), materialized only when a run genuinely needs a single national file — and then streamed (`sink_parquet`, already used). Removes I4's per-run copies.
4. **Ancillary logs retained only on failure.** On `success`/`cached`, delete `stdout.log`/`stderr.log`/`heartbeat.json`; keep `manifest.json` (cache + provenance). Optional: collapse per-chunk dirs into the partitioned dataset to cut the 44 K-file sprawl.
5. **Run bundles reference, never inline.** Large tensors (population/migration) always written to parquet and referenced by path/hash — `migration_flows_path` already exists for exactly this and is sitting `null`.
6. **SIDRA:** compress/expire the response cache; it is a fetch accelerator, not an archive — it can be dropped entirely once facts are materialized.

---

## 4. Work items (sequenced by ROI / risk)

### Tier 1 — reclaim now, low risk (est. −28 GB)
- **W1 — GC redundant raw layer.** Delete existing `raw.rds` and (after W6 gate) `microdatasus_processed.parquet`; **decouple the cache-hit check** (`subprocess.py:156`) so it validates on `processed.parquet` + `manifest.json`, not `raw_path`. *Risk: low — nothing reads `raw.rds`; normalize reads `processed.parquet`.* **≈ −26 GB.**
- **W2 — Prune debug ancillaries on success.** Delete `stdout/stderr/heartbeat` when status ∈ {success, cached}. **−44 K files.**
- **W3 — GC stale dev outputs.** Remove `actual_state_panels/` + `diagnostics/` (regenerable). **≈ −2 GB.**
- **W4 — Fix run-bundle inline-tensor bloat (I8).** Write `migration_values` to a referenced parquet; already needed before any full-window national run. **−0.45 GB/run.**

### Tier 2 — reduce steady state (est. −8 GB, prevents regrowth)
- **W5 — Stop writing `raw.rds` at the source** (`R` line 347): drop `saveRDS`; keep `raw_sha256` in the manifest. Prevents I1 from ever recurring.
- **W6 — `processed.parquet` SNAPPY → ZSTD + dictionary (lossless; all columns kept).** Set the codec in the R `write_utf8_parquet` (arrow supports `compression = "zstd"`). No column pruning — full raw fidelity retained. **14 GB → ~9–10 GB.** *Risk: low — identical rows/columns, only the codec changes.*
- **W6b — Retire `microdatasus_processed.parquet`** (the microdatasus semantic copy). Not consumed by any normalizer; only `schema_compare` reads it, and that is a dev audit that can run on demand rather than persist the parquet. Covered by W1 at the chunk level; W5's sibling change stops writing it at the source. **≈ −13 GB.**

### Tier 3 — architecture (bounded scaling)
- **W7 — Hive-partitioned canonical lake.** Replace the per-chunk dir tree + `combined`/`canonical` re-materialization with one partitioned parquet dataset per system, read via lazy `scan_parquet`. Removes I4 and most of the 44 K-file sprawl; national "combine" becomes a scan, not a copy.
- **W8 — SIDRA cache dedup/expiry (I6).** Compress or TTL the response cache; drop after facts materialize. **≈ −2 GB.**

---

## 5. Compatibility notes & risks

- **Cache-hit coupling** (`subprocess.py:156`, `acquire/datasus.py:101`) currently requires `raw` *and* `processed` to exist. W1 must rewrite this to key on `processed.parquet` + `manifest.json` only, or every chunk re-fetches. This is the single load-bearing change gating the −26 GB.
- **Provenance:** `raw_sha256` is recorded in the manifest at fetch (`subprocess.py:274`). Dropping the raw bytes keeps the hash → provenance intact, source bytes not.
- **Reprocessing:** normalize reads `processed.parquet`, not raw — so re-deriving the canonical/national layers needs *no* raw and *no* re-fetch. Safe to drop raw.
- **Column pruning (W6):** must be driven by the registry/normalizer schema (a superset union), never a hardcoded list, or a future carrier/disease field silently loses its source column. Add a test asserting the pruned set ⊇ every column any normalizer/substrate spec references.
- **`microdatasus_processed.parquet` retirement:** confirm `datasus/schema_compare.py` and the `microdatasus_probe` diagnostics are dev-audit only before W6b.
- **Reversibility:** raw is re-fetchable from DATASUS FTP; stale panels/diagnostics are re-generable. No optimization here destroys anything not reconstructable from source + code.

---

## 6. Estimated end state

| | Now | After Tier 1 | After Tier 2 | After Tier 3 |
|---|---:|---:|---:|---:|
| DATASUS | 40 GB | 14 GB | 6–8 GB | 6–8 GB |
| SIDRA (+cache) | 4.4 GB | 4.4 GB | 4.4 GB | ~1.5 GB |
| Panels/diagnostics | 2.1 GB | 0 | 0 | 0 |
| Runs/normalized/other | ~6 GB | ~5 GB | ~2 GB | ~2 GB |
| **Total** | **~53 GB** | **~23 GB** | **~13 GB** | **~10 GB** |

And, more important than the one-time reclaim: the **per-`(system, UF, year)` footprint drops ~3×**, so a full 2000–2024 national all-source acquisition lands in the low tens of GB instead of the 100 GB+ the current 3×-serialization trajectory implies.

---

## 7. DATASUS translation architecture & microdatasus retirement

There are **two coexisting translation paths**, and understanding which is load-bearing is what makes the storage cuts safe.

**Path A — microdatasus (the temporary backend, being retired):**
- `fetch_datasus()` downloads the `.dbc` and returns the **raw coded** DBF (all columns). *(Direct `read.dbc(as.is=TRUE)` is the fallback and does the same.)*
- `process_datasus_dispatch()` applies microdatasus's semantic `process_*()` labelling → `microdatasus_processed.parquet`. **This step is lossy** — it renames/drops columns and does the silent `0/9 → NA` collapse.
- **Nothing in the Python pipeline consumes `microdatasus_processed.parquet`.** The only reader is `datasus/schema_compare.py`, a dev audit that profile-diffs raw vs processed to *detect* microdatasus column drops. That's a metadata comparison — it does not require the semantic parquet to be persisted on disk.

**Path B — in-house codebook (the strategic authority, already in production):**
- `normalize/codebook.py` loads `config/registries/datasus/datasus_codebook.yaml` (concepts + per-system bindings + reference-table lookups) plus `reference/*.parquet` (CBO/occupation/naturalidade, mirrored 1:1 from microdatasus's shipped data).
- The normalizers (`normalize/{sim,sih,sinasc,cnes}.py`) read the **raw-coded `processed.parquet`** and apply the codebook vectorially (`categorical_exprs`), with explicit MSD §2.3 states (`missing`/`valid`/`unknown`/`invalid`) instead of microdatasus's silent NA collapse.

**Consequences for storage and for the retirement roadmap:**
1. The **raw→canonical injection is already lossless w.r.t. variables**: we translate the raw DBF (Path A's `fetch_datasus`/`read.dbc` output, all columns), *not* microdatasus's semantic output. So the user's concern — "microdatasus can drop variables when translating" — does not affect us, because we never consume the translated copy. `schema_compare` exists precisely to keep this honest.
2. `microdatasus_processed.parquet` (~13 GB) is **safe to stop persisting now** — it is not on the production path. Keep `schema_compare` as an on-demand profile check (or persist only the small column *profile*, not the parquet).
3. What still binds us to microdatasus is only the **fetch transport** (`fetch_datasus` download) and the R bridge — not translation. The direct-`.dbc` fallback already fetches without microdatasus. So the remaining retirement work is: (a) promote direct `read.dbc` (or a pure-Python DBC reader) to primary, (b) delete the `process_datasus_dispatch` write. Translation is *already* in-house.
4. **Column fidelity is a requirement, not an inefficiency.** `processed.parquet` must keep every raw DBF column so the codebook can bind new concepts later without re-fetching. This is why §3/§4 only ever change the *codec*, never the column set.

## 8. SIDRA storage lifecycle & inefficiencies

SIDRA is far smaller (~4.4 GB total) — the user's expectation was right — but it has the same *shape* of redundancy, milder.

**Lifecycle per chunk** (`sidra/extract.py`, `sidra/cache.py`):
1. `SidraClient.values_from_chunk()` fetches; on the way it writes the response to the **client cache** `data/cache/sidra/<namespace>/<url+params hash>.json` + a `.sidecar.json` (2.1 GB total).
2. `extract_one_chunk()` then writes a **second raw dump** `data/sidra/<workdir>/raw/<chunk_id>.json` containing `{chunk, status, sidecar, payload}` — i.e. the same payload again — **pretty-printed with `indent=2`**.
3. `normalize_sidra_payload_to_facts()` → **facts parquet** per chunk (`.../facts/<table>/<chunk>.parquet`).
4. The pipeline concatenates chunk facts → **`facts.parquet`** per workdir, then my national path concatenates those → **national combined facts**.

**Inefficiencies:**
| # | Issue | Detail | Reclaim |
|---|---|---|---:|
| **S1** | Raw JSON payload stored **twice** | client cache (§1) *and* the extract raw dump (§2) hold the same bytes, keyed differently | **~2 GB** |
| **S2** | Raw dumps are **pretty-printed** (`indent=2`) | ~20–30 % larger than compact JSON; and JSON is ~3–5× a parquet of the same facts | part of S1 |
| **S3** | Facts materialized in **3 layers** | per-chunk → workdir-combined → national — same rows re-copied | grows per run |
| **S4** | **Per-UF workdir explosion** at national scale | population is fetched per-UF (27 workdirs); the compendium is 27 UF × ~94 tables, each with its own `raw/` + `facts/` + combined — thousands of tiny JSON/parquet | file-count |

**SIDRA fixes** (Tier 2–3): keep **one** raw archive, not two — either rely on the client cache and drop the extract raw dump, or drop the cache after facts materialize (the raw response is only needed to re-derive facts, and facts are cheap to keep). Write raw dumps **compact** (or gzipped) if kept at all. Collapse the 3 facts layers into the same lazy-`scan_parquet` view strategy as DATASUS (W7). For national, fetch all-UF localities per table into **one** workdir instead of 27 (also fewer requests). Estimated SIDRA reclaim **~2–2.5 GB**, and a large drop in file count.

## 9. End-to-end lifecycle map (fetch → use)

```
DATASUS   FTP .dbc ──parse──▶ (.dbc deleted ✓)
                              ├─ raw.rds                         ✗ never read (drop)
                              ├─ processed.parquet   RAW-CODED   ✓ KEEP (full fidelity; SNAPPY→ZSTD)
                              └─ microdatasus_processed.parquet  ✗ not consumed; schema_compare only (drop)
          processed.parquet ──in-house codebook──▶ canonical.parquet ─▶ combined ─▶ national ─▶ substrate ─▶ panel
                                                    └──────── make these lazy scan_parquet views, not copies (W7) ───────┘

SIDRA     IBGE API ──▶ client cache JSON            ✓ (one raw archive) 
                    └▶ extract raw dump JSON        ✗ duplicate payload, pretty-printed (drop/compact)
                       facts.parquet (chunk) ─▶ facts.parquet (workdir) ─▶ national facts ─▶ substrate
                       └──────── collapse to one facts layer + lazy view (W7) ───────┘
```

**Chokepoints, ranked:** (1) DATASUS 3×-serialization per chunk — the whole `raw.rds` + `microdatasus_processed.parquet` = ~26 GB is off the production path; (2) SNAPPY codec on the one artifact we keep; (3) SIDRA raw-JSON double storage; (4) N-layer re-materialization (DATASUS *and* SIDRA) that should be lazy views; (5) debug-file + per-UF-workdir sprawl. None of the keepers lose a single column or row — fidelity is fully preserved.
