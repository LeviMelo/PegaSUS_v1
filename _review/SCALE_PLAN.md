# PegaSUS — National-Scale & 6 GB-VRAM Execution Plan

Goal: run **deep epidemiological workloads over the full Brazilian municipal grid
(5,570 municipalities)** on a single laptop — RTX 4050 **6 GB VRAM** (the binding
constraint), 32 GB DDR5, i7-13700H. This extends the MSD/TDD with concrete
memory-budgeted execution. Authority order: MSD > this plan > TDD (TDD built the
dysfunctional codebase — treat as hints, not truth).

> Current overall state: `MSD_CONVERGENCE_AUDIT.md`. Next feature tasks + entry points:
> `MSD_REMEDIATION_PLAN.md` §A. Near-term strategy (per user): prove the **full Alagoas
> (UF) scope** end to end with national-scale optimizations baked in, *before* the big
> upscale. The SIDRA acquisition client is now ruthless (parallel + gzip + cell-budget
> chunks) — national acquisition is feasible but time-bound by SIDRA's cell limits.

## 1. Where compute actually binds (measured reasoning, not assumption)

The pipeline is `D → SHE → EFG → PIRS → O_run`. Cost by stage at national scale:

| Stage | Dominant object | Binding resource | Strategy |
|---|---|---|---|
| Acquisition (DATASUS) | per-UF-year microdata (10⁵–10⁶ rows each) | disk + R | Already chunked `uf_year` (27 UFs × T). Each file bounded; never load all of Brazil at once. |
| SHE normalize | one UF-year parquet | RAM | Vectorized Polars (done for SIM/SINASC). Use `scan_parquet`+streaming for large UF files. |
| EFG executor | per-field tensors grouped to (year, muni) | RAM | Streaming groupby; output is tiny (≤5,570×T rows per field). |
| PIRS GLM | design matrix = support cells × covariates | trivial (≤5,570×T × ~30) | In-core NumPy. Not a constraint. |
| **PIRS HSIC** | residual×covariate kernel, **O(N²)** for N=cells | **RAM/VRAM** | **N≈5,570×T can be 5×10⁴–10⁵ → exact O(N²) is 10–40 GB. MUST use Nyström/RFF (O(N·m)).** |
| Population solver | Ω = munis×years×age×sex×race | **VRAM if dense** | Dense forbidden >10⁷ cells (MSD §2.8.12). Use SIDRA anchor (default) or sparse/block-coordinate/ADMM. |
| ST-DFM | latent factors × cells | VRAM | Gated; blocked by region; float32. |

**Key insight:** for the *default operational path* (SIDRA-anchored population, no
ST-DFM), VRAM is **not** the binding constraint — the heavy work is CPU/Polars
aggregation (bounded by per-UF chunking) and HSIC, which is bounded by
Nyström/RFF. VRAM only binds when the optional GPU solvers (population tensor,
ST-DFM) run. So national scale is reachable **CPU-first**, with GPU as an
opt-in accelerator under a strict VRAM budget.

## 2. Memory-budget rules (the 6 GB ceiling)

1. **Never materialize a Brazil-wide dense tensor.** Process per-UF, accumulate
   per-(muni,year) aggregates (already small). The national V_fields/Q_tensor is
   ≤ 5,570×T rows × N_fields — fits in RAM.
2. **HSIC at N>5,000 → feature maps, not kernels.** Already implemented in NumPy
   (`numpy_kernel_hsic_permutation_test`: exact ≤5,000, else RFF/Nyström). For
   GPU speed, port to torch with **float32 + tiled** kernels capped at
   `max_vram_fraction·6 GB`; chunk the N×m feature matmul.
3. **DuckDB for out-of-core joins** where a Polars frame would exceed RAM
   (national multi-year event concatenation). TDD prescribes `duckdb_polars_arrow`.
4. **float32 everywhere on GPU**, `max_vram_fraction: 0.80` (config exists).
5. **VRAM admission gate**: before any GPU op, estimate bytes (we have
   `compute/kernels.tensor_nbytes` + `compute/memory.py`); if it exceeds budget,
   fall back to CPU/Nyström/tiling instead of OOM.

## 3. SIDRA national context (the compendium)

`SIDRA_COMPENDIUM.md` = **96 curated tables, 13 epi groups, 4 tiers** (T1_CORE 41,
T2_CONTEXT 34, T3_OPTIONAL 11, T4_LOW 10), all 5,570-muni coverage. These are the
V_X context fields the EFG never built. Plan:
- **Parse the compendium into a machine-readable registry** (reproducible, not
  hand-typed) → drives ingestion + EFG context-field construction. *(Built this turn.)*
- Ingest by **tier** (T1_CORE first) and by **Default-Keep** flag — bounded,
  prioritized, not all-96-at-once.
- SIDRA chunking (≤49,900 cells, exists) bounds each request; national context =
  many chunks, persisted to the fact store.

## 4. Prioritized roadmap

1. ✅ **SIDRA compendium registry** — `scripts/build_sidra_compendium_registry.py` →
   `config/registries/sidra_compendium.json`. 96 tables, 95 with vars, 83 with
   classifications, tiered (T1_CORE 41 / T2 34 / T3 11 / T4 10).
2. ✅ **Per-municipality population tensor** — `load_sidra_population_totals_frame`
   (all Total-category localities → per-muni panel) + executor `_sidra_population_tensor`
   branch + frame-based materializer. **Verified: state compile over all 102 AL
   municipalities yields 102 per-muni crude mortality rates, mean 7.4/1000
   (epidemiologically correct).** This is the path that scales to 5,570 munis.
3. ⬜ **Streaming SHE/EFG execution** — `scan_parquet`/DuckDB so a UF (or national
   concat) never blows RAM.
4. ✅ **SIH-RD / CNES-ST vectorized decoders** — real, fast (0.3s for 162k+47k rows).
   SIH: year/residence/principal-ICD/death-flag/LOS/4 cost components (§2.4.2);
   CNES: facility id/muni/year, full QTLEIT*/QTINST* capacity vector (JSON),
   boolean-flag vector + invalid-flag clamp (§2.4.4/§2.4.0.4). All 4 DATASUS
   systems now decode to real canonical values.
5. ⬜ **GPU HSIC under VRAM budget** — torch float32, tiled Nyström, admission gate.
6. ⬜ **National orchestration** — loop UFs, accumulate, compile once; `execution_scale="national"`.
7. 🟡 **SIDRA V_X context-field construction in EFG** — ingestion path BUILT and verified
   (`she/sidra_context.py`, §2.9 regime-routed, one real GDP table proven). Remaining:
   ingest the compendium by tier/default-keep at breadth; §2.10 ST-DFM for multi-year
   `bounded_interpolate` regimes.
8. ⬜ **Streaming SHE/EFG** (`scan_parquet`/DuckDB) — memory-bounded national concat.
9. 🟡 **Ruthless SIDRA acquisition client** — BUILT (`sidra/api.py::fetch_chunks_parallel`,
   gzip transport, `sidra/plan.py::plan_sidra_chunks_unchecked` 95k chunks,
   `sidra/acquire.py`). 6.5× parallel speedup verified live. Remaining: national driver loop.
10. 🟡 **Demographic population tensor (§2.8)** — observed-census disaggregation BUILT
    (`she/demographic_tensor.py`, sex axis, stratified rates). Remaining: solver
    orchestration for independent/sim-informed modes (REMEDIATION_PLAN §A.1).

## 6. Verified results so far (by execution on real data)
- Smoke (Maceió 2022): 21 fields, valid bundle, crude mortality ≈ 24/1000 (city).
- State (Alagoas 2022, 102 municipalities): 102 per-municipality mortality rates,
  mean 7.4/1000 — real epidemiology over a full state grid.
- **Alagoas 2022 ALL-SOURCE (SIM+SINASC+SIH+CNES+SIDRA): compile success,
  37 fields, 16 edges, carriers = mortality / hospitalization / birth rates +
  facilities + population, across all 102 municipalities.** The complete
  UF-scope operational target is met.
- **Inference half now LIVE end-to-end**: PIRS selects outcome+covariates → fits
  GLM (ModelAssociations=1) → cross-fitted residuals → real RBF-kernel HSIC →
  2 hypotheses with real statistic/p-value/BH-BY q-value. The full
  `SHE→EFG→PIRS→HSIC→O_run` contract executes on real data.

### Convergence bugs fixed this turn (MSD §3.13 / §6)
PIRS was dead because three gates conflated **dashboard safety with model
eligibility** (MSD §3.13: model eligibility is governed by Q-state class, not
display safety):
- `run_candidates.pirs_candidate_rejection_reason` rejected all dashboard-unsafe
  fields → only 1 candidate. Fixed: eligibility by Q-state class only.
- `design_readiness` rejected dashboard-unsafe + quarantined_descriptive. Fixed.
- `run_candidates._role_for_field` mapped *every* measure kind to "outcome" → 0
  covariates → blocked design. Fixed: rates/densities = outcome; counts, capacity,
  costs, context = covariates.
- `output/validate` invariant checked a non-existent `residual_field_id` column on
  ModelAssociations; corrected to require a non-empty `ResidualAssociations` when a
  model is fitted.
Code trash removed: 16 dead `*_development` CLI stubs; `sidra metadata` registered.

## 5. Non-negotiables (carried from decontamination)
- No fixtures / fake green. Every stage computes, honestly blocks, or fails loud.
- Real values verified by execution, not labels.
