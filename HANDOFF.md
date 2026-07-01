# PegaSUS — Session Handoff (2026-07-01)

Compact state-of-play for resuming work. Branch: **`TDD-branch-(redo)`** (pushed to
`origin`, LeviMelo/PegaSUS_v1). Test env: conda `pegasus`
(`C:/Users/Galaxy/miniconda3/envs/pegasus/python.exe`). Suite: **246 pass / 0 fail**.

## The goal (win condition)
A **full-data Alagoas run** (all DATASUS + all SIDRA) through the **LDO** producing
**fine, discerning epidemiological relationships at scale**. The engine is built and
proven on real multi-year SIH/SINASC data; the remaining work is running it over the
full all-source panel (in progress) and lighting up the full SIDRA compendium.

## What is DONE (MSD-II + compliance)
- **Phases 0–4 built & verified** (synthetic + real). Commits are small and `MII-*`-tagged.
- **The LDO** (`src/pegasus/pirs/ldo/`): margins (copula) → GMRF-whitened sparse+low-rank
  precision → lag extension → stability selection → residual HSIC → certification, driven by
  `run_ldo(panel)`. Recovers a synthetic lag-7 edge exactly; on **real AL SIH/SINASC** it found
  9 stability-1.0 contemporaneous edges (cost components, count↔prevalence, Dengue↔Arbovirus).
- **Real-data robustness fix (key):** `pirs/ldo/covariance.py` pairwise-complete covariance
  (missing-aware) replaced impute-0, which had collapsed everything into `latent_shared`
  (complete-case cells were 0/1530). Also emit contemporaneous (lag-0) edges, not just lagged.
- **CommonPanel** (`she/panel.py`), **SpatialWeightGraph** (`geo/spatial_graph.py` +
  `config/registries/spatial/municipality_contiguity_queen.parquet`, real IBGE queen contiguity,
  5570 munis), **CTR kernel** (`she/reconstruction/`), **executable-authority registry**
  (`registries/callables.py` + `validators.validate_registry_authority`).
- **Geo-scope fix:** `efg/executor._with_geo` filters municipality_cod6 to the intent UF prefix
  (was leaking 937 national munis into AL runs → now 102).
- **SIDRA compendium FIXED** (`3ccbc40`): acquisition (`pipeline._acquire_sidra_compendium_context`)
  called `.get` on a SourceArtifact object and had **never completed**; now works — verified GDP/
  education/sanitation acquire real AL facts → `build_sidra_context_fields` → `context_gradient`
  nodes. **Wired into the harness** for contextual/full runs.
- **DATASUS fetch FIXED** (this session's deep dive): the slowness was NOT FTP rate-limiting.
  Real causes + fixes: (1) stalled FTP connections + no per-download timeout → R now uses a short
  per-download timeout + retry (`r_scripts/fetch_process_microdatasus.R`); (2) per-cell UTF-8
  sanitizer → vectorized per-column; (3) over-tight kill timeouts → generous heartbeat.
  Full multi-year all-source AL fetch now COMPLETES (was hanging 4h). Cache makes re-runs instant.

## IN PROGRESS right now
Full all-source compile: `config/intents/alagoas_2015_2022_allsource_ldo.json`
(core_vital, 2015–2022, all 4 DATASUS). Launched via the harness:
```
PEGASUS_RUN_INTENT=alagoas_2015_2022_allsource_ldo.json PEGASUS_RUN_ROOT=alagoas_2015_2022_allsource_ldo \
PEGASUS_RUN_YEARS=2015-2022 PEGASUS_RUN_SUBDIR=run_ldo_full \
PEGASUS_DATASUS_MAX_PARALLEL_REQUESTS=12 PEGASUS_DATASUS_R_TIMEOUT_SECONDS=1800 \
PEGASUS_DATASUS_HEARTBEAT_TIMEOUT_SECONDS=600 \
python -u scripts/dev/audits/actual_state_panel_runtime.py
```
DATASUS fetch is complete; currently normalizing (SIH is the slow ~6min record-path step) → then
SIDRA 9606 → EFG compile/materialize → `run_ldo_full/` gets the 17-key bundle + V_fields.

## NEXT STEPS (resume here)
1. When compile finishes, run the LDO over the full panel:
   ```python
   from pegasus.workflows.investigate import run_investigate
   class G: uf=["27"]; codes=[]
   class I: geography=G()
   run_investigate("data/actual_state_panels/alagoas_2015_2022_allsource_ldo/run_ldo_full",
                   intent=I(), resolution="year", K=3, lambda1=0.12, lambda2=0.5,
                   edge_threshold=0.12, n_subsamples=6)
   ```
   Map field_id→name via `run_ldo_full/V_fields.parquet` (`name` col). Expect fine cross-domain
   edges (mortality↔hospitalization↔births↔facilities). LDO real-data tuning that works:
   `lambda1≈0.12, lambda2≈0.5, edge_threshold≈0.12, K=2-3` (year resolution, T small).
2. **Full SIDRA**: run the same intent with `run_profile: contextual` to acquire all 94 tables →
   context_gradient fields → LDO discovers context↔health links. (Heavier; SIDRA API, ~chunked.)
3. Perf debt: SIH record-path normalize is ~2.2ms/row (slow). Vectorize-from-registry =
   `MII-REFACTOR-01`. LDO stability-selection is the slow LDO step on large panels.

## Watch-outs
- `gh` is NOT installed; push via `git push origin "TDD-branch-(redo)"`. No formal PRs (user pref).
- Commit style: short, objective, definitive (see memory `commit-message-style`).
- Development is NOT test-driven here (user directive): review code + validate via live/stress runs.
- Full detail + history in memory: `msd-build-status.md` (the running log).
- Month resolution: EFG count executor groups by YEAR only; true monthly grain through the EFG
  is not yet threaded (panel supports month keys but marks year-only fields honestly).
