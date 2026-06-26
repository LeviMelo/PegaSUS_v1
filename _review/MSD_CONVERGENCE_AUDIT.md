# PegaSUS — MSD Convergence: State of Record

**Purpose.** Single source of truth for *what is genuinely built and verified* vs *orphaned*
vs *missing* in the PegaSUS pipeline, measured against `MSD.md` (the authoritative spec).
Written so an AI with **no prior context** can orient and continue. Companion docs:
`MSD_REMEDIATION_PLAN.md` (defect history + precise next tasks with entry points) and
`SCALE_PLAN.md` (national-scale / 6 GB-VRAM strategy).

**Prime directives (from the user, non-negotiable):**
1. *More coding, less testing/auditing.* Fill architectural gaps; keep tests minimal.
2. *No scaffolding, no fake-green, no dumbing-down the MSD.* A label must never claim a
   computation that did not run. Honest failure (red) beats fake success.
3. *The engine is source-agnostic.* Adding a data source/event/ratio is a **registry edit**,
   not engine code. Carrier/ratio/event knowledge lives in registries, not Python.
4. *Real data only.* No fixtures in the production path. Verify by execution on real data.

**Environment.** Python via conda env `pegasus` (`C:/Users/Galaxy/miniconda3/envs/pegasus/python.exe`);
base python lacks polars. SIDRA API reachable. Real Alagoas-2022 data on disk under
`data/actual_state_panels/alagoas_2022_allsource/sources/` (normalized SIM/SINASC/SIH/CNES
events + processed SIDRA). Use `PYTHONIOENCODING=utf-8` (Windows console is cp1252).

---

## 1. The pipeline (MSD §1.2)

`D (DATASUS+SIDRA acquisition) → SHE (substrate harmonization) → EFG (epidemiological field
graph) → PIRS (penalized inference + selection) → HSIC (kernel independence) → O_run (bundle)`

The autonomous EFG is a **generic, registry-driven engine**: it admits source fields, builds
event counts, σ-restricts them (by ICD cause or demographic predicate), and forms
Radon–Nikodym ratios — all dispatched off registries, no per-source code.

---

## 2. VERIFIED BUILT (by execution on real Alagoas-2022 data)

| MSD | Capability | Where | Evidence |
|---|---|---|---|
| §2.4 | 4 real vectorized DATASUS decoders (SIM/SINASC/SIH/CNES) | `datasus/*_normalize.py` | real canonical values; 0.3s/209k rows |
| §2.4 | SHE substrate admission + zero-variance exclusion | `she/substrate.py` | registry-driven profiling |
| §2.8 | SIDRA population anchor (per-municipality panel) | `efg/materialize.py::_sidra_population_anchor_field`, `she/population/sidra_anchor.py` | 102 AL munis, total 3,127,683 ✓ |
| §2.8 | **Demographic population tensor** (sex strata) + stratified rates | `she/demographic_tensor.py`, `registries/demographic_axis.py` | sex pop sums to total; sex-specific mortality male>female |
| §2.8.3+ | **Population tensor solver orchestration now wired** | `she/population/orchestrator.py`, `workflows/population.py`, `workflows/compile.py`, EFG materialize/executor | compile tensor modes require real `population_strata`, run solver, append `population_tensor` artifact, admit/execute denominator field |
| §2.8 live acquisition | **SIDRA 9606 `population_strata` acquisition for tensor modes** | `workflows/pipeline.py` | live AL 2022 sex strata: 204 rows = 102 munis × {male,female}; solver tensor shape 102×1×1×2×1; sex-specific mortality RN produced |
| §2.9 | SIDRA contextual regime classifier (the gatekeeper) — **now live** | `sidra/regime.py` called by `she/sidra_context.py` | regime `direct`/`deflate` assigned to real context fields |
| §2.12.2/§3.7.4 | Canonical demographic axis alignment (sex codes→{male,female}) | `registries/demographic_axis.py` + `demographic_axis_maps.yaml` | SIM 1/2/0 + SIDRA 4/5/6794 → canonical |
| §3.3/§3.5 | Registry-driven carrier + ratio engine (NO hardcoded tables) | `clinical_event_definitions.yaml`, `registries/events.py`, `efg/{operators,dag,legality}.py` | behavior-preserving refactor verified |
| §3.10 | **Named Core Seed Registry** + `mandatory_fields` enforcement | `core_seed_registry.yaml`, `efg/core_seed_registry.py` | 32 canonical seeds resolve: V_M01-04, V_H01-03, V_B01, V_C01-03/06/08-11/14, V_K04-07, V_O01/04-07 + count seeds |
| §3.10.4-6 | **Ψ statistical-functional operator** (mean LOS/cost, median reporting delay) | `efg/operators.py::PSI_FUNCTIONAL`, `registries/functional.py`, `functional_fields.yaml`, executor `_functional_tensor` | mean LOS 6.64d; V_K05-07, V_O01 |
| §3.10.6 | **Observer-process share fields** (ill-defined cause / invalid ICD / missing race / investigation) | `clinical_event_definitions.yaml` (observer events) + general σ + RN | ill-defined ~7% ; V_O04-07 |
| §4 | Race-bridge prior **relocated out of `tests/fixtures/`** → `config/priors/race_bridge/` | `race_bridge_priors.yaml` | fixture contamination removed |
| §2.11 | **Cross-source divergence bridges** (morbidity↔mortality, fertility↔mortality) | `efg/operators.py::DIVERGENCE`, `bridge_grammars.yaml`, executor `_compute_bridge_tensor` | all-cause + cause-specific; log(adm/deaths)≈1.84, log(births/deaths)≈0.71; registry-driven, was dead executor |
| §3.10.7 | SIDRA V_X **context field ingestion** | `she/sidra_context.py` + materialize/executor hooks | real AL GDP (table 5938) → 102-muni context_gradient |
| §3.11 | **ICD cause-specific mortality** (σ_C chapter/block/**curated** restriction) | `efg/diagnostic_strata.py` + executor; curated via `icd_curated_groups.yaml` + `icd_groups.curated_group_for_icd` | partition exact; chapter I00–I99 leads; curated: ischemic-heart 2132 > cerebrovascular > diabetes (correct AL profile). V_M02/03/**04** |
| §2.6/§3.10.4 | **σ-restricted clinical events** (infant/neonatal/postneonatal/inpatient death, LBW/prematurity/anomaly) | registry `clinical_event_definitions.yaml` + general σ operator | IMR 12.9/1000; neonatal+postneonatal=infant exact; correct live-birth denominators |
| intent | `health_seeds` + `mandatory_fields` enforcement (kills hollow success) | `efg/diagnostic_strata.enforce_health_seeds`, `efg/core_seed_registry.enforce_mandatory_fields` | both fail loudly when unmet |
| §6.1–6.6 | PIRS: candidate→selection→GLM IRLS→cross-fitted residuals | `pirs/*`, `compute/glm.py` | model fitted on real data |
| §6.2/§6.5 | **Family selection + hurdle/ZINB + quasi-Poisson + Dunn–Smyth residuals** | `compute/glm.py` (`select_count_family`, `_fit_hurdle`, `randomized_quantile_residuals`), `pirs/model_execution.py` | zero-inflation→hurdle, overdispersion→NB; rq residuals std≈1 (calibrated); crossfit-compatible; data-aware `family_fitted` recorded honestly |
| §6.6.2 | **Deep-budget bootstrap-adjusted HSIC** D*=E_b[D_b]/(SD_b+ε) | `compute/glm.bootstrap_deviance_residual_replicates`, `pirs/hsic.bootstrap_adjusted_hsic`, wired in `pirs/hsic_run.py` (deep only) | 200 parametric replicates → D*, records ResidualMode/BootstrapCount/ResidualUncertainty; honest None fallback for hurdle/binomial; standard path unaffected |
| §6.7 | HSIC large-N path (Nyström/RFF feature maps, mode-switched by N) | `pirs/hsic.py` | already real; avoids n×n at national scale |
| §6.7–6.9 | HSIC: centered RBF kernel + permutation + BH/BY FDR | `pirs/hsic*.py` | real hypotheses w/ p,q |
| §10 | Honesty rails (execution errors fail loud, not relabel) | `workflows/msd_inference.py` | re-raises, no swallow |
| acquisition | **Ruthless cell-aware SIDRA client** (parallel + gzip + cell-budget chunks) | `sidra/api.py::fetch_chunks_parallel`, `sidra/acquire.py`, `sidra/plan.py::plan_sidra_chunks_unchecked` | 6.5× parallel speedup; live AL fetches |

**Full Alagoas all-source compile** (SIM+SINASC+SIH+CNES+SIDRA, 102 munis) executes end to
end: counts, crude + cause-specific + maternal-child + birth-outcome + capacity rates,
named seeds enforced, PIRS→HSIC live. This is the operational UF-scope target, met.

---

## 3. REAL BUT ORPHANED (substantial code, not wired into the live path)

These are **not stubs** — they are real implementations that no workflow/EFG calls. They
must be wired (and may need enrichment), **not** rewritten or skipped. See REMEDIATION_PLAN
for the precise wiring task per item.

| MSD | Module | Status |
|---|---|---|
| §2.8.3+ | `she/population/{solvers,sparse_admm,projected_gradient,block_coordinate,loss}.py` | Backends are now called by compile tensor modes through `she/population/orchestrator.py`, and `pegasus run` now acquires live sex-stratified 9606 `population_strata` for tensor modes. Remaining breadth work: age/race category projection; age categories include overlapping intervals and must not be bulk-requested without a registered basis/allocation kernel. |
| §2.10 | `she/stdfm/{pipeline,torch_solver,certification,objective,...}.py` | Real ST-DFM solver + certification. No live caller; `compile.py` hardcodes `stdfm`→skipped. Gate (§2.9) needs ≥3 temporal points → needs multi-year data to be meaningful. |
| §2.12.1 | `sidra/stitching.py` | Real longitudinal stitching (overlap calibration / methodological continuity). Not wired. |
| §2.12.2 | `sidra/projection.py` | Real classification projection matrix loader + legality. Not wired into EFG context construction. |
| §4 | `efg/race_bridge.py` + `registries/race_bridge.py` | Real fixedC dynamic-weight bridge + executor; prior still under `tests/fixtures/` (relocate). Fires only when intent requests race tensor mode. |

---

## 4. MISSING / BLOCKED (genuinely unbuilt or data-blocked)

- **§2.8 demographic-tensor breadth** — solver orchestration and live sex-strata
  acquisition are now live. Full munis×years×age×sex×race breadth still requires canonical
  age/race category maps. Age 9606 has overlapping interval and single-year rows; do not
  request all age categories without a registered non-overlapping basis or allocation kernel.
- **§2.10 ST-DFM execution** — wire `run_stdfm_pipeline` for `bounded_interpolate`-regime
  context fields; requires ≥3-temporal-point context data (acquire multi-year via the client).
- **§3.10.7 V_X breadth** — only one economic context table ingested as proof; ingest the
  curated compendium by tier (registry-driven; client is ready).
- **National scale** — streaming `scan_parquet`/DuckDB execution, GPU HSIC under VRAM budget,
  national UF-loop orchestration. (See SCALE_PLAN.)
- **Engine cleanups** — executor column-name heuristics (`GEO_COLUMNS`/`YEAR_COLUMNS`) and
  `diagnostic_strata.PRIMARY_DIAGNOSTIC_ROLES` should be registry-driven (same treatment
  applied to ratios). Race-bridge prior relocate out of `tests/fixtures/`.
- **Tests** — suite is label-asserting and partly stale (some collect-error on removed
  modules `pegasus.output.bundle`/`sidra_denominator_anchor`); a fraud-detector rewrite is
  deferred per the more-coding-less-testing directive.

---

## 5. How to run / verify (real data)

```bash
PY="C:/Users/Galaxy/miniconda3/envs/pegasus/python.exe"
# Build EFG directly from real normalized artifacts (fastest verification path):
PYTHONIOENCODING=utf-8 "$PY" - <<'EOF'
from pegasus.she.substrate import build_substrate_bundle
from pegasus.efg.dag import build_efg
from pegasus.efg.executor import execute_efg_result
base="data/actual_state_panels/alagoas_2022_allsource/sources"
arts=[{"path":f"{base}/normalized/sim_events.parquet","source_system":"SIM-DO","artifact_role":"processed_events"},
      {"path":f"{base}/processed/sidra/population_2022_al_n6_municipal_sidecar.parquet","source_system":"SIDRA","artifact_role":"normalized_facts"}]
efg=build_efg(substrate=build_substrate_bundle(artifacts=arts),
              intent={"health_seeds":["icd_chapter"],"mandatory_fields":[],
                      "geography":{"level":"municipality","codes":[],"uf":["AL"]},"geo_mode":"native"},
              operator_mode="standard")
print(len(efg.fields),"fields")
EOF
```
Artifact roles that drive admission: `processed_events` (DATASUS), `normalized_facts`
(SIDRA population Total anchor), `context_facts` (SIDRA V_X context), `population_strata`
(SIDRA disaggregated demographic tensor). Acquire SIDRA via `sidra/acquire.py::acquire_table_facts`.

Full compile: `pegasus run --intent <f>` (needs Rscript+network) or `run_compile(...)` with a
source manifest.
