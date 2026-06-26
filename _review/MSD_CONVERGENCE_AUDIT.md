# PegaSUS — Honest MSD Convergence Audit (deep review)

Triggered by justified skepticism that the "complete Alagoas e2e run" was real in
full breadth. **It was not.** This documents the true state, by evidence.

## 1. What is REAL (verified by execution)
- **SHE decoders** — SIM/SINASC/SIH/CNES raw→canonical, vectorized, real values (§2.4).
- **SHE substrate admission** + zero-variance exclusion (§2.4.0.6).
- **EFG generic engine** — `COUNT_MEASURE` per source carrier, `RN(count, population)`
  crude rates, `Bridge` divergence/race scaffolds (when fields exist).
- **Per-municipality SIDRA population anchor** (§2.8 anchor mode, Total only).
- **PIRS** — candidate→selection→GLM IRLS→cross-fitted residuals (§6.1–6.6).
- **HSIC** — real centered RBF kernel + permutation + BH/BY FDR (§6.7–6.9).
- **Honesty rails + computation-invariant validation** (§10).
- **Real national municipality crosswalk** (5,570 munis, all UFs) — replaces fixture.
- **SIDRA compendium registry** (96 tables) — catalog only, not yet ingested/built.

## 2. What is UNBUILT (the MSD analytical heart) — the gap
The all-source Alagoas compile yields **37 fields, 191 failed branches**. It is a
crude-rate skeleton. The following MSD-prescribed machinery does not exist:

| MSD | Capability | Status |
|---|---|---|
| §3.10 | **Core seed registry** (V_M/V_C/V_H/V_K/V_O/V_X named fields) | **NOT built.** `build_core_seed_set` only *classifies* existing fields; constructs nothing. |
| §3.11 | **ICD diagnostic traversal** (chapter/block/leaf, `Descend` gate) | **Zero implementation.** `health_seeds:[icd_chapter,icd_block]` is read but never used in `efg/dag.py`. No cause-specific mortality. |
| §3.10.4, §5.3 | **Maternal-child bridges** (infant/neonatal/postneonatal/perinatal/maternal mortality) | **Zero.** Listed in intent `mandatory_fields`, never produced. |
| §3.10.4 | Birth-outcome denominators | **Wrong** — LBW/prematurity computed as `/population` not `/live-births`. |
| §2.8 | **Demographic population tensor** (age×sex×race, block-coord/ADMM solver) | **NOT built.** Total anchor only → age-stratified rates fail (`numerator_axis_not_projectable:age`). |
| §2.10 | **ST-DFM** latent interpolation | Gated off / unbuilt. |
| §3.10.7, §2.12 | **SIDRA V_X context fields** (GDP/sanitation/education) + classification projection | Not constructed. |
| §4 | **Race bridge** | Fails (`race_bridge_required`); prior is a test fixture. |
| intent | **`mandatory_fields` enforcement** | **None.** Compile "succeeds" without producing mandated fields → hollow success. |

## 3. Fixture / test contamination still in PRODUCTION config
- ✅ **FIXED:** `municipality_crosswalk_codes.yaml` was a 2-entry `fixture_smoke_v1`
  → now real 5,570-municipality national crosswalk (`scripts/build_municipality_crosswalk.py`).
- ⬜ `config/registries/race_bridge_priors.yaml` → `prior_path: tests/fixtures/race_bridge/fixedC_valid.json`
  (test fixture in production registry). Note: an uncalibrated Dirichlet prior is
  MSD-legal *as a prior*, but it must not live under `tests/fixtures/` nor be
  presented as truth. Relocate + relabel.
- ⬜ Several `*_fixture` smoke intents in `config/intents/` (acceptable as dev
  intents if clearly named, but should not be the production path).

## 4. Honest verdict
The project is **NOT architecturally or operationally complete** for the Alagoas
breadth. What exists is a real, honest, thin crude-rate pipeline plus genuine
PIRS/HSIC inference machinery. The computationally-heavy core the MSD prescribes —
ICD traversal, the demographic tensor, maternal-child bridges, ST-DFM, SIDRA
context — is absent. No fraud (failures are honestly recorded), but large
incompleteness. Prior "complete" claims were overstated and are retracted.

## 5. Real remaining roadmap (prioritized, by epidemiological value × tractability)
1. **`mandatory_fields` / `health_seeds` enforcement** — fail compile if intent
   contracts are unmet (kills hollow success). *Small, high-honesty.*
2. **ICD chapter/block traversal (§3.11)** — `σ_C` restriction of underlying ICD
   by curated groups → cause-specific mortality. *The core epidemiological capability.*
3. **Core seed construction (§3.10)** — build the named V_M/V_C/V_H families.
4. **Maternal-child bridges (§3.10.4/§5.3)** with correct denominators.
5. **Demographic population tensor (§2.8)** — enables age/sex/race-stratified rates.
6. **SIDRA V_X context + projection (§3.10.7/§2.12)** from the compendium.
7. **ST-DFM (§2.10)** interpolation.
8. **Race bridge de-fixture + activation (§4).**

Items 2–5 are the bulk of the missing analytical heart and are multi-session
builds. This is the honest scope of remaining work before any national upscale.
