# PegaSUS — Master Compliance & Remediation Report

**Audience:** PegaSUS developers and coding agents (including smaller/less-capable LLMs).
**Authority:** This document is the working contract for bringing the `/src` codebase into full compliance with the Master System Document (MSD). When this report and an ad-hoc code comment disagree, this report wins until the MSD itself is amended.
**Scope of evidence:** Derived from a line-level read of the MSD (6,823 lines) and the `/src` dump (227 files, ~35k lines) plus the 42-file `config/registries` tree. Configuration files outside `registries/` were not provided and are flagged where relevant.

---

## 0. How to use this document

Each finding is written as a self-contained unit with a fixed shape so an agent can act on one finding without reading the whole report:

- **ID** — stable handle (e.g. `SHE-NORM-01`). Cite it in commits/PRs.
- **MSD anchor** — the exact section(s) the requirement comes from.
- **Requirement** — what the MSD demands, in plain language.
- **Current state** — what the code actually does, with `file::function` evidence.
- **Severity** — see legend below.
- **Remediation** — concrete steps, often with code or pseudocode. Written to be executed with minimal additional judgment.

### Severity legend

| Severity | Meaning |
|---|---|
| **S0 — Contract breach** | Violates a hard MSD invariant or a §10 hard-abort condition. Produces silently wrong science. Fix before any production run. |
| **S1 — Capability gap** | A mandated capability is absent or reduced to a stub; downstream MSD fields/outputs cannot be produced correctly. |
| **S2 — Fidelity gap** | Implemented but mathematically/semantically weaker than the MSD; results are usable but mislabeled or approximate. |
| **S3 — Hygiene** | Dead code, duplication, stale metadata, naming drift. No direct scientific impact but erodes auditability and invites regressions. |

### The single most important idea in this report

PegaSUS has a **two-tier structure**: a *contract/scaffolding tier* (decoders, registries, the legality predicate, the population objective, the HSIC/null machinery, the certification thresholds, the 17-key validator) that is **rigorous and faithful to the MSD**, sitting above a *production/numeric tier* (the vectorized normalizers actually wired into the pipeline, the GLM family/spatial coverage, the race-bridge numerics, the SIDRA-context path) that is a **reduced MVP**. The danger is that the faithful upper tier lends an **unearned impression of completeness**: the system *validates* far more than it *computes*. Most S0/S1 findings below are about closing that gap — making the production tier honor the contracts the upper tier already encodes.

---

## 1. Executive summary

**What is genuinely strong (keep, protect with tests):**

- **Composite decoders** (`datasus/decoders.py`) — faithful to §2.4.0.1–2.4.0.5.
- **Population tensor objective** (`she/population/loss.py`) — a real analytic loss with analytic sparse gradients implementing every §2.8.4–2.8.9 term and the §2.8.2 mode constraints. Best-in-codebase.
- **EFG legality predicate** (`efg/legality.py`) — the full seven-part structural predicate plus the §3.8.8 declaration gate, fully registry-driven.
- **Race declaration gate** (`efg/declaration.py`) — correctly blocks the §4.3 illegal direct division (a §10 abort).
- **HSIC scanner + null regimes + FDR + cross-fitting** (`pirs/hsic.py`, `pirs/nulls.py`, `pirs/fdr.py`, `pirs/crossfit.py`) — faithful to §6.5–6.9 including both §10 residual-mode and seasonal-null aborts.
- **ST-DFM certification** (`she/stdfm/certification.py`) — faithful §2.10.4 thresholds.
- **Output 17-key validator** (`output/validate.py`, `output/schemas.py`) — faithful §8 contract enforcement.
- **EFG physical executor** (`efg/executor.py`) — real Polars materialization of count/functional/RN/sum/bridge tensors with σ_C restriction, ecological-fallacy guard, and cross-join prohibition.
- **SIDRA regime classifier** (`sidra/regime.py`) — faithful §2.9.

**The headline problems (detailed in §4–§6):**

1. **S0 — The production normalizers bypass the decoders/registry and emit a thin substrate.** The four `normalize_*_events` functions re-hardcode decoder logic inline and null out large parts of the §2.4 canonical field tables (SIM causal chain & associated conditions; SIH secondary diagnoses, ICU, movement geography, facility CNPJ; SINASC continuous weight/APGAR/parity/gestation; CNES typed bed indices). They also collapse §2.3 missingness states.
2. **S0 — CNES emits a forbidden generic bed total** (`capacity_total_observed`) and skips the §2.4.0.5 CNPJ sanitization gate at ingest.
3. **S0 — SINASC bakes EFG-level indicators into SHE** (`low_birth_weight_flag`, etc.), a layering violation that foreclosed several `V_C` fields.
4. **S1 — PIRS has no spatial effect modes** (§6.3 `ICAR`/`UF_FE`/`municipality_FE` are entirely absent).
5. **S1 — SIDRA context + ST-DFM are inert end-to-end** (acquired to disk, but not regime-routed/stitched/projected into EFG nodes; ST-DFM skipped in compile).
6. **S2 — The race bridge runs only the fast-mode fixed-W approximation** (no local-π Bayesian crosswalk, no posterior simulation, no true partial-identification bounds; CV is mis-defined).
7. **S1 — The large-scale population solvers are scaffolds** (ADMM is a `legacy_warning_scaffold`; primal-dual and a real state-space smoother are absent).
8. **S3 — Registry hygiene** — four orphaned alias stubs kept alive only by a mis-targeted validator; stale filename lists; a 2-entry model registry contradicting an 8-family GLM; scaffold `output_schema.yaml`.
9. **Cross-cutting — The MSD itself lacks a "valid partial run" concept** (the internal tension), which the code resolves silently. §3 designs the fix.

---

## 2. Architecture map (codebase ↔ MSD)

The package layout maps cleanly onto the MSD's three formal components. This is a structural strength; remediation should preserve it.

| MSD component | Primary packages | Status overview |
|---|---|---|
| **SHE** (Substrate Harmonization Engine, §2) | `datasus/`, `she/`, `geo/`, `sidra/`, parts of `registries/` | Decoders & population objective strong; **production normalizers reduced**; SIDRA-context path inert. |
| **EFG** (Epidemiological Field Graph, §3) | `efg/`, `registries/` | Legality, declaration, executor, compression **faithful**; high-cardinality axis bound under-enforced. |
| **Race/Color epistemology** (§4) | `efg/race_bridge.py`, `registries/race_*` | Interfaces/guards correct; **numerics reduced to fixed-W**. |
| **Bridge grammars** (§5) | `efg/bridges.py`, `registries/bridge.py` | Present; FacilityFlow limited by missing SIH CNPJ/movement substrate. |
| **PIRS** (§6) | `pirs/`, `compute/glm.py` | HSIC/nulls/FDR/crossfit **faithful**; **no spatial modes**; family set partial. |
| **Materialization & Output** (§7–§8) | `output/`, `storage/`, `efg/materialize.py` | 17-key contract **faithful**. |
| **Orchestration** (§9) | `workflows/` | Live chain runs SHE→EFG→PIRS→HSIC→bundle; **ST-DFM skipped**, context inert. |

**A note on the registry-as-authority principle.** The MSD §2.4 makes the Source Field Registry the authority for the *raw→canonical* transformation (`Route(x) ∈ {Decode, Parse, PreserveMark, Exclude}`, with a per-raw-field `decoder`/`parser`). The implemented `config/registries/source_fields.yaml` is keyed by **canonical** field names and carries only post-decode semantics (carrier/unit/aggregation/role/admissibility/axes). It therefore powers the *EFG legality* layer correctly but does **not** drive decoding. Decoding authority currently lives in hardcoded Polars inside the `normalize_*_events` functions. This split is the root cause of finding `SHE-NORM-01` and is the single most consequential architectural decision to reverse.

---

## 3. The internal MSD tension — diagnosis and designed resolution

### 3.1 The contradiction

Three groups of MSD statements cannot all hold simultaneously for the system as built:

1. **§1.2 / §9** describe a *closed, mandatory* chain `D → SHE → EFG → PIRS → O_run`. §9.2 steps 16–20 explicitly *require* SIDRA regime routing, longitudinal stitching, classification projection, mandatory high-dimensional bounded pushforward, and gated ST-DFM. §2.1 lists `B_reconstructed`, `B_latent`, `B_cross-sectional` as substrate classes SHE *emits*.
2. **§8.1 / §10** make *any* valid production run require all 17 keys (hard abort otherwise).
3. **§3.10.8's own implementation note (dated 2026-06-26)** concedes the context/ST-DFM/demographic-tensor machinery is "not yet wired into the live SHE→EFG ingestion path" and "require[s] the curated context tables (and multi-year/disaggregated data…) to be acquired before end-to-end execution."

The code resolves this silently: `workflows/compile.py::_run_compile_impl` runs SHE→EFG→PIRS→HSIC and emits a structurally complete 17-key bundle, but **ST-DFM is explicitly skipped** (`telemetry.set_stage("stdfm", "skipped", ...)`, reason "compile intent does not request latent-factor fitting"), the SIDRA context compendium is acquired to disk yet never transformed into EFG `context_gradient`/`latent_context` nodes (consistent with SIDRA being a one-admissible-field stub in `source_fields.yaml`), and `B_reconstructed`/`B_latent`/`B_cross-sectional` are effectively empty. So §8.1 is satisfied *structurally* (17 files exist) while §1.2/§9's *substantive* requirements are not exercised — and nothing in the output distinguishes "context legitimately absent for this run profile" from "context silently dropped because the code is unwired." That ambiguity is exactly the *"missingness with silence"* the MSD's closing sentence forbids.

This is not merely an implementation lag. It is an **architectural under-specification in the MSD**: every requirement is phrased absolutely (§9 "must", §10 "aborts when", §12 "the architecture locks"), but the system must ship before national context data exists. The MSD has no notion of a *partial-but-valid run*.

### 3.2 Designed resolution — the Run Profile contract (recommended)

Introduce a first-class **Run Profile** (a.k.a. capability tier) into both the MSD and the code. This converts today's tacit scoping into an explicit, validated contract.

**3.2.1 Define profiles.** Add to the MSD (new §1.5) and to `UserIntent`:

```
RunProfile ∈ { core_vital, contextual, full }
```

- `core_vital` — SIM/SIH/SINASC event streams + the independent population denominator. Substrate classes `B_reconstructed`, `B_latent`, `B_cross-sectional` MAY be empty. ST-DFM, the SIDRA context compendium, and the demographic tensor beyond the denominator are NOT required.
- `contextual` — `core_vital` + the curated SIDRA compendium routed through §2.9–§2.12 (regime → stitch → projection → bounded pushforward) and, where gated, ST-DFM (§2.10). `B_cross-sectional` and (when certified) `B_latent` become non-empty.
- `full` — `contextual` + the full §2.8 demographic denominator tensor over `(s,t,a,x,r)` and the race bridge (§4) over real census strata.

**3.2.2 Make the 17-key contract profile-aware.** The bundle keys remain mandatory (all 17 files must exist — keep the §8.1 hard abort). What changes is the **populated-content assertion**: `output/validate.py` gains a `required_nonempty_for_profile` map. Pseudocode:

```python
PROFILE_NONEMPTY = {
    "core_vital": {"V_fields", "E_DAG", "Q_tensor", "P_vector", "UserIntent",
                   "VariableDictionary", "RunConfig", "ReproducibilityManifest"},
    "contextual": ... core_vital | {"Hypotheses", "ModelAssociations"},  # plus context fields in V_fields
    "full":       ... contextual | {"Maps"},  # plus race-bridge + demographic-tensor provenance
}

def validate_output_bundle(root, *, run_profile):
    # 1. ALL 17 keys must exist as files (unchanged, hard abort).
    # 2. Keys in PROFILE_NONEMPTY[run_profile] must be non-empty.
    # 3. Keys NOT required for the profile MAY be empty, BUT each empty key must
    #    carry an explicit reason row in Warnings, e.g.
    #    {"key": "B_latent", "state": "empty_by_profile", "profile": "core_vital"}
```

Rule 3 is the crucial anti-silence guarantee: an empty `B_latent` is legal in `core_vital` **only if** the bundle explicitly declares it empty-by-profile. A consumer can then always tell "absent by design" from "absent by bug."

**3.2.3 Make §9/§10 profile-conditional.** §9.2 steps 16–20 and the §10 aborts ("SIDRA request exceeding…", "ST-DFM proportion field promoted…", "high-dimensional SIDRA field exposed without legal bounded pushforward") become **active only when the profile selects the relevant family**. For `core_vital`, they are inert (no SIDRA context requested). For `contextual`/`full`, they are hard requirements — and if the context machinery cannot run, the compile **aborts** rather than emitting a silently-empty context (Rule 3 prevents the silent path).

**3.2.4 Code changes required.**

- `core/schemas.py::UserIntent` — add `run_profile: Literal["core_vital","contextual","full"] = "core_vital"`.
- `workflows/compile.py::_run_compile_impl` — branch ST-DFM, compendium ingestion, and demographic-tensor stages on `intent.run_profile`; on `contextual`/`full`, make the SIDRA→EFG context path mandatory (see `SIDRA-CTX-01`).
- `output/validate.py` — implement `PROFILE_NONEMPTY` and Rule 3.
- `output/schemas.py` — add an `empty_by_profile` warning schema row.
- MSD — add §1.5 "Run Profiles" and annotate §9.2 steps and §10 aborts with their governing profile.

**3.2.5 Why this is the right resolution.** It (a) makes today's de-facto scoped run *formally* valid instead of tacitly valid; (b) turns the §3.10.8 apology into a first-class scope declaration; (c) closes the structural-vs-substantive validity gap; (d) preserves every hard MSD invariant for the families a run actually touches; and (e) gives a clean, testable ladder for incrementally lighting up `contextual` then `full` as data and code land. The two alternatives — "keep absolute and gate compile" (blocks shipping until context data exists) and "leave implicit" (the current silent state, which violates the no-silence principle) — are both strictly worse.

### 3.3 A secondary, genuinely-internal MSD tension (denominators vs maternal-child fields)

§2.8.2 defaults denominators to `B^POP_independent` (λ_D = 0). But §2.6.5 (perinatal) needs fetal-death records + an eligible-fetal-death denominator, and §3.10.4 (`V_C01–V_C14`) presupposes lagged-birth denominators and continuous birth-outcome primitives — data preconditions the MSD never reconciles with the independent-denominator default or with data availability. Several `V_C` fields are therefore declared *mandatory* (`V_core`) while their inputs are simultaneously declared *fragile/incomplete* elsewhere. **Resolution:** fold these into the Run Profile mechanism — mark `V_C04` (perinatal), `V_C05` (maternal), and any lagged maternal-child bridge as **`contextual`-or-higher** fields, and have the seed registry (`efg/core_seed_registry.py`) emit them as `quarantined_descriptive` with `warning=preconditions_unmet` under `core_vital` rather than failing the run.


---

## 4. SHE — Substrate Harmonization Engine (§2)

### 4.1 Composite decoders — COMPLIANT (keep)

**ID `SHE-DEC-01` · MSD §2.4.0.1–2.4.0.5 · Severity: none (verify only).**
`datasus/decoders.py` faithfully implements `decode_sim_idade` (unit codes 1–5 → hours/days/months/years/100+years with correct `age_years`/`age_days`), `decode_physical_scalar` (bounds + sentinels + explicit states), `decode_count2`, `clamp_bool` (the critical `x>1 ⇒ InvalidFlagState` invariant), and `filter_cnpj` (zero-CNPJ nullification, length/digit checks).

**Two minor hardening tasks:**

- **`SHE-DEC-02` (S2)** — `filter_cnpj` does **no CNPJ check-digit validation**. Any 14-digit string is `ValidCNPJ`, and the `InvalidCNPJDigits` branch is dead (digits are already regex-filtered to decimals). The MSD's `digits ∈ 𝔻¹⁴` implies the check-digit set. *Fix:* add the standard Brazilian CNPJ mod-11 verification; route failures to `InvalidCNPJDigits`. This matters because the gate is the only thing preventing the §2.4.0.5 catastrophic many-to-many `{0...0} × {0...0}` join, and malformed-but-14-digit ids currently pass.
- **`SHE-DEC-03` (S2)** — `decode_sih_age` hardcodes its unit map inline (codes 2–5 only; 0/1 unhandled) despite §2.4.0.1 requiring `UnitMap_SIH` to be a registry table. *Fix:* move the map into `config/registries/composite_decoders.yaml` and read it; handle all documented `COD_IDADE` values; emit `UnknownAgeUnit` with `warning=age_unit_missing` for unmapped codes (already partially done).

### 4.2 Production normalizers bypass the decoders and emit a thin substrate — **S0, the central SHE defect**

**ID `SHE-NORM-01` · MSD §2.3, §2.4, §2.4.1–2.4.5, §2.5 · Severity: S0.**

**Current state.** The pipeline wires the *vectorized* normalizers, not the registry-driven ones:
- `workflows/datasus.py::run_datasus_normalize_sim` → `datasus/normalize.py::normalize_sim_do_events`.
- `workflows/cnes_sih.py` → `datasus/sih_normalize.py::normalize_sih_rd_events`, `datasus/cnes_normalize.py::normalize_cnes_st_events`.
- `workflows/sinasc.py` → `datasus/sinasc_normalize.py::normalize_sinasc_events`.

Each of these:
1. **Re-implements decoder logic inline** as Polars `when/then` chains (e.g. SIM IDADE slicing in `normalize_sim_do_events`, SIH age in `normalize_sih_rd_events`), duplicating `datasus/decoders.py` and bypassing the registry. The function `normalize_sim_do_events`' own docstring admits the registry path "looked up *raw* DATASUS column names in a registry keyed by *canonical* names, matched nothing, and emitted 60 canonical columns that were 100% null." The fix below addresses that root cause rather than working around it.
2. **Nulls out** (`pl.lit(None)`) large parts of the §2.4 canonical schema. Concretely:
   - **SIM-DO:** `cause_chain_norm` (LINHAA–D, §2.5.2.2 ordered terminal chain), `associated_conditions_norm` (LINHAII, §2.5.2.3), all observer-process marks (`medical_assistance`, `exam_performed`, `surgery_performed`, `autopsy_performed`, `certificate_date`, `investigation_*`), all maternal/perinatal marks, `mun_*_cod7`, and `facility_code` are emitted null. Only ~16 of ~60 columns carry data.
   - **SIH-RD:** no `secondary_icd_set` (§2.5.3), no ICU marks (`MARCA_UTI`, `UTI_MES_TO`, `UTI_INT_TO`), no `mun_movement` (MUNIC_MOV), no procedures, no `hospital_cnpj`/`maintainer_cnpj`. Cost components (VAL_SH/SP/UTI/TOT) **are** kept separate — good.
   - **SINASC:** no continuous `birth_weight_grams` (only a `low_birth_weight_flag`), no `apgar_1min`/`apgar_5min`, no reproductive-history counts (QTDFILVIVO/QTDFILMORT/QTDGESTANT/QTDPARTNOR/QTDPARTCES), no `gestational_weeks`, no `prenatal_visit_group`, no typed `anomaly_icd` (uses a raw regex, losing parse-state).
3. **Collapses §2.3 missingness states** — `sex`/`race` map null *and* invalid both to `"unknown"`; `race_missingness_state` is emitted null; the distinct `Unknown ≠ Missing ≠ Invalid ≠ Unparseable ≠ NotRecorded` requirement is lost.

**Why S0.** This is not a cosmetic subset; it silently removes mandated source reality. Downstream effects that produce *wrong or absent* science:
- `V_M05` (TerminalChainMentionShare), `V_M06` (AssociatedConditionMentionShare) — uncomputable (SIM chain/associated null).
- `V_H07` (SecondaryComorbidityMentionShare), `V_H06` (ICUUseShare), `V_K06` (ICUDaysPerHosp) — uncomputable (SIH secondary/ICU absent).
- `V_C07` (VeryLowBirthWeight <1500), `V_C10`/`V_C11` (LowApgar), `V_C12` (Kotelchuck), `V_C13` (Robson) — uncomputable (SINASC primitives absent).
- The §5.9 FacilityFlow bridge — unusable (no SIH CNPJ, no movement geography).
- §2.3's explicit-missingness invariant — violated globally.

**Remediation (the most important fix in this report).** Make decoding registry-driven and complete. Two acceptable paths; **Path A is strongly preferred** because it removes duplication and restores the registry-as-authority principle.

**Path A — Extend the registry to carry raw→canonical routing, and make one normalizer registry-driven.**

1. Extend `source_fields.yaml` (or a new `source_field_routing.yaml`) so each canonical field also declares its *source* columns and decoder/parser, matching the MSD §2.4 table shape:
   ```yaml
   SIM-DO:
     fields:
       age_years:   {raw_fields: [IDADE], route: Decode, decoder: decode_sim_idade, output_key: age_years, ...}
       cause_chain_norm: {raw_fields: [LINHAA,LINHAB,LINHAC,LINHAD], route: Parse, parser: parse_icd_ordered_chain, ...}
       associated_conditions_norm: {raw_fields: [LINHAII], route: Parse, parser: parse_icd_unordered_set, ...}
       medical_assistance: {raw_fields: [ASSISTMED], route: Decode, decoder: decode_categorical, ...}
       # ... every canonical field in §2.4.1
   ```
2. Make `datasus/declarative_normalize.py` the **single** record-level normalizer for all four systems. It already dispatches decoders by name via `_resolve_decoder_callable`; point it at the new `raw_fields`/`decoder` keys. Ensure it preserves every §2.3 state (the decoders already return them — stop collapsing to `"unknown"`).
3. For performance, keep a vectorized fast path **only as a Polars-compiled view of the same registry routing** — i.e. generate the `when/then` expressions *from* the registry rather than hand-writing them, so there is exactly one source of truth. If that is too large a step initially, run the record-level path and accept the throughput cost until profiled.
4. Delete the hand-rolled column logic in `normalize_*_events`. The `normalize_sim_do_record` shim that already delegates to the declarative path becomes the canonical entry; `normalize_sim_do_events` becomes a thin batch wrapper over it.

**Path B (if Path A is too large in one step) — Make the vectorized normalizers complete and decoder-backed.** Keep the four `_events` functions but: (a) replace every inline `when/then` decode with a call into `datasus/decoders.py` (use `map_elements`/`map_batches` or a vectorized re-expression that is *unit-tested to equal* the decoder); (b) populate **every** canonical column the registry declares (run `parse_icd` over LINHAA–D and LINHAII; decode APGAR, weight, parity, gestation, ICU, secondary dx, movement, CNPJ); (c) stop mapping invalid→`"unknown"` — emit the decoder's explicit state into the `*_state` columns.

**Acceptance test for either path** (`SHE-NORM-01`): on a fixture covering one record per state, assert that (i) `age_years`/`age_days` equal `decode_sim_idade`'s output exactly; (ii) `cause_chain_norm` has 4 ordered positions with per-position parse states; (iii) `associated_conditions_norm` is a non-null set; (iv) `birth_weight_grams` is continuous and `V_C07 <1500` is derivable; (v) every `*_state` column distinguishes Missing/Invalid/Unparseable. Add a regression test asserting **no canonical column declared admissible in `source_fields.yaml` is 100% null** on a non-trivial fixture.

### 4.3 CNES emits a forbidden generic bed total and skips the CNPJ gate — **S0**

**ID `SHE-CNES-01` · MSD §2.4.4, §3.3, §3.5 · Severity: S0.**
`datasus/cnes_normalize.py::normalize_cnes_st_events` computes `capacity_total_observed = sum_horizontal(all QTLEIT* + all QTINST*)`. This is precisely the generic "bed count" §2.4.4 forbids ("CNES-ST does not expose a single scalar bed count") and it mixes infrastructure indices (`QTINST`) with bed indices (`QTLEIT`), violating §3.5 ("additive only within the same capacity index"). The per-index vector is preserved as `capacity_vector_json` (good), but the registry then exposes `capacity_total_observed` as an admissible-adjacent field, so anything downstream can treat it as "Beds".

**Remediation.**
1. **Delete `capacity_total_observed`** from the normalizer output and from `source_fields.yaml`'s CNES section.
2. **Surface typed per-index carriers** the MSD names: `clinical_bed_capacity` (QTLEITP1), `surgical_bed_capacity` (QTLEITP2), `obstetric_bed_capacity` (QTLEITP3), plus the general `facility_bed_capacity_vector[k]` / `facility_room_capacity_vector[k]`. Each must carry `axes.capacity_index = k` so `efg/legality.py::_specialized_semantics_ok` (which already requires `capacity_index`) admits it.
3. If a composite is ever needed, it must go through the registered `CostComponentSum`-analog for capacity (a declared `K_bed_composite`), not a silent sum — see §3.3/§3.5. Do **not** reintroduce a default total.

**ID `SHE-CNES-02` · MSD §2.4.0.5, §2.4.4 · Severity: S0.**
`normalize_cnes_st_events` does **not** apply `filter_cnpj` to `CPF_CNPJ`/`CNPJ_MAN`. The §2.4.0.5 corporate linkage gate is therefore unenforced at ingest, and the §5.9 catastrophic many-to-many guard depends on it.

**Remediation.** Run `filter_cnpj` over `CPF_CNPJ` → `facility_cnpj` and `CNPJ_MAN` → `maintainer_cnpj`, emitting `facility_cnpj` + `facility_cnpj_state`. Add both to the CNES registry section. Add a `LinkageQualityShare`/`ZeroCNPJShare` observer field (`V_K09`) computed from the states.

**ID `SHE-CNES-03` · MSD §2.3 · Severity: S2.**
The normalizer does `fill_null(0)` on capacity before summing. §2.3 forbids silently treating missing as zero unless the registry declares it. *Fix:* keep nulls as nulls; only treat as zero where `composite_decoders.yaml`/`cnes_capacity_registry.yaml` declares a registry-level zero default for that index.

### 4.4 SINASC bakes EFG-level indicators into SHE (layering violation) — **S0/S1**

**ID `SHE-SINASC-01` · MSD §2.4.3 vs §3.10.4 · Severity: S0.**
`sinasc_normalize.py` emits `low_birth_weight_flag`, `prematurity_flag`, `cesarean_flag`, `adolescent_mother_flag`, `advanced_maternal_age_flag`. These are §3.10.4 EFG-level *outcome indicators* (`V_C06`, `V_C08`, `V_C09`), not §2.4 canonical *primitives*. Computing them in SHE (a) forecloses downstream re-thresholding and the primitives' other uses, and (b) is the proximate cause of `V_C07`/`V_C10`/`V_C11`/`V_C12`/`V_C13` being uncomputable.

**Remediation.** SHE must emit primitives; the EFG derives indicators.
- Emit `birth_weight_grams` (via `decode_physical_scalar`), `apgar_1min`, `apgar_5min`, `gestational_weeks`, `gravidez`/`pregnancy_type`, `parto`/`delivery_type`, `consultas`/`prenatal_visit_group`, and the five reproductive-history counts (via `decode_count2`).
- Move the threshold logic (`<2500`, `<1500`, `<37 weeks`, `parto==2`, age cutoffs, Kotelchuck, Robson) into the EFG seed registry (`efg/core_seed_registry.py`) as `σ_C` restrictions over the primitives, exactly as `V_C06–V_C14` specify.
- Keep `maternal_age_years` (already emitted) but drop the derived `adolescent`/`advanced` flags from SHE; derive them in EFG if needed.

### 4.5 Population tensor objective — COMPLIANT (keep, extend solvers)

**ID `SHE-POP-01` · MSD §2.8.2–2.8.9 · Severity: none (verify only).**
`she/population/loss.py::evaluate_population_loss` is the strongest math in the codebase: analytic loss + analytic sparse gradients for anchors (§2.8.10), cohort aging with terminal pool (§2.8.4), births (§2.8.5), death soft-prior (§2.8.6), ILR race composition (§2.8.8), migration second-difference (§2.8.7), age second-difference (§2.8.9). `validate_population_problem` enforces the §2.8.2 mode constraints (`independent ⇒ λ_D=0`; `sim_informed ⇒ λ_D>0`). Protect this with golden-value tests so solver refactors cannot silently change it.

**ID `SHE-POP-02` · MSD §2.8.12 · Severity: S1.**
Only `projected_gradient_small` and `sparse_block_coordinate` solvers are active (`registries/population_solver_registry.yaml` statuses `active`/`active_sparse`). `sim_informed_sparse_admm_scaffold_v1` is `legacy_warning_scaffold` ("retained only for reproducibility of earlier runs"); `she/population/sparse_admm.py::build_sim_informed_sparse_admm_scaffold` returns a scaffold object, not a solver. `she/population/state_space.py` provides index/coordinate plumbing only (`build_population_state_space`) with **no smoother**. The §2.8.12 `primal-dual_sparse` backend is absent. The 10⁷ threshold and the dense-scale hard abort (`assert_dense_population_tensor_allowed`, §10) are correctly present.

**Remediation (phased, only needed for `full` profile at Brazil scale).**
1. **ADMM (§2.8.12).** Implement the block decomposition the MSD spells out: partition `S*` by UF/macroregion/AMC cluster; per-block subproblem `argmin L_b + (ρ/2)‖A_b P_b − Z_b^k + u_b^k‖²` reusing `loss.py` per block; consensus `Z^{k+1} = Consensus(...)`; dual `u_b^{k+1} = u_b^k + A_b P_b^{k+1} − Z_b^{k+1}`; terminate on `‖r^k‖≤ε_pri ∧ ‖s^k‖≤ε_dual`. The per-block objective already exists — ADMM is an orchestration layer over it. Mark `status: active_sparse` only after a convergence test on a synthetic multi-UF problem.
2. **State-space smoother.** Implement an RTS/Kalman smoother over the cohort transition (the aging recursion is already linear-Gaussian-shaped) for the reduced UF/region path; wire `build_population_state_space` into an actual `solve`.
3. **primal-dual_sparse** — implement or, if deferred, **remove its band from the §2.8.12 selection table in the MSD** so the spec and code agree (do not leave a spec'd-but-absent backend).
4. Until 1–3 land, `select_population_solver` must **refuse** (not silently downgrade) when `|Ω_epi|` lands in a band whose only backend is a scaffold, emitting a clear `population_solver_unavailable_for_scale` abort. Today it falls through to block-coordinate, which may be acceptable but must be *explicit and warned*, not silent.

### 4.6 SIDRA context pipeline + ST-DFM are inert end-to-end — **S1**

**ID `SIDRA-CTX-01` · MSD §2.9–§2.12, §3.10.8, §9.2 steps 16–20 · Severity: S1.**
The building blocks exist and several are faithful: `sidra/regime.py` (regime classifier, §2.9 — correct), `sidra/stitching.py` (§2.12.1), `sidra/projection.py` (§2.12.2), `sidra/pushforward.py` (§2.12.3 bounded pushforward), `she/stdfm/*` (§2.10 with faithful certification thresholds). The live pipeline (`workflows/pipeline.py::_acquire_sidra_compendium_context`) acquires compendium facts to disk. **But** `workflows/compile.py::_run_compile_impl` skips ST-DFM and never routes context facts through regime→stitch→projection→pushforward into EFG `context_gradient`/`latent_context` nodes. `source_fields.yaml`'s SIDRA section has **one** admissible field, so even if facts reached SHE they would not become admissible substrate.

**Remediation (gated by the Run Profile; required for `contextual`/`full`).**
1. Build the SHE→EFG context bridge that the MSD §3.9.3 boundary order prescribes:
   `Y_raw --LongitudinalStitch--> Y† --Π_Clsf→Axis--> Y_axis --π^bound_*--> Y_bounded --Δ--> V_fields`.
   Wire `sidra/stitching.py` → `sidra/projection.py` → `sidra/pushforward.py` → `she/substrate.py` so each compendium artifact becomes a `ContextCells`-carrier substrate field with regime `R(q)` attached.
2. Expand `source_fields.yaml` SIDRA section so projected context fields are admissible `context_gradient` (and, when ST-DFM-certified, `latent_context`) with the §3.6 latent-provenance quarantine (dashboard-unsafe by default).
3. In `compile.py`, branch ST-DFM on `run_profile ∈ {contextual, full}` and on regime `bounded_interpolate`; run `she/stdfm/pipeline.py::run_stdfm_pipeline` and gate promotion via `certification.py`.
4. Enforce the §10 aborts here: SIDRA cell budget (49,900), mandatory bounded pushforward before EFG exposure (`SIDRA-CTX-02`), and the ST-DFM-proportion-without-denominator abort.

**ID `SIDRA-CTX-02` · MSD §2.12.3, §3.8.2 · Severity: S1.**
The §3.8.2 axes term in `efg/legality.py::evaluate_delta` only fails on alignment failure; it does **not** itself enforce the high-cardinality axis-demand bound (`Card(v) > Card_max^EFG ⇒ mandatory pushforward`) nor fail non-aggregable high-dimensional marginalization. The bounding logic lives in `sidra/pushforward.py`/`she/high_dimensional.py` but is not invoked on the legality path. *Fix:* before node materialization in `efg/dag.py`/`efg/materialize.py`, call the bounded-pushforward planner for any field whose raw axis product exceeds the EFG cap, and have the axes term return `Δ_axes = 0` when a high-dimensional field requires marginalization its aggregation law forbids (per §3.8.2 final clause). Add the §10 abort "high-dimensional SIDRA field exposed to EFG without legal bounded pushforward."


---

## 5. EFG — Epidemiological Field Graph (§3)

### 5.1 Legality predicate + declaration gate — COMPLIANT (keep)

**ID `EFG-LEG-01` · MSD §3.8 · Severity: none.**
`efg/legality.py::evaluate_delta` implements all eight terms (support, axes, carrier, unit, aggregation, provenance, quality, declaration). Carrier/unit/aggregation knowledge is fully registry-driven (no hardcoded sets). `_specialized_semantics_ok` enforces §3.3 generic-`Beds` rejection, §2.4.4 capacity-index requirement, and §2.4.2 cost-component requirement. The `RN` branch enforces additive numerator/denominator, registered ratio semantics, and blocks diagnostic-topology numerators from ratios without prior restriction. The one gap (high-cardinality axis bound) is tracked as `SIDRA-CTX-02`.

**ID `EFG-DECL-01` · MSD §3.8.8, §4.3 · Severity: none (one hardening task).**
`efg/declaration.py::evaluate_declaration_compatibility` correctly blocks `RN(admin-race numerator, self-declared denominator)` without `Bridge_R` (the §10 abort). **Hardening (`EFG-DECL-02`, S2):** when either field's race axis is *absent*, the gate returns OK (permissive). A numerator stratified by race but missing its `race_axis_type` metadata would slip through. *Fix:* if an operand is race-stratified (has a race axis column) but lacks `race_axis_type`, fail with `race_axis_declaration_unverifiable` rather than passing. This depends on `SHE-NORM-01` emitting `race_axis_type` reliably.

### 5.2 EFG executor — COMPLIANT (keep)

**ID `EFG-EXEC-01` · MSD §3.9, §3.10, §3.11 · Severity: none.**
`efg/executor.py` is a real physical executor: `_count_tensor` (with σ_C ICD chapter/block restriction via `_add_icd_stratum`, demographic stratification joins, declarative AND-list `_apply_restrict_conditions`), `_functional_tensor` (Ψ operators), `_compute_rn_ratio` (ecological-fallacy guard, **cross-join forbidden** per §3.8.1, stratifier survival, denom-fragility tracking), `_sum_tensor`, and bridge/race execution. Note the RN ecological-fallacy guard aggregates the numerator up to the denominator's support when mismatched — acceptable, but ensure it can never silently *broadcast a rate*; it currently operates on counts, which is correct.

### 5.3 Two-stage compression — COMPLIANT (keep)

**ID `EFG-CMP-01` · MSD §3.15 · Severity: none.**
`efg/empirical_compression.py` computes `max(|ρ_Pearson|, |ρ_Spearman|) ≥ 0.98` on materialized vectors only, never on metadata-only/planned nodes (§7), folding the lower-utility node without data loss. Topological pre-compression (§3.15.1) lives in `efg/equivalence.py`.

### 5.4 State tensor completeness — S2

**ID `EFG-Q-01` · MSD §3.12 · Severity: S2.**
The `Q_tensor` **required** columns in `output/validate.py` (`REQUIRED_Q_TENSOR_COLUMNS`) omit `CV`, `MoranI`, `temporal_roughness`, and `spatial_entropy` from the §3.12 fourteen-component tuple. But `MoranI` feeds `n_eff` (§3.12.3) and `CV` feeds `Quality` (§3.14) and the §3.12.14 state classification. If they are not computed/persisted, `n_eff` and `Quality` are wrong, which cascades into state assignment and utility scoring. *Fix:* add the four columns to `REQUIRED_Q_TENSOR_COLUMNS`, ensure `efg/q_tensor.py` computes them, and assert in a test that `n_eff` reflects the Moran's-I correction (`n_eff = (Σw)²/Σw² · 1/(1+max(0,MoranI))`).

### 5.5 Carrier/operator/seed registry completeness — verify against §3.3/§3.9/§3.10

**ID `EFG-REG-01` · MSD §3.3, §3.10 · Severity: S1 (pending `SHE-NORM-01`).**
The carrier registry (`carrier.yaml`) and seed registry (`core_seed_registry.yaml`) are well-formed, but several core `V_*` fields (`V_M05/06`, `V_H06/07`, `V_C07/10/11/12/13`, FacilityFlow) are *unreachable* until `SHE-NORM-01`/`SHE-CNES-*`/`SHE-SINASC-01` populate their inputs. *Action:* after the SHE fixes, add an integration test that compiles a fixture and asserts each `V_core` field either materializes or emits a *typed* `FailedBranch`/`quarantined` reason — never a silent absence.

---

## 6. Race/Color bridge (§4)

**ID `RACE-01` · MSD §4.5.3, §4.6, §4.7 · Severity: S2.**

**Current state.** `efg/race_bridge.py::fixedc_dynamic_weight_bridge` validates the emission-prior object (the §10 "no valid prior" abort — good), preserves raw admin counts and the Missing/Unknown category (§4.8 — good), and emits all §8.6 metadata keys (the validator `output/validate.py::_validate_race_bridge_contract` enforces them). But the numerics are the **fast-mode fixed-W approximation only**:
- It applies the prior matrix *directly* as the crosswalk `W` (`posterior = Σ_source n·W`). The §4.5.3 Bayesian local-posterior crosswalk — whose defining property is dependence on the **local** self-declared baseline `π_i^(k) = P_i / Σ P` via Bayes (`W_{j,i} = C_{i,j} π_i / Σ_m C_{m,j} π_m`) — is **not** computed. So despite the name "dynamic_weight", `W` is *not* locality-dynamic.
- The §4.6 posterior simulation (Dirichlet draws of `C`, `π`; Poisson draws of `ν`) for standard/deep budgets is absent.
- The §4.7 partial-identification bounds are a heuristic `value·(1±width)` band, **not** the `inf/sup` over the emission-matrix credible set `S_C`.
- `race_bridge_cv` measures dispersion *across race categories of the posterior vector*, **not** the §4.6 `CV_racebridge = SD_b(ρ̂^(b))/E_b(ρ̂^(b))` *across bootstrap draws*. This wrong quantity then feeds the §4.10 downgrade triggers and the §4.9 verified-promotion gate, so fields can be mis-stated.

**Why S2 not S0.** The §4.3 illegal-division and §10 no-prior aborts *are* enforced, and outputs are labeled as bridged (the forbidden labels of §4.11 are not emitted). So results are not catastrophically wrong — but they are **mislabeled as more certain/locally-calibrated than they are**, which is exactly what §4 exists to prevent.

**Remediation (phased).**
1. **Local-π crosswalk (correctness, do first).** Pass the local self-declared denominator composition `π_i^(k)` into the bridge and compute `W^(k)_{j,i}` by Bayes per §4.5.3. The emission matrix `C` (not `W`) is the registry prior (`race_bridge_priors.yaml`). Rename the operator to reflect that `W` is now locality-dependent.
2. **Fix the CV.** Compute `CV_racebridge` from bootstrap/posterior draws of `ρ̂`, per §4.6. Until simulation exists (step 3), use the fixed-W variance approximation `Var(ν̂_i) = Σ_j W_{j,i}² ν̃_j` (the §4.6 fast-mode formula) and label `BridgeMode = fast`. Do **not** keep the cross-category-spread definition.
3. **Posterior simulation (standard/deep).** Implement the §4.6 draws (`C^(b) ~ Dirichlet(α)`, `π^(b) ~ Dist(P̂, U_POP)`, `ν̃^(b) ~ Poisson`) → per-draw `W^(b)` → `ρ̂^(b)`; emit `E`, `Median`, `CI_95`, `Var`.
4. **Partial-ID bounds (high-stakes).** Implement `ρ_lower/ρ_upper = inf/sup over C ∈ S_C` (linear-fractional in `C`; solvable as a small optimization per cell, or by sampling the credible set as a conservative approximation). Wire `SensitivityWidth` from these, feeding the §4.10 `> θ_sens ⇒ quarantined_descriptive` downgrade.
5. **State rules.** Confirm `efg/q_tensor.py`/state assignment applies §4.9 `State(Bridge_R(v)) ≤ fragile` by default and the §4.10 downgrade triggers using the *corrected* CV/SensitivityWidth.

---

## 7. PIRS — Parametric Inference & Residual Scanner (§6)

### 7.1 HSIC scanner, nulls, FDR, cross-fitting — COMPLIANT (keep)

**ID `PIRS-HSIC-01` · MSD §6.5–6.9 · Severity: none.**
`pirs/hsic.py::select_hsic_mode` matches §6.7 (`n_eff<100→disabled`; `>5000 ∧ {standard,deep}→Nyström`; `>5000 ∧ fast→RFF`; else exact); `validate_residual_mode_for_hsic` enforces the §10 in-sample-on-standard/deep abort; the scanner uses add-one permutation p-values and a real torch exact/Nyström/RFF kernel. `pirs/nulls.py` implements the full §6.8 table, the §6.9 season-preservation assertion (a §10 abort), and the `<5 blocks → descriptive` fallback. FDR (`pirs/fdr.py`) has BH and BY. Cross-fitting (`pirs/crossfit.py`) preserves spatial/temporal blocks (§6.6.1). One small item: §6.8 lists **Storey q** for the facility-stock regime; only BH/BY are implemented (`PIRS-FDR-02`, S3) — add Storey or annotate the MSD that facility-stock uses BY.

### 7.2 GLM family coverage — S1

**ID `PIRS-FAM-01` · MSD §6.2 · Severity: S1.**
The real engine `compute/glm.py` is a genuine IRLS GLM with eight families and correct §6.5 residual triples: `gaussian_identity`, `poisson_count_with_log_offset`, `negative_binomial`, `quasi_poisson`, `hurdle_poisson`/`hurdle_nb`, `gamma`/`sih_gamma_cost_component`, `binomial_proportion`. So §6.2 **count routing is implemented** (overdispersion + zero-inflation handled). **Missing** (zero occurrences in the dump): `beta_binomial`, `dirichlet`/`multinomial` (simplex/compositional outcomes, e.g. race or cause-composition shares), `lognormal`, two-part beyond hurdle, and `student_t`.

**Remediation.** Add to `compute/glm.py` (IRLS triples) and to the model registry: `beta_binomial` (overdispersed proportions — important for the many share outcomes: LBW, cesarean, ICU-use, sanitation coverage), `dirichlet`/`multinomial_logit` (simplex outcomes), `lognormal`/two-part (skewed cost components with zero mass), and optionally `student_t` (robust panels). Route them in `select_count_family`'s sibling selectors per §6.2's outcome-shape logic (zero-mass → two-part/hurdle; bounded proportion → beta-binomial; simplex → Dirichlet).

### 7.3 Spatial effect modes — **S1, definitive gap**

**ID `PIRS-SPAT-01` · MSD §6.3 · Severity: S1.**
`ICAR`, `UF_FE`, and `municipality_FE` appear **nowhere** in the codebase. The GLM is non-spatial (`model_execution.py` describes a "deterministic local residualization model"). §6.3 mandates `SpatialEffectMode ∈ {none, UF_FE, municipality_FE, ICAR}` with the precedence: `fast ⇒ UF_FE`; `|T|≥10 ∧ ζ_Y≤0.10 ⇒ municipality_FE`; `|T|<10 ∨ ζ_Y>0.10 ⇒ ICAR`; `MoranI≈0 ⇒ none`. Spatial dependence is acknowledged only in the state tensor (Moran's I) and HSIC nulls — but the *parametric estimates themselves are uncorrected for spatial autocorrelation*, biasing standard errors on essentially every spatial outcome.

**Remediation.**
1. Add a `SpatialEffectMode` selector (`pirs/design.py` or a new `pirs/spatial.py`) implementing the §6.3 precedence from budget, `|T|`, `ζ_Y`, and `MoranI(Y)`.
2. Implement the modes:
   - `UF_FE`/`municipality_FE` — add fixed-effect dummy columns to the design matrix (cheap; the IRLS GLM already supports arbitrary covariates).
   - `ICAR` — an intrinsic conditional autoregressive spatial random effect using the adjacency from `geo/adjacency.py`. For a first pass without a full Bayesian backend, implement the ICAR precision as a penalty term in a penalized-IRLS (GMRF prior), or document a deferral and **gate** any field requiring ICAR to `descriptive_association_only` until implemented (so SEs are never silently wrong).
3. Record the chosen mode in `ModelAssociations` and the variable dictionary.

### 7.4 Stale modeling metadata — S3

**ID `PIRS-REG-01` · Severity: S3.**
`registries/model_registry.yaml` lists only two families (`gaussian_ols`, `poisson_log_offset`) and `pirs/families.py::family_for_outcome` returns only gamma-or-gaussian — both contradict the eight-family `compute/glm.py`. `pirs/models.py` similarly references only a poisson model id. *Fix:* regenerate `model_registry.yaml` from the `glm.py` family list (single source of truth), delete or rewrite `families.py::family_for_outcome` to call `glm.select_count_family` and the new proportion/simplex selectors, and remove the vestigial `pirs/models.py` poisson stub if unused.

---

## 8. Output contract (§7–§8) — COMPLIANT (keep), with two tasks

**ID `OUT-01` · MSD §8.1 · Severity: none.**
`output/schemas.py::OUTPUT_BUNDLE_FILES` enumerates exactly 17 keys matching §8.1; `output/validate.py` enforces presence, the V_fields/E_DAG/Q_tensor/VariableDictionary column contracts, the race-bridge metadata contract (§8.6), and reproducibility hashes. Keep.

**Tasks:**
- **`OUT-02` (S3)** — `config/registries/output_schema.yaml` is a "Slice 0 scaffold" that does not list the 17 keys (the real contract is in `output/schemas.py`). Either populate it as the authoritative source and have `output/schemas.py` load it, or delete it and stop referencing it in validator file lists. Do not keep a stub that *looks* like the contract.
- **`OUT-03` (S2)** — fold the Run Profile `PROFILE_NONEMPTY` + empty-by-profile warning rule (§3.2.2) into `output/validate.py`. This is what makes scoped runs *honestly* valid.

---

## 9. Registry hygiene (`config/registries/`) — S3 cluster, fix together

Determined by cross-referencing each loader's opened filename against the file tree and timestamps.

**ID `REG-DEAD-01` · Severity: S3 — four orphan alias stubs.**
`aggregation_registry.yaml`, `carrier_registry.yaml`, `provenance_registry.yaml`, `unit_registry.yaml` are dead. The loaders (`registries/{carrier,unit,provenance,aggregation}.py`) open the short-name files (`carrier.yaml`, etc.). Each orphan self-documents: *"Macro-Slice 27A compatibility alias. Canonical X semantics live in X.yaml; this file remains required by registry validators."*

**ID `REG-VALIDATOR-01` · Severity: S3 (latent bug) — the validator points at the dead files.**
`registries/validators.py::validate_registry_tree` lists the four `_registry.yaml` orphans as *required* and does **not** list the live short-name files. So the validator enforces dead files and ignores the real ones — a circular dependency keeping the orphans alive.

**Combined fix for `REG-DEAD-01` + `REG-VALIDATOR-01`:**
1. In `validate_registry_tree`, replace `carrier_registry.yaml`/`unit_registry.yaml`/`aggregation_registry.yaml`/`provenance_registry.yaml` with `carrier.yaml`/`unit.yaml`/`aggregation.yaml`/`provenance.yaml`.
2. Delete the four orphan stubs.
3. Run the registry-tree validator + full test suite to confirm nothing else referenced them.

**ID `REG-STALE-LIST-01` · Severity: S3.**
A `REGISTRY_FILES = ('sidra_registry.yaml', 'sidra.yaml', 'sidra_tables.yaml')` tuple references three files that **do not exist**. The real SIDRA registries are `sidra_views.yaml`, `sidra_category_maps.yaml`, `sidra_stitching.yaml`, `sidra_regime_registry.yaml`, `sidra_compendium.json`, `sidra_table_seed.jsonl`. *Fix:* correct the tuple to the real names (or delete if unused).

**ID `REG-SCAFFOLD-01` · Severity: S3.**
Near-empty scaffold registries: `output_schema.yaml` (see `OUT-02`), and several Jun-06 seeds (`sidra_stitching.yaml`, `sidra_category_maps.yaml`, `sidra_regime_registry.yaml`, `composite_decoders.yaml` at 284–909 B). *Action:* either populate to their MSD role (esp. `composite_decoders.yaml`, which `SHE-DEC-03` needs) or mark clearly as intentional seeds in a `STATUS:` field and exclude from the "required & populated" validator set.

**ID `REG-EOL-01` · Severity: S3.**
Line-ending drift (CRLF in `carrier.yaml`/`unit.yaml`/`source_fields.yaml`; LF elsewhere). *Fix:* add a `.gitattributes` normalizing YAML to LF and re-save, to keep diffs clean.


---

## 10. Cross-cutting structural issues

**ID `XCUT-01` (S1) — Two-tier "compliant spec / reduced production" pattern.** The dominant signal. The contract layers are faithful; the production layers are reduced. Every remediation should be framed as *making the production tier honor the contract the upper tier already encodes*. The matching test discipline (§12) is: for every faithful contract, add a test that fails if the production tier doesn't satisfy it (e.g. "no admissible canonical column is 100% null", "n_eff reflects Moran's I", "race CV is bootstrap-derived").

**ID `XCUT-02` (S3) — Hardcoded duplication of registry-owned logic.** The four `normalize_*_events` re-implement `datasus/decoders.py` inline; `declarative_normalize.py` (the registry-driven path) is effectively dead in the table pipeline. Resolved by `SHE-NORM-01` Path A. Until then, **any decoder bug must be fixed in two places**, which is itself a defect.

**ID `XCUT-03` (S0 root) — Layering violations (SHE doing EFG's job).** SINASC SHE emits outcome indicators (`SHE-SINASC-01`); CNES SHE emits a generic total (`SHE-CNES-01`). Principle to enforce in review: **SHE emits typed primitives + states + provenance; the EFG derives indicators, rates, and composites.** Add a lint/review checklist item: "Does this SHE output encode a threshold, ratio, or composite? If yes, it belongs in EFG."

**ID `XCUT-04` (S3) — Dead/contradictory metadata.** Orphan registries (`REG-DEAD-01`), mis-targeted validator (`REG-VALIDATOR-01`), stale file lists (`REG-STALE-LIST-01`), 2-entry model registry vs 8-family GLM (`PIRS-REG-01`), scaffold output schema (`OUT-02`). These collectively make the registry tree an unreliable description of the system — fix as one hygiene sweep.

**ID `XCUT-05` (S2) — Silent-absence anti-pattern.** Multiple paths drop data without a typed reason (nulled canonical columns, inert context, skipped ST-DFM, empty substrate classes). The MSD's closing invariant forbids "missingness with silence." The Run Profile's empty-by-profile rule (§3.2.2) plus the "typed FailedBranch/quarantine, never silent" test (`EFG-REG-01`) are the systemic countermeasures.

---

## 11. Consolidated compliance matrix

| Area | MSD § | Contract tier | Production tier | Severity | Finding IDs |
|---|---|---|---|---|---|
| Composite decoders | 2.4.0 | Faithful | Bypassed by normalizers | S0/S2 | SHE-DEC-01/02/03, SHE-NORM-01 |
| SIM-DO fields/topology | 2.4.1, 2.5.2 | Registry rich (37 fields) | Chain/associated/observer nulled | S0 | SHE-NORM-01 |
| SIH-RD fields | 2.4.2, 2.5.3 | Cost components ✓ | No secondary dx/ICU/movement/CNPJ | S0 | SHE-NORM-01 |
| SINASC fields | 2.4.3 | Flags (co-designed) | Primitives dropped; layering breach | S0 | SHE-SINASC-01, SHE-NORM-01 |
| CNES capacity/linkage | 2.4.4, 2.4.0.5 | — | Generic total; no CNPJ; opaque JSON | S0 | SHE-CNES-01/02/03 |
| Zero-variance drop | 2.4.0.6 | `she/zero_variance.py` present | Verify wired pre-design-matrix | S2 | (verify) |
| Population objective | 2.8.2–2.8.9 | **Excellent** | — | none | SHE-POP-01 |
| Population solvers | 2.8.12 | 10⁷ guard ✓ | ADMM scaffold; no primal-dual/state-space | S1 | SHE-POP-02 |
| SIDRA regime | 2.9 | Faithful | Inert in compile | S1 | SIDRA-CTX-01 |
| SIDRA stitch/project/bound | 2.12 | Modules exist | Not wired to EFG | S1 | SIDRA-CTX-01/02 |
| ST-DFM | 2.10 | Cert faithful | Skipped in compile | S1 | SIDRA-CTX-01 |
| EFG legality (7-part) | 3.8 | **Faithful** | — | none | EFG-LEG-01 |
| Axes high-card bound | 3.8.2, 2.12.3 | Partial | Not enforced on legality path | S1 | SIDRA-CTX-02 |
| Declaration/race gate | 3.8.8, 4.3 | Faithful | Permissive on missing axis | S2 | EFG-DECL-01/02 |
| EFG executor | 3.9–3.11 | **Faithful** | — | none | EFG-EXEC-01 |
| State tensor | 3.12 | Missing CV/Moran/roughness/entropy as required | — | S2 | EFG-Q-01 |
| Compression | 3.15 | Faithful | — | none | EFG-CMP-01 |
| Core seed reachability | 3.10 | Defined | Several V_* unreachable until SHE fixes | S1 | EFG-REG-01 |
| Race bridge | 4.5–4.7 | Guards/labels ✓ | Fixed-W only; CV mis-defined | S2 | RACE-01 |
| HSIC/nulls/FDR/crossfit | 6.5–6.9 | **Faithful** | — | none/S3 | PIRS-HSIC-01, PIRS-FDR-02 |
| GLM families | 6.2 | — | Counts+gamma+binomial ✓; no simplex/beta-binom/lognormal | S1 | PIRS-FAM-01 |
| Spatial effect modes | 6.3 | — | **Entirely absent** | S1 | PIRS-SPAT-01 |
| Modeling registry | 6.2 | Stale (2 vs 8 families) | — | S3 | PIRS-REG-01 |
| Output 17-key | 8.1 | **Faithful** | — | none/S2/S3 | OUT-01/02/03 |
| Registry hygiene | — | Orphans + bad validator | — | S3 | REG-DEAD/VALIDATOR/STALE/SCAFFOLD/EOL |
| Run-profile contract | 1.2/8.1/9/10 | Absent (tension) | Resolved silently | S0 (process) | §3.2 design |

---

## 12. Prioritized remediation roadmap

Phased so each phase produces a *more honest* system even if later phases slip. Within a phase, IDs are roughly dependency-ordered.

### Phase 0 — Make the system honest (do first; small, high-leverage)
- `REG-DEAD-01` + `REG-VALIDATOR-01` + `REG-STALE-LIST-01` — registry hygiene sweep (an afternoon; removes traps).
- §3.2 **Run Profile contract** — add `run_profile` to `UserIntent`, branch compile stages, implement `PROFILE_NONEMPTY` + empty-by-profile in `output/validate.py` (`OUT-03`). This *immediately* makes the current scoped run formally valid and eliminates `XCUT-05` silent-absence for substrate classes.
- `PIRS-REG-01` — reconcile model registry with `glm.py`.

### Phase 1 — Restore source reality (the S0 core; unblocks the most science)
- `SHE-NORM-01` (Path A preferred) — registry-driven, complete normalization; emit every admissible canonical column and all §2.3 states. **This is the highest-impact fix in the report.**
- `SHE-SINASC-01` — emit SINASC primitives; move thresholds to EFG.
- `SHE-CNES-01/02/03` — drop generic total, emit typed bed indices, apply `filter_cnpj`, stop null→0.
- `SHE-DEC-02/03` — CNPJ check digits; registry-driven SIH age map.
- `EFG-REG-01` — integration test: every `V_core` materializes or emits a typed failure.
- `EFG-Q-01` — complete the state tensor (CV/Moran/roughness/entropy) and fix `n_eff`.

### Phase 2 — Correct the inference layer
- `PIRS-SPAT-01` — spatial effect modes (at minimum UF_FE/municipality_FE; ICAR or an explicit gate-to-descriptive).
- `PIRS-FAM-01` — beta-binomial + simplex (+ lognormal/two-part) families.
- `RACE-01` steps 1–2 — local-π crosswalk + corrected CV (correctness before simulation).
- `EFG-DECL-02` — fail-closed on race-stratified operands missing axis metadata.

### Phase 3 — Light up the `contextual` profile
- `SIDRA-CTX-01` — wire compendium → regime → stitch → projection → bounded pushforward → EFG context nodes; run gated ST-DFM in compile.
- `SIDRA-CTX-02` — enforce the high-cardinality axis bound and its §10 abort.
- `RACE-01` steps 3–4 — posterior simulation + partial-ID bounds for high-stakes race claims.

### Phase 4 — Light up the `full` profile (Brazil scale)
- `SHE-POP-02` — ADMM, then state-space smoother; resolve or remove primal-dual from the MSD.
- Demographic tensor over full `(s,t,a,x,r)` with census strata; race bridge over real composition.

### Cross-phase test discipline (apply continuously)
For each "Faithful" contract, add a guard test that fails if production diverges:
- No admissible canonical column is 100% null on a representative fixture (`SHE-NORM-01`).
- `n_eff` reflects the Moran's-I correction (`EFG-Q-01`).
- `race_bridge_cv` is bootstrap/variance-derived, not cross-category spread (`RACE-01`).
- Standard/deep HSIC never consumes in-sample residuals (already a §10 abort; assert it fires).
- Every `V_core` field materializes or yields a typed `FailedBranch` (`EFG-REG-01`).
- Output bundle: all 17 keys present **and** profile-required keys non-empty, with empty keys carrying an `empty_by_profile` reason (`OUT-03`).

---

## 13. MSD amendments required (spec must change, not just code)

These are cases where the MSD itself is under- or over-specified relative to a shippable system. Amend the MSD alongside the code so the two never silently disagree.

1. **Add §1.5 Run Profiles** (`core_vital`/`contextual`/`full`) and annotate §9.2 steps 16–20 and the §10 aborts with their governing profile. (§3.2 here is a drop-in draft.)
2. **Reconcile §2.8.2 independent-denominator default with §2.6.5/§3.10.4 maternal-child preconditions** — mark perinatal/maternal/lagged fields as `contextual`+ and define their `core_vital` fallback state. (§3.3 here.)
3. **§2.8.12 solver table** — either commit to implementing `primal-dual_sparse` and a real `state-space_smoother`, or remove those bands so the spec matches the code.
4. **§6.8 FDR** — state explicitly that facility-stock uses BY if Storey-q is not implemented (`PIRS-FDR-02`).
5. **§8.1** — restate as "all 17 keys present; profile-required keys non-empty; empty keys carry an explicit reason" (the anti-silence guarantee).

---

## 14. Appendix A — Files of concern (quick index)

| Path | Issue | Finding |
|---|---|---|
| `datasus/normalize.py::normalize_sim_do_events` | Inline decoders; nulls chain/associated/observer | SHE-NORM-01 |
| `datasus/sih_normalize.py::normalize_sih_rd_events` | No secondary dx/ICU/movement/CNPJ | SHE-NORM-01 |
| `datasus/sinasc_normalize.py::normalize_sinasc_events` | Flags not primitives; PESO decoder unused | SHE-SINASC-01 |
| `datasus/cnes_normalize.py::normalize_cnes_st_events` | Generic total; no CNPJ; null→0 | SHE-CNES-01/02/03 |
| `datasus/declarative_normalize.py` | Registry-driven path, effectively dead | XCUT-02 / SHE-NORM-01 |
| `datasus/decoders.py::filter_cnpj`, `decode_sih_age` | No check digits; hardcoded unit map | SHE-DEC-02/03 |
| `she/population/sparse_admm.py`, `state_space.py` | Scaffold / plumbing only | SHE-POP-02 |
| `workflows/compile.py::_run_compile_impl` | ST-DFM skipped; context not wired | SIDRA-CTX-01 |
| `efg/legality.py` (axes term) | High-card bound not enforced | SIDRA-CTX-02 |
| `efg/declaration.py` | Permissive on missing race axis | EFG-DECL-02 |
| `efg/race_bridge.py::fixedc_dynamic_weight_bridge` | Fixed-W only; CV mis-defined | RACE-01 |
| `compute/glm.py` | Missing simplex/beta-binom/lognormal | PIRS-FAM-01 |
| `pirs/*` | No spatial effect modes anywhere | PIRS-SPAT-01 |
| `pirs/families.py`, `pirs/models.py`, `model_registry.yaml` | Stale vs glm.py | PIRS-REG-01 |
| `output/validate.py` (Q_tensor required cols) | CV/Moran/roughness/entropy not required | EFG-Q-01 |
| `registries/validators.py::validate_registry_tree` | Requires dead `_registry.yaml` files | REG-VALIDATOR-01 |
| `config/registries/{aggregation,carrier,provenance,unit}_registry.yaml` | Orphan alias stubs | REG-DEAD-01 |
| `config/registries/output_schema.yaml` | Scaffold; not authoritative | OUT-02 |

## 15. Appendix B — Glossary of MSD objects referenced

- **`B` / substrate classes** — `B_official/harmonized/deflated/reconstructed/latent/cross-sectional/excluded` (§2.1).
- **`Δ` legality** — seven structural terms (support, axes, carrier, unit, aggregation, provenance, quality) × declaration gate (§3.8).
- **Carrier** — typed numerator/denominator semantics (Deaths, HospitalAdmissions, LiveBirths, Population, FacilityCapacityVector_k, …; §3.3).
- **`Q(v)` state tensor** — 14-component diagnostic vector driving state classification (§3.12).
- **`Bridge_R`** — Bayesian ecological race-axis bridge (§4.5).
- **HSIC** — Hilbert-Schmidt Independence Criterion residual scanner (§6.7).
- **ST-DFM** — Spatio-Temporal Dynamic Factor Model for gated latent context reconstruction (§2.10).
- **17-key bundle** — the immutable run output contract (§8.1).
- **Run Profile** — *proposed* capability tier (`core_vital`/`contextual`/`full`) introduced in §3.2 of this report to resolve the §1.2/§8.1/§9/§10 vs §3.10.8 tension.

---

*End of report. Cite finding IDs in commits and PRs. When a fix lands, update the matrix row's severity to `RESOLVED (commit …)` so this document tracks live compliance.*
