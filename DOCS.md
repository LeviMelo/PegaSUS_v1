# PegaSUS Documentation Index & Lineage

The planning docs are **strata** — deposited over time, each newer generation absorbing
what came before. Reading an old stratum as authoritative is how contradictory refactors
happen. This index is the single source of truth for *which doc governs what*, and it is
the thing to update when a doc's status changes.

## Precedence (when two live docs disagree)

1. **File/module disposition** (what to delete, move, keep): `PEGASUS_REFACTOR_MASTER_PLAN.md`
   wins — it is the only plan whose every claim was verified against the live tree.
2. **Registry / REG-07 direction**: `PEGASUS_MSD_III.md` §II.1 wins over the
   `OPERATIONAL_IMPLEMENTATION_PLAN.md` §REG-07 (whose "mechanical, one source at a time"
   framing is wrong — REG-07 is a decode-validated contract merge; see the master plan §2).
3. **Run-legality / mandatory-chain semantics**: `MSD-III` §VII.2 (DataScope/ExecutionStage +
   typed empty keys) supersedes `MSD-I` §9's absolute "closed mandatory chain" language.
4. Otherwise the **newest** doc (by the table below) governs; older strata are reference only.

## Governing specifications (current — obey, but verify against code)

| Doc | Role | Notes |
|---|---|---|
| `PEGASUS_MSD_III.md` | **The architecture spec of record.** | Newest MSD. §II.1 = registry direction; §III = LDO; §V = compute/scale; §VII = run legality; §XI = phases/acceptance. Known internal tensions logged in the master plan §3 — resolve, don't obey blindly. |
| `PEGASUS_OPERATIONAL_IMPLEMENTATION_PLAN.md` | **The TDD** — build/ticket plan (MII-*, REG-07, PANEL-01, LDO-06…). | Subordinate to MSD-III on architecture and to the master plan on file disposition. Its file-count/"DEL" claims are unreliable. |
| `PEGASUS_REFACTOR_MASTER_PLAN.md` | **Refactor plan of record**, with a live EXECUTION STATUS ledger. | Source-verified; carries the `[PLAN-CONFLICT]` reconciliations. |

## Foundational / code-referenced (keep — cited by code, do not delete)

| Doc | Why kept |
|---|---|
| `PEGASUS_MSD_I.md` (7015 L) | Foundational data-plane spec; cited by MSD-III "authoritative by reference". §9 superseded by MSD-III §VII.2 for run-legality. |
| `PEGASUS_MSD_II.md` | Cited in code as `MSD-II §II.1/§II.4` (registry authority, SpatialWeightGraph) across ≥5 modules. |
| `PEGASUS_DISEASE_SEMANTIC_AXIS.md` | The disease-axis spec (§II.13); current. See also `docs/ICD_LIBRARY_REVIEW.md`. |
| `PEGASUS_ARCHITECTURE_ADDENDUM.md` | **Superseded by MSD-III**, but referenced by MSD-II (`ARCH-*` items / four pillars). Retained only for that cross-reference; treat as historical. |

## Reference material (keep — not plans)

| Doc | Role |
|---|---|
| `PEGASUS_COMPLIANCE_AND_REMEDIATION.md` | Finding-ID dictionary (S0/S1…) retained by MSD-III §0.1. **Not a plan of record.** |
| `PEGASUS_COMPLETION_ROADMAP.md` | **The settled completion roadmap (2026-07-09)** — full-scale-default reframe, RaceBridge ecological redesign, data-layer first-class, storage contract, phased plan. The forward plan of record. |
| `PEGASUS_REPO_HEALTH_ASSESSMENT.md` | 2026-07-08/09 repo-health review (5 dimensions) + 4 improvement deep-dives + tiered remediation. Findings feeding the roadmap. |
| `PEGASUS_OUTPUT_QUERY_LAYER.md` | FEAT-P3+P4 Output Query Layer design (self-describing hypotheses export; rate-recompute pruned). |
| `docs/ICD_LIBRARY_REVIEW.md` | ICD/CID library health review (simple-icd-10 + icd-mappings; the WHO-2019 A90/A91/U06 gap + remediation). |
| `DATASUS_DESC.md`, `DATASUS_CONPENDIUM.md`, `SIDRA_DESC.md`, `SIDRA_COMPENDIUM.md` | Data-source field/table references. |
| `README.md` | Repo entry point. |

## Active subordinate plans (close into MSD-III when their work lands)

| Doc | Closes into |
|---|---|
| `PEGASUS_COMPUTE_BUILD_OPTIMIZATION_PLAN.md` | MSD-III §V (POP-02 GPU/blocked build). |
| `PEGASUS_STORAGE_OPTIMIZATION_PLAN.md` | MSD-III §V (STORE-02 lazy views). |
| `PEGASUS_OUTPUT_QUERY_LAYER.md` | MSD-III §VIII (FEAT-P3 export + FEAT-P4 multi-denominator). |

## Deleted (git history preserves them; do NOT resurrect as guidance)

- `docs/*.md` — 13 five-line Jun-6 stubs describing the **dead slice-bundle architecture** (mentioned PIRS, old EFG); `docs/production_boundaries.md` + `docs/pegasus_codebase_contract.yaml` (the boundary-audit tooling, now exterminated).
- `TDD_old.md` (3264 L) — the previous TDD, superseded by `OPERATIONAL_IMPLEMENTATION_PLAN.md`.
- `HANDOFF.md` — ephemeral handoff note.
- `_review/*` — Jun-25/26 remediation snapshots (`MSD_REMEDIATION_PLAN`, `MSD_CONVERGENCE_AUDIT`, `SCALE_PLAN`, `slice28za_context_*`), long since absorbed into MSD-III.

## Also removed from the repo

- `scripts/dev/**` (~57.5k LOC): chatbot-era one-shot "updater" codemods + per-slice boundary
  "audit" linters. They mutate/lint a *past* state of `src/`; re-running corrupts current code.
  The 6 real tools live at `scripts/` root (`doctor.py`, `validate_registry_hashes.py`,
  `build_municipality_crosswalk.py`, `build_sidra_compendium_registry.py`, `check_cuda.py`,
  `check_r_microdatasus.R`). **No functional source belongs under `scripts/` — ever.**
