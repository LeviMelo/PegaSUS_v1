# National pancreatic cancer (C25) study — how it was produced

The exhaustive results are in `REPORT.md` (+ `c25_results.json`, `report.html`).

## Reproduce / extend
- **Descriptive + inference (2021–2022, on-disk data):** `python studies/pancreatic_c25_national/analysis.py`
  (leverages the compiled run `data/runs/national_c25_poptensor/` + the FAL-POP national population
  tensor + `pegasus.measurement.age_standardization`).
- **Full 1996–2023 temporal range** (trends/APC): run the SIM fetch in a *persistent terminal*
  (the multi-hour R/microdatasus fetch could not persist under the agent's Windows background execution):
  `python -m pegasus.cli run --intent config/intents/national_sim_fullrange_fetch.json`
- **Socio-economic determinants** (income/education/sanitation via the 94-table SIDRA compendium,
  LDO determinant edges): `python -m pegasus.cli compile --intent config/intents/national_pancreatic_c25_determinants.json`
  then the investigate stage. Runs ~11 min+ (the 563s population solve dominates — cache it as a
  foundational asset to reuse). `contextual` profile + dropping `no_sidra_compendium` + `execution_stage: investigate`
  is the 3-field change that unblocks the determinant question.

## Named scientific limitation
No smoking covariate exists in SIDRA (the dominant modifiable C25 risk factor) — tobacco-attributable
fraction must come from VIGITEL/PNS, outside this pipeline.
