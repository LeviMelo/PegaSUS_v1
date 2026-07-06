# ICD-Code Technical Backbone Review — Disease Semantic Axis

Scope: the ICD/CID substrate under `src/pegasus/disease/` (`icd_adapter.py`,
`concept_registry.py`, `graph.py`). Reviews the two ICD libraries the axis depends on
(`simple-icd-10`, `icd-mappings`), characterizes the dengue/arbovirus coverage gap, and
gives a concrete remediation recommendation.

Everything below was empirically verified against the **installed** `pegasus` env
(`simple_icd_10==2.1.1`, `icd-mappings==0.6.2`), not just from documentation.

---

## 1. `simple-icd-10` — the hierarchy backbone

- **PyPI / GitHub / docs:** [pypi.org/project/simple-icd-10](https://pypi.org/project/simple-icd-10/) ·
  [github.com/StefanoTrv/simple_icd_10](https://github.com/StefanoTrv/simple_icd_10) ·
  [simpleicd10.stefanotravasci.it](https://simpleicd10.stefanotravasci.it/)
- **Installed version:** `2.1.1`.
- **Edition shipped:** **WHO ICD-10, 2019 version.** Codes + descriptions are scraped
  from the WHO site and embedded as an XML tree in the package `data/` folder (offline, no
  network, deterministic). Catalogue size measured live: **12,542 nodes** (chapters, blocks,
  categories, subcategories).
- **API (all present and used by the adapter):** `is_valid_item`, `add_dot`/`remove_dot`,
  `get_ancestors`, `get_descendants`, `get_parent`, `get_children`, `is_leaf`,
  `get_nearest_common_ancestor`, `is_ancestor`/`is_descendant`, `is_chapter`/`is_block`/
  `is_category`/`is_subcategory`, `get_description`, `get_index`, `get_all_codes`.
  This is a full closure/hierarchy toolkit — exactly what the axis is designed around
  (full catalogue + operators, not a hand-curated selection). No API gap.
- **Known limitations:**
  - It is **WHO ICD-10, not ICD-10-CM and not Brazilian CID-10.** No Portuguese
    descriptions; no CID-10-specific subcategories.
  - It is pinned to the **2019 WHO edition**, whose arbovirus block was restructured
    relative to older editions and relative to CID-10 (see §3). No multi-release support.
  - Descriptions are English only.

## 2. `icd-mappings` (`icdmappings`) — the grouper backbone (already a dependency)

- **Installed version:** `0.6.2`. Used by `concept_registry.py` via `Mapper().map(...)`.
- **Mappers available from `icd10`** (verified live): `icd9`, `block`, `chapter`, `ccsr`,
  `ccir`, `ccc_category`, `ccc_subcategory`.
- **Important discrepancy vs. the code comments:** `concept_registry._GROUPERS` references
  a `"ccir"` grouper as the "Chronic Condition Indicator" and calls `map(..., target='cci')`
  in one probe path. The installed library exposes **`ccir`** (Chronic Condition Indicator
  *Refined*), **not** `cci` — a bare `target='cci'` raises. The registry's actual config uses
  `ccir`, so the live path is fine; but any code/comment implying a `cci` target is wrong for
  0.6.2 and should be scrubbed to avoid a future foot-gun.
- **Coverage advantage:** its internal code table is broader than WHO-2019 and **includes the
  dengue codes** (see §3). This is the lever for remediation.

---

## 3. THE GAP: dengue / arbovirus codes absent from WHO-2019

**Root cause.** The Brazilian CID-10 (DATASUS V2008) arbovirus block is
`A90–A99` and **begins at A90 (Dengue clássico), A91 (Dengue hemorrágico)**
([DATASUS a90_a99](http://www2.datasus.gov.br/cid10/V2008/WebHelp/a90_a99.htm),
[Portal Afya A90](https://portal.afya.com.br/codigos/I/grupo/A90-A99/cid/A90)).
The **WHO 2019** edition that `simple-icd-10` ships restructured this block: its block node
is literally **`A92-A99`** — **A90 and A91 do not exist as nodes at all.** In WHO-2019, dengue
was folded into `A97` ("Dengue", with `A97.0–A97.9`), so the standalone A90/A91 categories
that all DATASUS/SIH/SIM/SINAN data uses are simply not in the tree.

**Empirically measured against `simple_icd_10==2.1.1`:**

| Code | Meaning (CID-10 BR) | `is_valid_item` (WHO 2019) | `icd-mappings` block | Notes |
|------|---------------------|:---:|------|-------|
| **A90** | Dengue clássico | **False** | `A90-A99` | absent from WHO tree; core arbovirus code |
| **A91** | Dengue hemorrágico | **False** | `A90-A99` | absent from WHO tree |
| A92 | (block parent) Chikungunya/Zika | True | `A90-A99` | WHO block node is `A92-A99` |
| A92.0 | Chikungunya virus disease | True | `A90-A99` | present |
| A92.5 | Zika virus disease (WHO) | True | `A90-A99` | present (WHO spelling) |
| A92.8 | Zika (CID-10 BR sub) | True | `A90-A99` | present |
| A95 | Febre amarela (yellow fever) | True | `A90-A99` | present |
| **U06** | Zika emergency use (some editions) | **False** | `U00-U49` | absent from WHO-2019 U-block |
| U07.1 | COVID-19 | True | `U00-U49` | present |
| B34(.9) | Viral infection, unspecified | True | `B25-B34` | present |
| Q02 | Microcephaly | True | `Q00-Q07` | present |

So the **entire gap surface** that matters epidemiologically for Brazil is small and precise:
**A90, A91** (dengue — the highest-burden arbovirus in the country) and **U06** (a Zika
emergency-use code seen in some vintages). Chikungunya (A92.0), Zika (A92.5/A92.8), yellow
fever (A95), and microcephaly (Q02) are all **present and correct** in WHO-2019. This is not a
broad catalogue failure — it is a handful of load-bearing codes.

### How the axis handles it today (this is already partly designed-for)

The adapter does **not** silently break on these. Verified live through
`pegasus.disease.icd_adapter.code_info`:

- `A90 → status=source_system_specific, chapter=I, category=A90, ancestors=()`
- `A91 → status=source_system_specific, chapter=I, category=A91, ancestors=()`
- `U06 → status=source_system_specific, chapter=XXII, category=U06, ancestors=()`

The `_CID10_CHAPTERS` range table in `icd_adapter.py` correctly resolves the **chapter** for
these WHO-absent codes, and the code is typed `source_system_specific` and **never coerced**
to a WHO subcode — exactly the §II.13.2 guard. `concept_registry.assertions_for_code` then still
emits a chapter assertion (`source=cid10_catalog`) and a `source_specific_*` assertion, and the
downstream `projection_status` is degraded, so nothing is silently dropped.

### What is nonetheless lost for A90/A91/U06

Because these are absent from the WHO tree, the WHO-backed operators return empty/false:
`get_ancestors → ()`, `get_descendants → ()`, `is_leaf → False`, and **`code_info.block` is
`None`**. Concretely this degrades two things:

1. **`DiseaseGraph._structural_weight`** (`graph.py`): dengue↔dengue and dengue↔other-arbovirus
   edges lose the **0.5 "same block" tier**. Two dengue variables still get 1.0 (same 3-char
   category) and dengue↔A92 chikungunya still get 0.25 (same chapter), but the biologically
   correct "same arbovirus block" weight of 0.5 collapses to 0.25. The `L_D` prior over the
   arbovirus cluster is therefore **weaker than it should be** for exactly the diseases Brazil
   cares most about.
2. **`nearest_common_ancestor` / `distance`**: dengue has no WHO ancestor chain, so NCA-based
   hierarchy distance to any other code is `None` for A90/A91 — hierarchy distance is unavailable
   for the dengue nodes.

`icd-mappings@0.6.2`, already a dependency, **does** know A90/A91/U06 and returns the correct
block (`A90-A99`, `U00-U49`) and chapter for all of them — so the missing block is recoverable
without adding any new dependency.

---

## 4. Alternative / complementary libraries assessed

| Library | Edition | Hierarchy ops | Has A90/A91? | CID-10 / pt-BR | Offline | Fit |
|---------|---------|---------------|:---:|:---:|:---:|-----|
| **simple-icd-10** (current) | WHO ICD-10 **2019** | full (anc/desc/parent/children/leaf/NCA) | **No** | No | Yes | Backbone; only the A90/A91/U06 hole |
| **icd-mappings 0.6.2** (current dep) | ICD-10-CM-ish grouper table | block/chapter/ccsr/ccir/ccc + icd9 | **Yes** (block+chapter+ccsr) | No | Yes | **Ideal gap-filler; already installed** |
| **simple-icd-10-cm** (StefanoTrv) | ICD-10-**CM** (US, 2026 by default, `change_version`) | same API family + excludes/7th-char | **Yes** (A90.x present in CM) | No | Yes | Same author, drop-in-ish API — but CM ≠ WHO ≠ CID-10; deeper US-specific tree would *change* every ancestor/leaf result, not just fill A90 |
| **rmnldwg/icd** ([github](https://github.com/rmnldwg/icd)) | ICD-10 **multi-release** (`get_codex(release="2019")`, also older) | `exists/get/ancestry/tree/children` | Depends on release; older releases have A90/A91 | No | Yes | Multi-release is attractive, but different API surface (would require rewriting the adapter) and no pt-BR |
| **icd-codex / icdcodex** ([github](https://github.com/icd-codex/icd-codex)) | ICD-9/10 networkx tree + node2vec | graph/embedding | partial | No | Yes | Embedding-oriented; authors themselves discourage node2vec — not a catalogue authority |
| **WHO ICD API** ([icd.who.int/icdapi](https://icd.who.int/icdapi)) | Authoritative WHO, all releases + ICD-11 | full REST | Yes (pick release) | Multilingual incl. pt | **No — network + OAuth token** | Authoritative but violates the offline/deterministic design; not suitable as the hot-path backbone |
| **icd10-cm / others** | ICD-10-CM lookups | validity/description | Yes | No | Yes | Lookup-only, weaker hierarchy ops than simple-icd-10 |

**None of the offline libraries ships Brazilian CID-10 with Portuguese descriptions.** The only
pt-BR authority is DATASUS itself (`www2.datasus.gov.br/cid10/V2008/...`, the CID-10 XML/CSV) or
the WHO ICD API (network). So "switch to a library that natively has CID-10" is not an available
option — CID-10 authority in this repo already correctly lives in the `_CID10_CHAPTERS` table and
`config/registries/.../disease/`, and DATASUS remains the source of truth.

---

## 5. Recommendation

**Do not switch the backbone. Keep `simple-icd-10` (WHO-2019) and layer a narrow, deterministic
fallback that fills the block/hierarchy for the handful of CID-10 codes absent from WHO-2019 —
sourcing that fallback from `icd-mappings@0.6.2`, which is already a dependency and already
resolves A90/A91/U06.** Rationale: the gap is tiny and precise (dengue A90/A91 + Zika-emergency
U06), switching to ICD-10-CM (`simple-icd-10-cm`) would silently change *every* ancestor/leaf/NCA
result to the US clinical tree (a far larger behavioral blast radius than the hole it fixes), and
no offline lib offers CID-10 anyway.

Concrete remediation, in priority order:

1. **Fill `block` for WHO-absent CID-10 codes via `icd-mappings`.** In
   `icd_adapter.code_info`, when `status == source_system_specific`, resolve `block` from
   `icd-mappings` `target='block'` (verified: `A90/A91 → A90-A99`, `U06 → U00-U49`). This
   immediately restores the **0.5 same-block tier** in `DiseaseGraph._structural_weight` for the
   dengue/arbovirus cluster — the single highest-value fix, and it adds no dependency. Keep the
   `source_system_specific` status and the "never coerce to a WHO subcode" guard intact; only the
   block/chapter metadata is enriched.
2. **Add an explicit `who_absent_cid10` code table** (small YAML in
   `config/registries/.../disease/`) enumerating the known WHO-2019-absent-but-CID-10-present
   categories (A90, A91, U06, plus any others surfaced by a catalogue diff — see step 4) with
   their CID-10 block + pt-BR label. This makes the gap **declared and testable** rather than
   implicit, and gives dengue nodes a real block/label even offline. `_structural_weight` can then
   consult this table before falling back to chapter-only.
3. **Optional NCA/distance repair for dengue:** for codes in the table, synthesize a block-level
   pseudo-ancestor (`A90-A99`) so `nearest_common_ancestor`/`distance` return a finite result
   within the arbovirus cluster instead of `None`. Low priority; only matters if hierarchy
   *distance* (not just the graph weight) is used for dengue.
4. **Add a guard test** that diffs the WHO-2019 catalogue against the CID-10 category set actually
   present in DATASUS data and asserts every WHO-absent code is either in the `who_absent_cid10`
   table or explicitly waived — so a future arbovirus code (or a CID-10 vintage change) can't
   silently reintroduce a hole. Pin A90, A91, A92.0, A92.5/A92.8, A95, U06, Q02 as the named
   acceptance set (the arbovirus/microcephaly cluster this system exists to study).
5. **Housekeeping:** scrub any `target='cci'` reference in `concept_registry.py` /
   comments — `icd-mappings@0.6.2` exposes `ccir`, not `cci`; a bare `cci` target raises.

Net: one small enrichment in `icd_adapter.code_info` (block via the existing `icd-mappings` dep)
+ a declared gap table + a diff test closes the epidemiologically load-bearing hole without
changing the backbone, without new dependencies, and without violating the offline/deterministic
and "CID-10 is authoritative, never coerce to WHO" contracts.

---

## Sources

- [simple-icd-10 · PyPI](https://pypi.org/project/simple-icd-10/) ·
  [GitHub StefanoTrv/simple_icd_10](https://github.com/StefanoTrv/simple_icd_10) ·
  [docs](https://simpleicd10.stefanotravasci.it/)
- [simple-icd-10-cm · PyPI](https://pypi.org/project/simple-icd-10-cm/)
- [rmnldwg/icd (multi-release ICD-10/11)](https://github.com/rmnldwg/icd)
- [icd-codex](https://github.com/icd-codex/icd-codex) · [icdcodex docs](https://icd-codex.readthedocs.io/en/latest/readme.html)
- [WHO ICD API](https://icd.who.int/icdapi)
- [DATASUS CID-10 V2008 — A90-A99 block](http://www2.datasus.gov.br/cid10/V2008/WebHelp/a90_a99.htm) ·
  [Portal Afya CID-10 A90](https://portal.afya.com.br/codigos/I/grupo/A90-A99/cid/A90) ·
  [Portal Afya CID-10 A91](https://portal.afya.com.br/codigos/I/grupo/A90-A99/cid/A91)
