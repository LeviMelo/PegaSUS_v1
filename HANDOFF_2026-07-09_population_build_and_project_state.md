# Handoff — national population build memory wall + project state (2026-07-09)

**Audience.** A fresh, more capable coding agent taking over this project cold. I'm the outgoing
agent (Claude Opus 4.8). Treat everything below as *field notes and measured observations*, not
settled truth — I was wrong more than once today and I'll flag where. The repo already has a lot of
governing docs; this file is a **map + a debrief of an unfinished thread**, not another authority. When
my notes and the live code disagree, believe the code (that's a house rule here — see
`CLAUDE.md` §IX).

---

## 0. Orientation in 60 seconds

- **Project:** PegaSUS — a data-intensive Brazilian public-health epidemiology engine (DATASUS +
  IBGE/SIDRA). It builds a demographic **population tensor** (the person-time denominator), an EFG
  field-graph, and an LDO inference layer (sparse+low-rank graphical model, causal orientation,
  stability selection). Governing spec docs: `PEGASUS_MSD_III.md`, `PEGASUS_MSD_I.md`,
  `PEGASUS_OPERATIONAL_IMPLEMENTATION_PLAN.md`. Precedence/lineage index: `DOCS.md`. Cross-module
  status tracker: `PEGASUS_ISSUE_LEDGER.md`. Read `CLAUDE.md` first — it's the operating discipline
  and it's genuinely load-bearing here (empirical validation, "docs are hypotheses", profile before
  optimizing, report negative results).
- **The win condition** (per project memory): a national C25 (pancreatic cancer) mortality study,
  2000-2025, at full national+temporal scale. Engine exists (`studies/`), national data mostly
  fetched. The population tensor is the denominator it needs.
- **What's blocking right now:** materializing the **national** population tensor as a persisted,
  build-once asset. The build is correct but its **peak RAM (~18-20 GB) sits right at the edge of this
  machine's ~20.8 GB idle-available headroom**, so it aborts. Details in §1 — this is the live
  conundrum.
- **Environment:** Windows 11, `pegasus` conda env at `C:/Users/Galaxy/miniconda3/envs/pegasus/`
  (has torch 2.5.1+CUDA, polars, scipy, sklearn). GPU: RTX 4050 Laptop, 6.44 GB VRAM. Machine RAM:
  33.9 GB total, ~12-13 GB baseline usage, so ~20 GB realistically available for a build. Use
  `PYTHONIOENCODING=utf-8`; the population build path sets `MIMALLOC_PURGE_DELAY=0`. Run long jobs
  backgrounded, **not** concurrently with pytest (I learned this the hard way — see §1.3).

---

## 1. The immediate conundrum: national population-tensor materialization is memory-blocked

### 1.1 What I was doing
The population tensor is a *build-once, versioned, scope-invariant foundational asset* (see
`src/pegasus/assets/foundation.py::resolve_or_build_population_tensor` and the FAL-POP-VER work). The
goal was to run the **national** build once and persist it so studies slice it instead of rebuilding.
I invoked the build with `mode="sim_informed_denominator"`, `reconstruct_migration=True`, no race
bridge prior. Shape is `(5570 munis, 24 years, 101 single-year ages, 2 sexes, 5 races)` ≈ **135M
cells**.

### 1.2 The measured memory trace (this is the useful part)
I instrumented a national build with a per-0.3s RSS sampler + stage markers + a **safe 3 GB-available
watchdog** (so it aborts long before it can OOM the machine). Idle baseline available was 20.8 GB.
Trace (`scratchpad/national_isolated/memtrace.log`), RSS = process resident set:

```
  1.5s  rss= 4.1GB   read_strata exit            <- 18MB parquet -> 4.1GB frame (categorical strings, ~220x)
 81.1s  rss= 8.6GB   sim_death_priors (skipped)  <- records list[dict] + 2000-census built here
 81.6s  rss= 7.2GB   SOLVE_BLOCKED entry
 ...    rss~ 7-9GB   (68 locality-blocks, memory-graceful, ~490s)
573.4s  rss= 9.0GB   SOLVE_BLOCKED exit
605.3s  rss=17.4GB   WATCHDOG ABORT (avail<3GB), peak still climbing  <- THE EMIT STAGE
```

**Reading:** three cost centers, and I was wrong about which dominates until I measured (§IX again):
1. **`_read_population_strata` → 4.1 GB.** An 18 MB zstd parquet (very repetitive categorical strings)
   expands ~220× because it's read with plain Utf8, no dictionary/Categorical encoding.
   (`build/strata.py:47`).
2. **Record construction → ~8.6 GB.** The strata frame is iterated row-by-row into a Python
   **`list[dict]`** (~11.25M rows, then the 2000-census disaggregation ~doubles it), then collapsed to
   a columnar `records_df` and freed. The code comment itself estimates "~12M-row list[dict] (~6 GB at
   national scale)" (`build/orchestrator.py:330-389`). This coexists with the 4 GB strata frame until
   `del strata`.
3. **The EMIT → ≥17.4 GB (dominant, still rising at abort).** After the solve, the emit builds whole
   135M-cell float64 arrays `pop`, `mig`, `anch`, `simd` (~4.3 GB) and then a **nominally streamed**
   per-locality-block parquet write. The comment at `orchestrator.py:626-629` claims it "never
   materializes the full ~8 GB label frame" and is "byte-identical to a monolithic emit" — but the
   **measurement contradicts the comment**: RSS climbs ~8 GB during this stage. I did not finish
   isolating exactly where inside the emit loop the ~8 GB goes (candidates: object/string label arrays
   via `np.tile`, `frame.to_arrow()` transient doubling, ParquetWriter row-group buffering, or the
   per-block frames not being released). **This is the single highest-value thing to nail down** — the
   solve is already memory-graceful; the emit is not, despite claiming to be.

**Bottom line:** true peak ≈ 18-20 GB at the emit, machine has ~20 GB usable → the build is right at
the cliff. On a fully idle machine it *might* squeak through; under any concurrent load it aborts.

### 1.3 A confound I fell into (so you don't repeat it)
My first materialization attempt (3 tries: GPU, GPU, CPU fallback) aborted all three at my 2 GB RAM
floor. I initially concluded the build peaked ~20 GB everywhere. **Partly a confound:** I was running
the full pytest suite (which loads torch/polars across many processes and spawns subprocess-test
children) *concurrently* with the build during two of the attempts. That inflated the early aborts.
When I later ran the build *alone*, the record-construction stage only reached ~8.6 GB — but the
**emit stage still genuinely spikes to ~17.4 GB+** even idle. So: the confound explained the *early*
aborts; the emit spike is *real*. Lesson (now in project memory): never run pytest concurrently with a
national build on this box.

### 1.4 Things about this build I am NOT sure are correct (please verify against the spec)
- **`sim_informed_denominator` + no race bridge prior.** With a real 5-race axis and no bridge prior,
  `_sim_death_priors` *short-circuits and returns None* before reading the SIM file
  (`build/priors.py:194`) — so the "SIM-informed" death prior is effectively absent, and race
  structure comes only from the census composition prior. I don't know whether the intended national
  invocation should (a) pass a RaceBridge prior, (b) use `independent_denominator` mode, or (c) accept
  this degradation. `loss.py:70-72` ties the mode to `lambda_D`. **Verify what the correct national
  denominator config actually is (MSD §2.8).** I may have run it with the wrong knobs.
- **`reconstruct_migration=True` at national scale refuses.** The O→D flow reconstruction exceeds
  `MAX_DENSE_FLOW_PAIRS=8000` and now *surfaces a skip* (my §V fix, commit `9517ab0`) rather than
  silently no-op'ing — the migration-affinity spatial kernel is absent nationally and downstream falls
  back to structural contiguity. That's honest but it means the emitted `migration` column is the
  per-locality net residual, not O→D flows. Enabling national flows needs a sparse backend (ledger
  POP-FLOW-02).

---

## 2. What I did today (all committed on branch `TDD-branch-redo`)

The day started as a "adopt CUDA/GPU more aggressively" mandate and evolved into the memory wall
above. Commits, newest last:

| Commit | What | Confidence |
|---|---|---|
| `ee6df3b` | POP-02 GPU: torch SPG solver for the per-block population solve (f32 bulk, f64 reductions) | High — validated: autograd matches numpy analytic to 1e-16 (f64)/1e-7 (f32); recovery max 1.6e-7 rel-to-scale; ~16× the per-block solve |
| `ca1fd33` | POP-02 GPU: hardened (finiteness guards, cuda.synchronize, per-block empty_cache) | High |
| `d714d7b` | POP-02 GPU: gate GPU to ≤12 locality-blocks (~single UF); national → crash-free CPU | Medium — see caveat below |
| `9517ab0` | Population flows: surface the migration-flow refusal (stop a §V silent no-op) | High — 4 return paths validated |
| `1293e0d` | Ledger update | — |
| `f6be90d` | POP-02: subprocess-isolated national build (retry + hard-timeout + tree-kill + RAM floor) | High mechanism, see caveat |

### 2.1 The GPU population solver (POP-02) — the one clean CUDA win
`src/pegasus/denominators/reconstruction/torch_solver.py`. Ports the 8-term population loss + SPG
(Barzilai-Borwein + GLL line search + Duchi simplex projection) to torch. Machine-identical to the
numpy path; ~16× on the per-block solve. Dispatched from
`projected_gradient.py::solve_projected_gradient_small` when a block clears 200k cells. This works and
is validated. **But** the national many-block loop intermittently *hard-segfaults* (exit 139) from a
native torch+polars(rayon) transition race — the tensor is correct when it completes (I verified
170M/191M/203M national totals in an earlier session) but it crashes ~2 of 3 national runs.

### 2.2 The crash gate (`d714d7b`) and the subprocess isolation (`f6be90d`)
Two layers of response to that segfault:
- **Gate** (`d714d7b`): route national (68 blocks) to the crash-free CPU solve; keep GPU for
  study/state scale (≤12 blocks, tested crash-free). A `prefer_gpu` flag threaded
  orchestrator→solvers→projected_gradient. Monotone-safe.
- **Isolation** (`f6be90d`): `src/pegasus/denominators/population/build/isolated.py` runs the whole
  build in a **child process** with (a) retry past the transient segfault, (b) a hard per-attempt
  timeout that kills the whole **process tree** (`taskkill /F /T` — a shell `timeout` can't kill a
  child tree on Windows), (c) a RAM-floor abort, (d) attempt plan = GPU×2 then a guaranteed CPU
  fallback with CUDA hidden. `run_population_build_isolated` re-enables national GPU via
  `allow_national_gpu` because the crash is recoverable under isolation. `compile.py:352` routes
  `NATIONAL_FULL_HISTORY` builds through it. The mechanism is **validated** (5 focused tests in
  `tests/unit/test_population_isolated_runner.py`: crash→CPU-fallback recovery, immediate success,
  hang→tree-kill with no orphans, policy guard). **This is the "hard terminator" the user asked for
  and it works.** The caveat: it converts a national **memory** failure into a clean, safe *failure*
  (all attempts abort at the RAM floor) — it does not *fix* the memory. The RAM-floor default (2 GB)
  is arguably too aggressive given the build legitimately needs ~18 GB; the real fix is reducing the
  emit peak (§1.2 item 3).

### 2.3 The LDO eigh GPU port — a measured NEGATIVE result (please don't redo it blindly)
The internal survey + the issue ledger's task list claimed "batched ADMM eigh GPU, 5-15×" was the #1
LDO GPU win (and one tracker entry even marked it *done* — it is **not**; live code is numpy
`np.linalg.eigh`). I profiled it before building (`scratchpad/profile_ldo_eigh.py`,
`validate_gpu_eigh_ops.py`). Findings:
- A single CUDA f64 eigh at q=360 is 6.9× a numpy 1-thread eigh (17.9 ms vs 122 ms), and GPU/CPU
  agree to 2.6e-15 with identical recovered edges. So the *op* ports cleanly.
- **But** eigh is only ~60% of a fit (pairwise_correlation on the 122k-col feature matrix + whitening
  are the rest) → a single fit is only **1.76×**; the dominant LDO cost (`stability_select`, 20
  refits, 244 s) **already parallelizes ~4.5× across CPU cores**, which a single GPU can't beat for
  independent refits; and **batching is counterproductive** at q=360 (GPU already compute-bound).
  Realistic net ~1.7-2× with regression risk. I chose **not** to build it (`CLAUDE.md` §III: match
  effort to measured value; §VII: report negative results). Recorded in the ledger under GPU-01..08.
- **The broader reframe** (my honest read, treat as a hypothesis you can overturn): PegaSUS's heavy
  numerical ops (eigh, GEMM) already individually saturate a single small GPU at national scale, so
  the "batch many small ops" pattern that makes GPUs shine mostly doesn't apply here. The population
  *iterative solve* was the exception (one big dense iterative kernel) and it's done. I did **not**
  find a large untapped CUDA reservoir. HSIC permutation null (`ldo/hsic.py:301` already has a GPU
  path; :324 loop unbatched) is the same compute-bound story — I judged it marginal. You may see
  further than I did here.

---

## 3. Remaining goals

### 3.1 Explicit / active (this is where the user's attention is)
1. **Materialize the national population tensor** (the immediate blocker, §1). Needs the emit-stage
   memory reduced so the ~135M-cell build fits comfortably (target: peak well under ~12 GB so it's
   safe on a loaded machine, which is the user's stated standard — "fit on a loaded machine and handle
   memory gracefully"). Then run it once (idle machine, no concurrent pytest) and persist via
   `resolve_or_build_population_tensor` so the pipeline reuses it by content-identity.
2. **Run the full contextual compile + investigate** at national+temporal scale (tasks #31/#32):
   compile the panel over ALL DATASUS systems + ALL SIDRA context tables, then the LDO investigate
   stage (determinants, disease-axis comorbidity). The user has a HARD "no cherry-picking" directive:
   every run uses all systems at full national+temporal scale (project memory: `full-data-no-cherrypicking`).
3. **The pancreatic C25 national study** — the win condition. `studies/pancreatic_c25_national/`
   engine exists and was validated on real AL 2022. Needs the national denominator (item 1) + the
   national compile (item 2). Known past blockers were Windows fetch-persistence and >10 min compile,
   not registration/memory (per project memory `pancreatic-c25-study-delivered`) — re-verify, those
   notes may be stale.

### 3.2 Memory / performance work implied by §1 (a coherent §VIII workstream)
- **Emit stage** (`orchestrator.py:625-700`): the dominant ~8 GB spike. Confirm the cause (object
  label arrays / to_arrow / writer buffering) and make it genuinely streaming/O(block). The comment
  claims it already is; the measurement says otherwise. Highest leverage.
- **`_read_population_strata`** (`strata.py:47`): 18 MB → 4.1 GB. Read with `Categorical` dtype and/or
  column projection; likely a clean ~3 GB win, low risk. I had this queued as the next step when I was
  interrupted.
- **Record construction** (`orchestrator.py:330-389`): the `list[dict]` (~6-12 GB transient). Vectorize
  `_canonical_stratum` (`strata.py:32`) into polars expressions (the SIM path already does this in
  `priors.py::_stratify_age_column`), and reframe the 2000-census disaggregation + AMC carve to operate
  on a columnar frame instead of a list. Bigger refactor; must stay byte-identical (there are
  block-invariance + AL-parity validations you can lean on — see project memory
  `population-solver-national-bottleneck`).
- **Caveat from me:** POP-MEM (a prior optimization pass) reduced the *non-SIM* national build to
  "~5-6 GB" but **validated on AL (102 munis), ~55× smaller than national** — so its peak claims never
  exercised the national emit. Distrust the "5-6 GB" figure at national scale; it's a §IX
  code-ahead-of-its-record situation in reverse (the record over-claims).

### 3.3 Project-level / ledger (mostly settled, a few genuinely open)
`PEGASUS_ISSUE_LEDGER.md` is the tracker and is largely honest, but per its own preamble the codebase
runs *ahead* of its docs — verify any "OPEN" before building. Genuinely-open highlights I'm aware of:
- **POP-FLOW-02**: national O→D migration flows need a sparse backend (dense refuses >8000 pairs). The
  silent no-op is now surfaced (`9517ab0`); enabling the feature is open.
- **RaceBridge ecological reformalization** (project memory `racebridge-ecological-reformalization`):
  the admin↔self-declared race distortion is *ecological* (aggregate), reformalized as hierarchical
  Poisson ecological deconvolution; the estimator is built + validated (`measurement/race_ecological.py`),
  the region-conditioned confusion matrix `C` is data-blocked (needs PNS/PNAD linkage). Current
  national `C` is honestly flagged as uncalibrated.
- **FEAT-P3e**: a CLI wrapper is the only remainder of the Output Query Layer; the ledger argues it
  should be built to the *study's* real consumption needs, not speculatively.
- A long tail of DEFERRED items (causal installment P5, Zika acceptance test, DIS-06/07, LDO
  cross-fit) — read the ledger's status column, don't inherit my framing.

---

## 4. How to navigate this repo (pointers, not gospel)

- **`CLAUDE.md`** — the operating discipline. Not boilerplate here; it encodes hard-won lessons
  (empirical validation over test-suite-green, profile the real workload, isolate one numeric change,
  report negative results, verify current state before trusting any status claim including your own).
- **Persistent memory** at `C:/Users/Galaxy/.claude/projects/C--Users-Galaxy-LEVI-PegaSUS/memory/`
  with an index `MEMORY.md`. One fact per file. The most relevant to this handoff:
  `population-solver-national-bottleneck` (national build memory, now partly superseded by §1 here —
  the emit spike wasn't in it), `gpu-cuda-adoption` (the measured GPU reality + the corrected float32
  premise: f32 fine for squared-residual objectives, **f64 required for eigh/logdet**),
  `full-data-no-cherrypicking`, `background-jobs-persist`, `prefer-architecture-over-tests` (the user
  wants architectural milestones + one focused proof-of-capability test per feature, not test
  batteries), `commit-message-style`, `git-hygiene-codecontext` (stage explicit paths, never
  `git add -A`; `data/` is gitignored — I added `data/denominators/**`).
- **Test harness**: `tests/{unit,contract,synthetic,acceptance}`. 389 unit+contract green as of the
  last commit. Run: `python -m pytest tests/unit tests/contract -q`. Population/reconstruction tests
  are fast; the LDO synthetic-recovery tests are the slow ones.
- **The population build call chain** (for §1):
  `workflows/compile.py:352 _build_fn` → (national) `build/isolated.py::run_population_build_isolated`
  → child → `build/orchestrator.py::solve_population_tensor_from_sidra_strata` → record construction
  (330-389) → `_solve_locality_blocked` (142+, the memory-graceful solve) → emit (625-700).
- **Scratchpad** (throwaway probes, safe to ignore/delete):
  `C:/Users/Galaxy/AppData/Local/Temp/claude/.../scratchpad/` — `diagnose_national_mem.py` (the RSS
  tracer that produced §1.2), `materialize_national.py`, `profile_ldo_eigh.py`,
  `validate_gpu_eigh_ops.py`, `validate_isolated_runner.py`.

---

## 5. If I were continuing (suggestions, fully overridable)

You'll likely have a better plan than mine. But concretely, the path I'd have taken next:
1. **Nail the emit spike** with a finer probe (RSS around each line of `orchestrator.py:625-700`, and
   check whether `emit_block` frames + `to_arrow()` accumulate). It's ~8 GB and the code claims it
   shouldn't exist — a good bet for a clean, high-leverage fix.
2. **Do the low-risk `_read_population_strata` Categorical/projection fix** (~3 GB) regardless.
3. Re-measure peak; if under ~10 GB, **materialize the national tensor** on an idle machine (no
   concurrent pytest) and persist it. Verify census-year totals ≈ 169.9M / 190.8M / 203.1M for
   2000/2010/2022 and 135,041,040 rows.
4. **Resolve the `sim_informed` + no-bridge question** (§1.4) against MSD §2.8 before trusting the
   tensor's race/death structure — I'm genuinely unsure I ran it with the right knobs.
5. Then the national compile+investigate and the C25 study.
6. Revisit whether the isolation RAM-floor default (2 GB) should be lower once the emit is fixed, and
   whether keeping `compile.py` routing national through the isolated runner is right (it's safer than
   the in-process path either way — it turns a potential OOM-thrash into a clean failure — so I left it
   wired).

One meta-note: I spent a lot of today oscillating on decisions the measurements could have settled
faster. The five-minute probe really does overturn plausible narratives here (it overturned mine about
where the memory went, twice). Lead with the probe.

— outgoing agent, 2026-07-09
