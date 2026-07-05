# PegaSUS Compute & Build Layer — Analysis & Optimization Plan

**Status:** analysis + plan (investigation done; no code changed yet)
**Trigger:** a ~16 GB Python process during the national population-tensor solve; GPU apparently idle throughout; suspicion of memory inefficiency and an unclear question of whether a nationwide tensor was even produced.

**Three findings, three verdicts:**
1. **Did the national tensor build?** *Yes* — the 2-year national run produced a real all-municipality demographic tensor (`population_tensor_solver_independent_denominator_{total,age_group,sex,race}` over 5,570 munis). It concluded. But it needed ~16 GB *for 2 years*; the full 2000–2024 window would **OOM**.
2. **Is the OOM justified?** *No.* The dominant cost is the problem being held as ~10 `n_cells`-length **Python float tuples** (~32 B/element), then **copied again to numpy** at solve time. Converting to numpy (float32) is a ~8× reduction and unblocks full-national.
3. **Is the GPU used?** *No — never, for the population tensor.* The GPU kernels exist (`she/reconstruction/torch_kernels.py`) but have **zero callers**; all four solver backends are numpy/scipy CPU. (HSIC and STDFM *do* use the GPU.)

---

## 1. Population-tensor memory — the 16 GB process

### Anatomy
`build.solve_population_tensor_from_sidra_strata` constructs a `PopulationTensorProblem` with `shape = (localities, periods, ages, sexes, races)` and `n_cells = ∏shape`, then solves it.

- 2-year national: `5,570 × 2 × ~100 × 3 × 5 ≈ 16.7M cells`
- full 2000–2024 national: `5,570 × 25 × ~100 × 3 × 5 ≈ 209M cells`

### Root cause — storage representation (`she/reconstruction/schema.py`)
`PopulationTensorProblem` holds **~10 arrays as Python tuples**: `anchors`, `hard_anchor_mask`, `births`, `death_rates`, `sim_deaths`, `race_composition_prior`, `closure_totals`, `migration_bounds`, `initial_population`, `initial_migration`. A Python `float` in a tuple costs ~32 B (8 B pointer + 24 B boxed float) vs **8 B numpy float64 / 4 B float32**.

| representation | per element | 16.7M cells × 10 arrays | 209M cells × 10 arrays |
|---|---:|---:|---:|
| Python float tuples (current) | ~32 B | ~5.3 GB | **~67 GB → OOM** |
| numpy float64 | 8 B | ~1.3 GB | ~16.7 GB |
| numpy float32 | 4 B | ~0.67 GB | **~8.4 GB** |

**And it is double-counted:** `projected_gradient.py` (and the sparse backends) call `np.array(problem.anchors, …)` etc. at solve time, so the tuple *and* its numpy copy are both resident. Peak ≈ tuple set + numpy set + solver intermediates (gradients, `projected`, `grouped.reshape`, …). That is the observed 16 GB at 2 years, and ~50–70 GB at full national.

### Second offender — `migration.hop_distances` (all-pairs BFS)
`reconstruct_migration_flows` calls `hop_distances(adjacency, nodes)` over **all** nodes, which BFS-records `(source, target) → dist` for every reachable pair. On a connected national contiguity graph that is ~`5,570² ≈ 31M` entries in a **Python dict** (~150 B/entry ⇒ ~5 GB) — built *before* the `MAX_DENSE_FLOW_PAIRS = 8000` guard can reject. So even though national migration reconstruction ultimately refuses (pairs ≫ 8000), it first pays the all-pairs-BFS memory.

### Solver policy (`registries/population.py`) — not the bottleneck, but relevant
- `projected_gradient_small` (dense analytic): `max_cells = DENSE_NATIONAL_CELL_THRESHOLD = 10M`.
- `sparse_block_coordinate`: `max_cells = 250M`.
So national (16.7M / 209M) already routes to the **sparse** backend and does *not* abort on cell count — the ceiling is problem construction, not the solver.

### Fixes (tiered)
- **M1 — numpy-array storage (biggest unlock, moderate refactor).** Replace the tuple fields in `PopulationTensorProblem` with numpy arrays (or accept both and normalize once in `__post_init__`). Solve reads them directly — no re-copy. ~4× (float64) to ~8× (float32) smaller, no double storage. Turns 209M-cell national from ~67 GB → ~16 GB (f64) / ~8 GB (f32).
- **M2 — float32 for the denominator solve.** The population denominator does not need float64 precision; `config/compute.yaml` already declares `cuda.dtype: float32`. Halves M1 again.
- **M3 — locality-blocked solve (the real scaling fix).** The tensor is **separable across localities** given each locality's own closure totals (the only cross-locality coupling is migration, which is already a bounded, optional term). Solve in muni-blocks (e.g. 200–500 munis/block), streaming block results to the tensor parquet. Peak memory becomes O(block), independent of national scale — and each block is a natural GPU batch (see §2).
- **M4 — bounded-radius hop distances.** Replace all-pairs `hop_distances` with a per-source BFS truncated at `max_hops` (the only distances the gravity prior uses). Memory drops from O(N²) to O(N · avg-neighbours-within-max_hops). Also fixes the pre-guard blowup.

## 2. GPU / CUDA — available, wired elsewhere, orphaned here

**Environment:** torch 2.5.1, `cuda.is_available() = True`, **RTX 4050 Laptop GPU (6 GB VRAM)**.

**Where GPU IS used:** `pirs/hsic.py` and `she/stdfm/torch_solver.py` call `resolve_torch_device(..., prefer_cuda=True)` and run on CUDA when the task is in `config/compute.yaml → cuda.enabled_for`.

**Where it is NOT (but config says it should be):** `config/compute.yaml` lists `population_tensor_kernels` under `enabled_for`, and `she/reconstruction/torch_kernels.py` implements GPU kernels (`population_tensor_runtime`, `project_nonnegative`, `squared_residual_sum`) — but **nothing calls it**. `solve_population_tensor_problem` dispatches only to numpy/scipy backends (`projected_gradient`, `sparse_admm`, `sparse_block_coordinate`, `state_space`). The GPU path is dead code; the config flag is aspirational.

**Should the population tensor use the GPU?**
- **Not as a dense national tensor.** 209M cells × 4 B × several working arrays ≈ 8 GB > 6 GB VRAM — it would not fit. So "just move the solve to CUDA" is not viable at national scale on this GPU.
- **Yes, as blocked batches (M3).** Per-muni-block problems (200–500 munis × periods × strata) are small, independent, and identically shaped — the ideal GPU batch. Wiring `torch_kernels` behind the block loop, with the existing `resolve_torch_device` VRAM preflight and a CPU fallback, is the correct way to actually use the RTX 4050. The projected-gradient step (residual + non-negative projection) is exactly what `torch_kernels` already implements.

### Fixes
- **G1 — wire `torch_kernels` into the block solve (after M1/M3).** Add a `torch`-backed backend option selected via `resolve_torch_device("population_tensor_kernels", prefer_cuda=True)`; fall back to numpy when CUDA is absent or the block exceeds the VRAM preflight. Keeps determinism via `seed_everything`.
- **G2 — honest telemetry.** Record `device` (cuda/cpu) and VRAM preflight in the solver telemetry so "was the GPU used" is answerable from the run bundle (today it is silent).
- **G3 — either use or drop the flag.** Until G1 lands, `population_tensor_kernels` in `enabled_for` is misleading — annotate it as pending so the config does not imply a capability that is not wired.

## 3. Tier-3 — layer re-materialization (fetch → build)

The consumed DATASUS/SIDRA data is copied through several full materializations per run:

```
processed.parquet (per chunk)
   → data/processed/datasus_combined/<sys>/uf=X/years=Y/…/processed.parquet   (per-UF concat)
   → data/normalized/datasus/<sys>/uf=X/years=Y/…/canonical.parquet           (per-UF normalize)
   → data/normalized/national/<sys>__processed_events.parquet                 (national concat)
```

Each arrow is a **full copy**. For national SIM that is ~200 MB × 3–4 layers; for national SIH it would be far larger. The combine/normalize layers exist only to hand one file per system to the substrate.

### Fixes
- **T1 — normalize is already streamable; make combine a lazy scan.** `_combine_processed_datasus_chunks` eagerly `pl.read_parquet`s every chunk then concats. Replace with `pl.scan_parquet(glob)` → normalize as a LazyFrame → `sink_parquet` once. Removes the `datasus_combined` intermediate entirely (normalize consumes the chunk glob directly).
- **T2 — Hive-partitioned canonical, national as a view.** Write `canonical.parquet` into a Hive-partitioned dataset (`normalized/datasus/<sys>/uf=…/year=…`), and let the national "combine" be a `scan_parquet(hive)` the substrate reads lazily — materialize a single national file only if a consumer truly needs one, via `sink_parquet`. Removes the `normalized/national` copy.
- **T3 — same for SIDRA facts** (chunk → workdir-combined → national): collapse to one partitioned facts dataset + lazy scan.

## 4. Sequenced work items

| Tier | Item | Effect | Risk |
|---|---|---|---|
| **1** | **M1** numpy-array `PopulationTensorProblem` + solve reads directly | 209M-cell national ~67 GB → ~16 GB; removes double-copy | med — touches schema + 4 backends; guarded by existing solver tests |
| **1** | **M4** bounded-radius `hop_distances` | O(N²) → O(N·k) dict; fixes pre-guard blowup | low |
| **2** | **M2** float32 denominator solve | halves M1 | low-med — assert rate parity vs f64 on a state fixture |
| **2** | **M3** locality-blocked solve (stream blocks) | peak memory O(block), scale-independent; enables GPU batching | med-high — block boundary must respect migration coupling (bounded/optional term) |
| **2** | **T1** lazy combine (drop `datasus_combined`) | −1 copy/run | low |
| **3** | **G1/G2** wire `torch_kernels` behind block solve + device telemetry | actually use the RTX 4050; answerable device provenance | med — VRAM preflight + CPU fallback + determinism |
| **3** | **T2/T3** Hive-partitioned canonical/facts + national-as-view | −1–2 copies/run, bounded steady state | med |
| **cleanup** | **G3** annotate/park the aspirational `enabled_for` flag | config honesty | trivial |

## 5. Verification
- **Memory:** peak-RSS probe on a state run (small) and a synthetic 209M-cell problem *construction* (assert it stays < a few GB with M1, and OOMs today) — no full solve needed to prove the storage win.
- **Correctness invariant:** M1/M2/M3 must not move the numbers. Re-run the national 2-year pop-tensor and assert the `independent_denominator_{total,sex,age,race}` fields match the current run within tolerance (float32: loose tol).
- **GPU:** after G1, assert solver telemetry records `device="cuda"` on this machine and identical results (within f32 tol) to the CPU path; assert graceful CPU fallback when `CUDA_VISIBLE_DEVICES=""`.
- **Tier-3:** assert the national compile still validates `OK` and produces the same field set with the intermediates removed.

## 6. Bottom line
The national demographic tensor **works today but does not scale** — not because the math is heavy, but because the problem is stored as Python tuples and copied twice, and the GPU that could batch it is unwired. **M1 (numpy) + M4 (bounded BFS)** are the high-ROI, low-risk unlocks that make full-national feasible on CPU; **M3 (blocking) + G1 (GPU)** are the scaling+throughput follow-ons; **T1–T3** bound steady-state disk as in the storage plan. None of it changes a produced number beyond float32 tolerance.
