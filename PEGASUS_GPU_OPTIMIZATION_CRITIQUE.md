# PegaSUS Optimization + GPU-Acceleration Rematch

Objective 3: speed/memory/compute redesigns, and — critically — a FRESH assessment of GPU/CUDA/PyTorch acceleration for the heavy math (EFG, and especially the LDO), given the mathematical developments since the last (pessimistic) GPU assessment. Environment: torch 2.5.1, CUDA on RTX 4050 Laptop (6.4 GB VRAM), `compute.yaml dtype=float32, max_vram_fraction=0.80`. GPU currently wired ONLY for HSIC (pirs_hsic_*), race-bridge, STDFM — **nothing in the LDO estimator core**.

**Why the rematch is justified (the reframing):** the earlier assessment judged *single* operations — one `p≈130` eigh, one `S` solve — which are indeed trivial and CPU-fast. But the LDO's actual workload is now dominated by MANY REPEATED small-dense operations, and that count has GROWN with every math development since: stability selection (`n_subsamples` refits × ~500 ADMM iters), the O4 λ-grid path refits, the O3 exact-vs-approx slice refits, the O5 temporal-holdout refits, the multiresolution coarse+fine refits, and the O(p²) pairwise HSIC scan. GPUs win on exactly this — batched identical small kernels — not on one big op. The workload shape changed; the conclusion should be re-derived.

**Full granularity — nothing compacted.** Lead's direct assessment first; the `gpu-optimization` workflow appends full findings.

---

## Part A — Lead's direct GPU-rematch assessment

### G1 · `GPU-STAB-BATCH` — HIGH — batch the stability-selection ADMM refits on GPU (the ideal batch)
- **detail:** `stability_select` runs `n_subsamples` (12–20) INDEPENDENT CPW-ADMM fits, each ~500 iterations whose hot step is a `p·(K+1)` (≈130) symmetric eigh (`_prox_neg_logdet`, `_psd_project_shifted`). This is the textbook GPU batch: many identical small dense eigendecompositions. `torch.linalg.eigh` is batched — run all `n_subsamples` subproblems' eigh as one `(n_sub, p, p)` batched call per ADMM iteration. A batch of 20× 130×130 eigh is negligible VRAM (~20·130²·4 B ≈ 1.4 MB) and saturates the GPU. This parallelizes the LDO's dominant wall-clock without any accuracy change (eigh is exact).
- **value:** HIGH — the single biggest LDO speedup; also composes with A15 (redundant per-subsample whitening) once the whitening is precomputed once.
- **plan:** a `torch`-backed batched ADMM in a new `ldo/torch_admm.py` behind `resolve_torch_device("ldo_admm", prefer_cuda=True)` with a numpy fallback; the S/L/U/R iterates as `(n_sub,p,p)` GPU tensors; soft-threshold + batched eigh + PSD-clip all batched. Determinism via `seed_everything`. Validate vs the numpy ADMM within float32 tolerance.

### G2 · `GPU-HSIC-SCAN` — HIGH — the residual HSIC scan is CPU-only despite hsic.py having GPU paths; it is the largest LDO allocation
- **detail:** `residual_scan.py` reimplements the HSIC pairwise scan importing only the NUMPY private helpers (`_bandwidth`, `_np_nystrom_features`, `_np_rff_features`) — it never touches the GPU, even though `hsic.py` has CUDA-mode paths (now orphaned per O9). The scan builds a per-variable representation cache (dense `n_eff×n_eff` kernels or `n_eff×features` maps) and scores every O(p²) pair under a shared permutation set — the single largest national LDO allocation (~48 GB at exact `n_eff=5000`, per `estimate_residual_scan_bytes`). Kernel Gram matrices, RBF evaluations, feature maps, and permuted-statistic computation are all dense GEMM/elementwise — ideal for the GPU, and float32 halves the memory to fit VRAM in tiles.
- **value:** HIGH — both a large speedup (batched kernels/permutations) AND the memory relief that lets the scan run at national scale (tile the pairs across VRAM instead of a 48 GB CPU allocation).
- **plan:** route the live residual scan through a GPU HSIC (unifying with hsic.py per O9 — a modularization + GPU win together): feature maps + centered-kernel statistics + the permutation null as batched torch ops on tiles that fit `0.8·6.4 GB`; CPU fallback. Resolves O9 and the residual-scan memory ceiling simultaneously.

### G3 · `GPU-KRON` — MEDIUM — the Kronecker joint operator is native torch.einsum territory (once load-bearing)
- **detail:** `kron.py`'s 3-factor vec-trick matvec, factored solve, and CG are mode-wise tensor contractions — `torch.einsum` on GPU is the natural backend, and the national space factor (sparse S≈5570) + dense var/time factors would benefit. But currently `joint_logdet` is telemetry-only (not load-bearing), so the payoff waits on the operator driving something real (a joint marginal-likelihood, or a Kronecker-preconditioned solve used in the fit). Pairs with the math-ledger question of whether the separable operator should become the actual inference substrate.
- **value:** MEDIUM (HIGH if the joint operator becomes load-bearing).
- **plan:** a torch backend for `KroneckerPrecision.matvec/solve` (drop-in), enabled when the operator feeds a real computation; keep the numpy path as reference.

### G4 · `GPU-WHITEN` — MEDIUM — spatial-whitening sparse matvecs at national S
- **detail:** `whitened_lagged_correlation` accumulates `Ft @ (Q_sparse @ Ft.T)` over T time slices — sparse `(S×S)·(S×F)` matvecs at S≈5570. `torch.sparse` matmul on GPU could accelerate, but sparse-GPU overhead is real and the op is memory-bound; the win is uncertain and needs measurement. Precomputing the whitened features ONCE (A15) is the bigger lever and a prerequisite (don't accelerate a redundant computation).
- **value:** MEDIUM, measure first.
- **plan:** after A15 (compute whitening once), benchmark a torch.sparse whitening vs scipy; adopt only if it beats CPU including transfer.

### G5 · `GPU-POP-BLOCKED` — MEDIUM — re-open the population-tensor GPU path as batched locality blocks (honestly deferred, but the math now fits)
- **detail:** The population denominator solve was made CPU locality-blocked (POP-02 M3) and the GPU flag was removed (honestly, no wired path). But blocked solve = many small IDENTICAL per-block problems = a GPU batch, and the projected-gradient step (residual + non-negative projection) is exactly what `she/reconstruction/torch_kernels.py` already implements (orphaned). With M2 (float32 storage) landed and the blocked structure in place, a batched-block GPU solve is now a clean fit within 6 GB (each block ≪ VRAM).
- **value:** MEDIUM — throughput on the national denominator build.
- **plan:** wire `torch_kernels` behind the block loop with a VRAM preflight + CPU fallback (the G1 pattern from the compute plan); device telemetry so "was the GPU used" is answerable.

### G6 · `GPU-VRAM-SERIALIZE` — MEDIUM (prerequisite) — VRAM preflight has no reservation; concurrent CUDA admits can co-OOM 6 GB
- **detail:** `devices.py:resolve_torch_device` reads instantaneous free VRAM with no lock/reservation. With G1/G2/G5 introducing more CUDA tasks (some concurrent with the already-GPU HSIC/STDFM), two tasks can each pass preflight then collectively exceed 6 GB. A prerequisite for safely turning on more GPU paths.
- **value:** MEDIUM (enabler).
- **plan:** a process-wide CUDA admission semaphore / VRAM budget; re-check under lock; queue rather than co-admit.

### G-THEME — the reframed conclusion
The earlier "GPU gains are minor" was correct for the workload AS IT WAS (few, small ops). The current LDO is dominated by **repeated small dense ops** (stability/λ-path/holdout/exact-vs-approx refits) and **one huge dense scan** (HSIC residual, the 48 GB allocation). Those are precisely GPU-favorable, and float32 (already the declared dtype) makes them fit 6 GB in tiles. **G1 (batched stability ADMM) and G2 (GPU HSIC scan) are the two high-value, math-validity-neutral wins; both should be built.** They also compose with modularization (G2 unifies the two HSIC implementations, O9) and with the redundancy fixes (A15).

*(Direct assessment continues; the `gpu-optimization` workflow's full findings — per-op profiling, EFG kernels, streaming, numerical-method swaps — append to Part B.)*

## Part B — `gpu-optimization` workflow findings (full, verbatim)
*(pending)*
