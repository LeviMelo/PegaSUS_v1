# Working principles — data-intensive, mathematically-heavy projects

Operating discipline for projects built on serious data flows and mathematical machinery
(estimators, inference engines, spatial/temporal models, large pipelines). These are general and
domain-agnostic; the parenthetical examples are illustrative, not specific requirements. The single
governing idea: **claims — in specs, docs, prior code, or your own reasoning — are hypotheses until
measured. Earn every conclusion.**

---

## I. Specifications and documentation are fallible hypotheses

- **Empirically validate prescriptions before implementing them — especially mathematics and
  performance.** A document (however authoritative) states an intention; the code must state a
  measured result. Build the smallest probe that measures the actual quantity the prescription
  claims to improve, and let the number decide.
- **A correct prescription can still fail in three distinct ways. Name which one you face:**
  (a) *naive implementation* — right idea, wrong constants/algorithm, slower or unstable than the
  thing it replaces; (b) *correct-but-marginal in the operating regime* — the effect exists but is
  negligible at the scale/data you actually run; (c) *correct-but-wrong-layer* — the effect is real
  but belongs to a different component of the model than the one the prescription touches. Each has
  a different fix; conflating them wastes effort.
- **Respect source provenance, but hold even the top authority as fallible.** Where multiple sources
  conflict, prefer the newest and most rigorously reviewed, and never cite a superseded or
  weaker source to settle a question. Then still test it — the highest authority is a strong prior,
  not a proof.

## II. Validate mathematics by measurement and inspection — not by test suites

- **A green test suite does not establish mathematical correctness.** Synthetic tests routinely pass
  while the underlying math is wrong (an inverted sign, a wrong degrees-of-freedom, a leak). For
  correctness, read the math directly and reason about it; for behavior, measure the real quantity.
- **Prefer throwaway probes over committed test batteries for validating math.** A short, disposable
  script that computes the actual estimand (a design effect, a residual, a recovery rate) and prints
  it is decisive and cheap. Do not pad commits with large synthetic test scripts — they add
  maintenance surface and false confidence. Keep one focused proof-of-capability test per feature,
  not a battery.
- **Isolate one numeric change at a time.** Sensitive recovery/identification behavior can hinge on a
  single constant. When a change breaks a recovery check, *diagnose why* (what went to zero, and by
  what mechanism) before reverting — the break is information.

## III. Diagnose the mathematical structure before engineering a fix

- **Locate where a phenomenon lives in the model, then fix it there.** Decompositions assign roles:
  a *sparse* component holds local/direct structure, a *low-rank* component holds dense/global
  structure; a whitening/precision step removes *local* nuisance, a factor/covariate step removes
  *global* confounding. A phenomenon with a given mathematical structure (e.g. low-rank, long-range)
  can only be addressed by the component that owns that structure — no amount of elaboration on the
  wrong component will move it. Establish the structure (rank, range, locality, stationarity)
  empirically first.
- **Match effort to empirically-established value.** When a probe shows a candidate improvement is
  marginal, stop polishing it and redirect. Elaborateness should track measured payoff, not
  aspiration or doc prescription.
- **Distinguish "the safe default is adequate" from "the default is a lazy shortcut."** Sometimes the
  simple baseline is genuinely near-optimal (and you can prove it with a benchmark); keeping it is
  then a justified conclusion, not negligence. Document the evidence either way.

## IV. Statistical honesty under dependence

- **Deflate the effective sample size for dependence.** Serial correlation, spatial autocorrelation,
  and overlapping windows all make N observations worth fewer than N independent ones. Propagate a
  design-effect-corrected `n_eff` into every standard error, power gate, and significance threshold,
  using the correct formula for the actual estimand (the design effect for a correlation differs
  from that for a mean).
- **Use dependence-robust multiplicity control when the tests are correlated.** Across a large,
  correlated hypothesis space, independence-assuming FDR can be anti-conservative; a
  dependence-robust variant is monotone-conservative (it can only shrink the discovery set) and is
  the safe default. Actually *enforce* the correction — a computed-but-unused q-value protects
  nothing.
- **Cross-fit, or gate, in-sample scoring (double-dipping).** Fitting a model and scoring it on the
  same data biases the score; the bias scales with the parameter-to-sample ratio `p/n`. Where a full
  cross-fit is too invasive, at minimum *flag or refuse* certification in the regime where `p/n` is
  large enough for the bias to bite (negligible when `n ≫ p`; severe when they are comparable).
- **Guard against leakage and circularity in engineered features.** A quantity derived from a
  variable X must not be used as a prior, weight, or denominator for an estimate about X. Type
  features by provenance and enforce the guard at the point of use, not by convention.

## V. Parameters, defaults, and safety of changes

- **Auto-determine deep mathematical parameters from the data and problem dimensions; do not expose
  them as static user-set constants** — unless the knob serves a genuine functional purpose
  (compute budget, analysis depth). A regularization strength, a range, a shrinkage — these should be
  read from the data (a rate like `√(log p / n)`, an empirical correlogram, a stability criterion),
  and the auto-determination should itself be validated for robustness before you trust it.
- **Prefer monotone-safe changes.** A change that can only tighten or widen (never fabricate a signal)
  bounds its own downside — it cannot manufacture false positives or break recovery by addition. For
  invasive changes that move point estimates, validate explicitly that ground-truth recovery
  survives before adopting.
- **Never silently cap, truncate, or degrade.** If a computation bounds coverage (top-N, no-retry,
  sampling, a fallback path), emit that fact. Silent truncation reads downstream as "covered
  everything" when it did not.

## VI. Tools, data, and acquisition

- **Search for and use established, maintained libraries and datasets. Do not hand-roll ad-hoc
  substitutes, and do not declare a task blocked for lack of data without first searching.** A
  consolidated, well-maintained package is more correct and less debt than on-the-fly raw tables. If
  a capability or dataset is missing, the first move is to find the standard tool that provides it.
- **Acquired capability compounds; foundations are not wasted when the immediate hypothesis fails.**
  Data or infrastructure obtained for one idea (that then proves marginal) is often the necessary
  input to the idea that works. Evaluate acquisitions on their downstream optionality, not only the
  triggering task.

## VII. Process discipline

- **Enforce reliable kill switches on anything long-running; never launch unbounded work.** Know which
  timeout mechanism actually terminates the process on your platform (a shell `timeout` may not kill
  a child process tree). Bound every long operation, and run the full validation *once* after a
  complete batch is implemented — not per intermediate, mathematically-unfinished build.
- **Reconnoiter the current code before building, and verify doc/memory claims against it.** Docs and
  notes are point-in-time; code moves. When a source names a file, function, constant, or "already
  solved" status, confirm it in the live code before relying on it. Parallelize the reconnaissance
  when the surface is wide.
- **Commit in small, single-purpose units.** Each commit isolates one change; the message states what
  changed, why, and the empirical evidence for it. Do not bundle an unvalidated experiment with a
  landed fix.
- **Report faithfully — including negative and surprising results.** "This prescribed feature is
  marginal," "the effect is actually low-rank," "I was directionally wrong" — clearly stated, these
  redirect effort correctly and are as valuable as a success. Never overstate a result or hide a
  refutation; the goal is the true answer, not a satisfying narrative.

## VIII. Performance and scale are empirical — profile, localize, test at the real operating point

- **Profile the real workload before optimizing; attack the dominant cost, not the assumed one.** Read
  the per-stage/per-function breakdown (telemetry, `cProfile`, `py-spy`) on a representative run and let
  Amdahl decide where effort goes. The dominant cost is routinely surprising — an inner solver, not the
  I/O or the "big" model step you assumed. Optimizing a non-dominant stage is bounded waste, however
  clever.
- **Localize *where* a slowdown lives before engineering it — the performance analogue of §III.**
  Isolate the cause by toggling one term/flag at a time. "Slow" resolves to distinct, differently-fixed
  mechanisms: per-iteration cost (vectorize / leave Python objects), *iteration count* (conditioning,
  convergence, step rule), memory thrash (storage layout), or the wrong stage entirely. A solve can be
  slow not from Python-vs-numpy (already numpy) nor storage (already numpy) but from a single
  ill-conditioning penalty on a fine grid — found by toggling that one term; the fix then belongs at the
  optimizer (preconditioning/acceleration), not the loop. And beware: a *naive* accelerated/preconditioned
  optimizer often converges fast to the WRONG point — validate the delicate fix against the reference
  optimum, not merely against being faster (§I.a, §V).
- **Validate at (or near) the actual operating scale — behavior inverts across scale.** Convergence,
  conditioning, and memory are scale-dependent: a solver that converges in tens of iterations on a
  small/coarse proxy can grind for thousands on the real fine/large grid; a routine that fits in RAM at
  one scale thrashes at another. A miniature that "passes" proves little about the workload you run.
- **Distrust a resource-virtue label until measured.** A component named for a virtue can violate it —
  a "memory-bounded" solver built on Python lists used *more* memory than the dense numpy path it
  replaced, and was slower. The name is a claim; the profile is the fact.

## IX. The code is usually ahead of its record — verify current state, including your own past claims

- **Treat every status claim as stale-by-default.** Across cycles a large fraction of "open / broken /
  TODO / already-solved" flags — in roadmaps, audits, saved memories, agent reports, and prior
  self-conclusions — proved already-fixed or adequate on live inspection. The default expectation is
  that the code has moved on: establish the CURRENT state before acting, and don't inherit a plan's
  framing of the problem.
- **A five-minute probe overturns conclusions often enough to be mandatory.** It is not a formality; it
  routinely refutes plausible, authoritative-sounding claims (a "67 GB" store that is now numpy, an
  "in-RAM cliff" that is 0.3 MB, a "grinds forever" solve that converges at 170). Run it before
  building and let the number, not the narrative, decide what is real.
- **Audit your own prior claims with the skepticism you apply to others'.** Your past "done," "the
  bottleneck is X," and headline diagnoses are hypotheses too. Re-verify and correct them promptly when
  a measurement disagrees — without narrative-protection.
- **Existence in code is not use in the live path (orphaned-but-callable ≠ wired-live).** A capability
  that compiles, and even has tests, may not be what the running pipeline calls. Confirm with the actual
  call chain *and* runtime evidence (telemetry of the backend/branch that executed), not the presence of
  a function.
