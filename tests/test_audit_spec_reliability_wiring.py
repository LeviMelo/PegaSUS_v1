"""ADVERSARIAL AUDIT — are the §3.12 spec-correct reliability stats WIRED to the live path?

Claim under audit (commit d11922c): the corrected Kish / fragility / sampling-CV /
second-difference-roughness statistics "measure the right quantities" for the live run.

Refutation hypothesis: the three corrected functions
(_denom_fragility_share, _sampling_cv, _second_diff_roughness) live ONLY inside
q_tensor.compute_q_state behind the spec_reliability flag, and compute_q_state has ZERO
production callers. The Q_tensor.parquet that the live LDO consumes is written by
compile_attach._q_row -> _vector_diagnostics, which uses the OLD first-difference roughness
and inline dispersion CV and never touches the corrected functions. Therefore the "fix" is
an unused API — unreachable in production.

Each test below is designed to FAIL if the fix is unwired (flaw real).

Run: C:/Users/Galaxy/miniconda3/envs/pegasus/python.exe -m pytest tests/test_audit_spec_reliability_wiring.py -q
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import polars as pl

import pegasus.efg.compile_attach as ca


# ---------------------------------------------------------------------------
# (A) Live-path wiring: the compile stage writes Q_tensor via _q_row/_vector_diagnostics.
#     A straight ramp is the adversarial input: its SECOND-difference curvature (the spec
#     §3.12.11 quantity) is exactly zero, while the FIRST-difference roughness the live path
#     actually computes is large and proportional to the slope. If the corrected stat were
#     wired into the live write, the persisted temporal_roughness would be ~0; if the OLD
#     proxy is still what production writes, it is large. We assert the SPEC value — so the
#     test FAILS while the fix is unwired.
# ---------------------------------------------------------------------------

def _ramp_panel_and_vector(slope: float = 100.0, n_years: int = 12):
    """A single national year-series that is a perfect straight line.

    Panel index is one municipality across n_years so `year` drives temporal roughness.
    """
    years = list(range(2000, 2000 + n_years))
    vector = [float(slope * i) for i in range(n_years)]  # 0, slope, 2*slope, ... a line
    panel = pl.DataFrame(
        {
            "row_id": list(range(n_years)),
            "year": years,
            "municipality_cod6": ["270000"] * n_years,
        }
    )
    return panel, vector


def test_live_compile_writes_spec_correct_curvature_roughness():
    panel, vector = _ramp_panel_and_vector(slope=100.0, n_years=12)

    diag = ca._vector_diagnostics(vector, panel=panel)
    rough = diag["temporal_roughness"]
    assert rough is not None, "live path produced no temporal_roughness at all"

    # Spec §3.12.11 curvature of a straight line is ZERO (second difference vanishes).
    # If the corrected _second_diff_roughness were the one wired into the live compile path,
    # this would be ~0. The OLD first-difference proxy returns slope/mean ≈ 0.18 for this ramp.
    assert rough < 1e-6, (
        "LIVE compile path did NOT use the spec-correct second-difference roughness: "
        f"a straight ramp has zero curvature but the persisted temporal_roughness={rough:.6f} "
        "(this is the OLD first-difference proxy — the corrected stat is unwired)."
    )


# ---------------------------------------------------------------------------
# (B) The corrected functions have NO production caller. compute_q_state is the only
#     function that invokes them, and grep of src/ shows compute_q_state itself is never
#     called outside tests. Assert compile_attach never references the three corrected names.
# ---------------------------------------------------------------------------

def test_compile_attach_calls_spec_correct_stat_functions():
    # Remediation contract: the live Q_tensor writer NOW routes through the corrected §3.12 stats
    # (AST-based — a mention in a comment does not count, only a real call).
    tree = ast.parse(Path(inspect.getfile(ca)).read_text(encoding="utf-8"))
    targets = {"_denom_fragility_share", "_sampling_cv", "_second_diff_roughness"}
    called: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fn = node.func
            nm = fn.attr if isinstance(fn, ast.Attribute) else (fn.id if isinstance(fn, ast.Name) else None)
            if nm in targets:
                called.add(nm)
    assert called, "compile_attach must call the corrected §3.12 stats on the live Q_tensor write"


def test_no_production_caller_invokes_compute_q_state_or_corrected_stats():
    """Walk every src/pegasus/**.py and confirm no production module CALLS compute_q_state
    or the three corrected stat functions. (Definitions in q_tensor.py itself don't count.)"""
    root = Path(inspect.getfile(ca)).resolve().parents[1]  # .../pegasus
    targets = {"compute_q_state", "_denom_fragility_share", "_sampling_cv", "_second_diff_roughness"}
    definer = (root / "efg" / "q_tensor.py").resolve()

    callers: dict[str, list[str]] = {t: [] for t in targets}
    for py in root.rglob("*.py"):
        if py.resolve() == definer:
            continue  # the module that defines them is allowed to reference them
        try:
            tree = ast.parse(py.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                fn = node.func
                nm = fn.attr if isinstance(fn, ast.Attribute) else (fn.id if isinstance(fn, ast.Name) else None)
                if nm in targets:
                    callers[nm].append(str(py.relative_to(root)))

    # Audit expectation: EMPTY. If the fix were wired to production, compute_q_state (or a
    # corrected stat) would be called somewhere under src/pegasus. This asserts a live caller
    # exists, so it FAILS while the fix is unwired.
    live = {k: v for k, v in callers.items() if v}
    assert live, (
        "No production module under src/pegasus/ calls compute_q_state or any corrected §3.12 "
        "stat (_denom_fragility_share/_sampling_cv/_second_diff_roughness). The corrected "
        "statistics are an unused API: the live Q_tensor.parquet is written by "
        "compile_attach._q_row/_vector_diagnostics, which uses the OLD proxies. UNWIRED FIX."
    )


# ---------------------------------------------------------------------------
# (C) Even the flag that *could* reach spec_reliability=True is never set true in production,
#     and — critically — it does NOT gate the three corrected stat functions anyway. The
#     investigate.run_investigate default is spec_reliability=False, and state_reliability_weights
#     (the sole live consumer of Q_tensor) reads only n_events/n_denom/n_eff/denom_fragility/
#     provenance_risk — never cv or temporal_roughness. So _sampling_cv / _second_diff_roughness
#     have no live consumer even in principle.
# ---------------------------------------------------------------------------

def test_live_reliability_weight_consumer_ignores_sampling_cv_and_roughness():
    from pegasus.workflows.investigate import state_reliability_weights

    src = inspect.getsource(state_reliability_weights)
    # The live consumer reads none of the corrected-stat output columns.
    for col in ("cv", "temporal_roughness", "sampling_cv", "second_diff"):
        assert f'"{col}"' not in src and f"'{col}'" not in src, (
            f"state_reliability_weights reads column {col!r} — re-audit; audit expects it does not, "
            "so the corrected sampling-CV / roughness stats have no live consumer."
        )
