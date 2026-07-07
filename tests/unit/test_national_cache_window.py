"""T1.1 — the national DATASUS warm-cache must be window-safe (no silent truncation).

A 2000-2024 request must NOT reuse a national artifact built for 2000-2020 just because the
(years-less) file path exists. Reuse is gated on a `.window.json` sidecar matching the exact
requested years; a missing or mismatched window fails closed to a full re-acquire.
"""

from __future__ import annotations

import json

import polars as pl

from pegasus.workflows.acquire.sidra_national import _national_datasus_cached


def _make_artifact(data_root, system, years):
    nat = data_root / "normalized" / "national"
    nat.mkdir(parents=True, exist_ok=True)
    out = nat / f"{system}__processed_events.parquet"
    pl.DataFrame({"municipality_cod6": ["270430"], "year": [2020]}).write_parquet(out)
    if years is not None:
        out.with_suffix(".window.json").write_text(json.dumps({"years": years}))
    return out


def test_missing_window_sidecar_forces_reacquire(tmp_path):
    _make_artifact(tmp_path, "SIM-DO", years=None)   # legacy artifact, no window sidecar
    assert _national_datasus_cached(["SIM-DO"], "2000-2024", tmp_path) is None


def test_window_mismatch_forces_reacquire(tmp_path):
    _make_artifact(tmp_path, "SIM-DO", years="2000-2020")
    # a wider request must NOT reuse the narrower cached artifact
    assert _national_datasus_cached(["SIM-DO"], "2000-2024", tmp_path) is None


def test_exact_window_match_reuses(tmp_path):
    _make_artifact(tmp_path, "SIM-DO", years="2000-2024")
    arts = _national_datasus_cached(["SIM-DO"], "2000-2024", tmp_path)
    assert arts is not None and len(arts) == 1


def test_any_system_missing_forces_reacquire(tmp_path):
    _make_artifact(tmp_path, "SIM-DO", years="2000-2024")   # only one of two systems present
    assert _national_datasus_cached(["SIM-DO", "SINASC"], "2000-2024", tmp_path) is None
