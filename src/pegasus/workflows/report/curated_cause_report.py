"""Curated-cause epidemiological reporter (describe-only).

A study is a QUERY + DESCRIPTION over a PegaSUS run bundle, never a
recomputation of the epidemiology. This reporter reads the artifacts the
pipeline produced — the σ_C curated-cause count tensor (additive), the
``CuratedCauseMortality`` MeasuredQuantity (numerator + SIDRA-denominator
exposure), the Q_tensor (spatial/temporal structure + uncertainty), and the
observer-quality shares — and arranges them into a spatial/temporal/uncertainty
report for one curated cause group (e.g. ``neoplasm_pancreas`` = ICD-10 C25).

Every number rendered is either a value the pipeline emitted or a legal
aggregation of a produced additive measure (national/UF totals of an additive
count; pooled rate = Σnumerator / Σexposure of the produced MeasuredQuantity).
No cause assignment, denominator, or rate is computed here from raw records.
"""

from __future__ import annotations

import html
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import polars as pl

from pegasus.geo.uf import uf_from_datasus_cod6


class CuratedCauseReportError(ValueError):
    """Raised when the bundle lacks the artifacts a curated-cause report requires."""


# ---------------------------------------------------------------------------
# Bundle access
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _Bundle:
    run_dir: Path
    v_fields: pl.DataFrame
    id_by_name: dict[str, str]
    tensor_dir: Path

    @classmethod
    def load(cls, run_dir: str | Path) -> "_Bundle":
        run_dir = Path(run_dir)
        vf_path = run_dir / "V_fields.parquet"
        if not vf_path.exists():
            raise CuratedCauseReportError(f"No V_fields.parquet in run dir: {run_dir}")
        v = pl.read_parquet(vf_path)
        id_by_name: dict[str, str] = {}
        for name, fid in zip(v["name"].to_list(), v["field_id"].to_list()):
            id_by_name.setdefault(str(name), str(fid))
        return cls(run_dir=run_dir, v_fields=v, id_by_name=id_by_name, tensor_dir=run_dir / "Tables" / "efg_tensors")

    def field_id(self, name: str) -> str | None:
        return self.id_by_name.get(name)

    def tensor(self, name: str, *, measured_quantity: bool = False) -> pl.DataFrame | None:
        fid = self.field_id(name)
        if fid is None:
            return None
        suffix = ".measured_quantity.parquet" if measured_quantity else ".parquet"
        path = self.tensor_dir / f"{fid}{suffix}"
        if not path.exists():
            return None
        return pl.read_parquet(path)

    def q_row(self, name: str) -> dict[str, Any] | None:
        fid = self.field_id(name)
        q_path = self.run_dir / "Q_tensor.parquet"
        if fid is None or not q_path.exists():
            return None
        q = pl.read_parquet(q_path).filter(pl.col("field_id") == fid)
        return q.row(0, named=True) if q.height else None

    def run_config(self) -> dict[str, Any]:
        # RunConfig carries compile/population settings; UserIntent carries scale/stage/window.
        # Merge both (UserIntent fills gaps) so the report footer is complete either way.
        merged: dict[str, Any] = {}
        for candidate in ("RunConfig.json", "UserIntent.json"):
            p = self.run_dir / candidate
            if not p.exists():
                continue
            try:
                payload = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            for k, v in payload.items():
                merged.setdefault(k, v)
        return merged


# Field names the σ_C curated-cause traversal emits (see efg/diagnostic_strata.py).
_COUNT_FIELD = "SIM-DO.SIM-DO__processed_events.count.curated_cause_group"
_RATE_FIELD = "CuratedCauseMortality"
_GROUP_COL = "curated_cause_group"


# ---------------------------------------------------------------------------
# Report model
# ---------------------------------------------------------------------------

@dataclass
class CuratedCauseReport:
    cause_group: str
    icd_label: str
    scope: str
    years: tuple[int, int]
    total_deaths: float
    deaths_by_year: list[tuple[int, float]]
    municipalities_with_deaths: int
    national_rate_per_100k_by_year: list[tuple[int, float | None]]
    by_uf: list[dict[str, Any]]
    spatial: dict[str, Any]
    quality: dict[str, Any]
    provenance: dict[str, Any]
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "cause_group": self.cause_group,
            "icd_label": self.icd_label,
            "scope": self.scope,
            "years": list(self.years),
            "total_deaths": self.total_deaths,
            "deaths_by_year": self.deaths_by_year,
            "municipalities_with_deaths": self.municipalities_with_deaths,
            "national_rate_per_100k_by_year": self.national_rate_per_100k_by_year,
            "by_uf": self.by_uf,
            "spatial": self.spatial,
            "quality": self.quality,
            "provenance": self.provenance,
            "warnings": self.warnings,
        }


def _uf_of(cod6: str | None) -> str | None:
    if cod6 is None:
        return None
    try:
        return uf_from_datasus_cod6(str(cod6))
    except Exception:
        return None


def build_curated_cause_report(
    run_dir: str | Path,
    *,
    cause_group: str = "neoplasm_pancreas",
    icd_label: str = "Malignant neoplasm of pancreas (ICD-10 C25)",
) -> CuratedCauseReport:
    """Read a run bundle and describe one curated cause's national epidemiology."""
    bundle = _Bundle.load(run_dir)
    warnings: list[str] = []

    count = bundle.tensor(_COUNT_FIELD)
    if count is None:
        raise CuratedCauseReportError(
            f"Run bundle has no curated-cause count tensor ({_COUNT_FIELD}). "
            "The intent must declare health_seed 'curated_cause' (not 'curated_cause_mortality')."
        )
    if _GROUP_COL not in count.columns:
        raise CuratedCauseReportError(f"Count tensor is missing the '{_GROUP_COL}' axis.")
    groups = set(count[_GROUP_COL].unique().to_list())
    if cause_group not in groups:
        raise CuratedCauseReportError(
            f"Curated cause '{cause_group}' not in materialized groups ({sorted(g for g in groups if g)[:8]}…)."
        )

    c = count.filter(pl.col(_GROUP_COL) == cause_group)
    # Real municipalities only (null-muni rows are aggregate/unknown residuals).
    c_muni = c.filter(pl.col("municipality_cod6").is_not_null())

    total_deaths = float(c["value"].sum() or 0.0)
    by_year = (
        c.group_by("year").agg(pl.col("value").sum().alias("deaths")).sort("year")
    )
    deaths_by_year = [(int(y), float(d)) for y, d in zip(by_year["year"].to_list(), by_year["deaths"].to_list())]
    years = (min(y for y, _ in deaths_by_year), max(y for y, _ in deaths_by_year)) if deaths_by_year else (0, 0)
    munis_with_deaths = c_muni.filter(pl.col("value") > 0)["municipality_cod6"].n_unique()

    # --- Rates from the produced MeasuredQuantity (numerator + SIDRA exposure) ---
    rate = bundle.tensor(_RATE_FIELD, measured_quantity=True)
    national_rate_by_year: list[tuple[int, float | None]] = []
    uf_exposure: dict[str, float] = {}
    if rate is not None and {_GROUP_COL, "numerator_count", "exposure"} <= set(rate.columns):
        r = rate.filter((pl.col(_GROUP_COL) == cause_group) & pl.col("municipality_cod6").is_not_null())
        ry = (
            r.group_by("year")
            .agg(pl.col("numerator_count").sum().alias("num"), pl.col("exposure").sum().alias("exp"))
            .sort("year")
        )
        for y, num, exp in zip(ry["year"].to_list(), ry["num"].to_list(), ry["exp"].to_list()):
            rate_val = (float(num) / float(exp) * 1e5) if exp and float(exp) > 0 else None
            national_rate_by_year.append((int(y), rate_val))
        # UF exposure (population) pooled over the window, for UF crude rates.
        r_uf = r.with_columns(
            pl.col("municipality_cod6").map_elements(_uf_of, return_dtype=pl.Utf8).alias("uf")
        )
        ue = r_uf.group_by("uf").agg(pl.col("exposure").sum().alias("exp"))
        uf_exposure = {u: float(e) for u, e in zip(ue["uf"].to_list(), ue["exp"].to_list()) if u}
    else:
        warnings.append("No CuratedCauseMortality MeasuredQuantity — rates unavailable (denominator not compiled).")

    # --- Spatial: by-UF deaths (additive) + pooled crude rate where exposure exists ---
    c_uf = c_muni.with_columns(
        pl.col("municipality_cod6").map_elements(_uf_of, return_dtype=pl.Utf8).alias("uf")
    )
    ud = c_uf.group_by("uf").agg(pl.col("value").sum().alias("deaths")).sort("deaths", descending=True)
    by_uf: list[dict[str, Any]] = []
    for u, d in zip(ud["uf"].to_list(), ud["deaths"].to_list()):
        if not u:
            continue
        exp = uf_exposure.get(u)
        crude = (float(d) / exp * 1e5) if exp and exp > 0 else None
        by_uf.append({"uf": u, "deaths": float(d), "rate_per_100k": crude})

    # --- Spatial structure + uncertainty from the produced Q_tensor ---
    q_count = bundle.q_row(_COUNT_FIELD) or {}
    q_rate = bundle.q_row(_RATE_FIELD) or {}
    spatial = {
        "moran_i": q_rate.get("moran_i") if q_rate.get("moran_i") is not None else q_count.get("moran_i"),
        "spatial_entropy": q_rate.get("spatial_entropy") if q_rate.get("spatial_entropy") is not None else q_count.get("spatial_entropy"),
        "cov_S": q_rate.get("cov_S") if q_rate.get("cov_S") is not None else q_count.get("cov_S"),
        "interpretation": None,
    }
    mi = spatial["moran_i"]
    if isinstance(mi, (int, float)):
        if mi > 0.2:
            spatial["interpretation"] = "spatially clustered (positive autocorrelation)"
        elif mi < -0.05:
            spatial["interpretation"] = "spatially dispersed (negative autocorrelation)"
        else:
            spatial["interpretation"] = "no strong spatial autocorrelation"

    quality = {
        "cv": q_rate.get("cv") if q_rate else q_count.get("cv"),
        "n_eff": q_rate.get("n_eff") if q_rate else q_count.get("n_eff"),
        "provenance_risk": q_rate.get("provenance_risk") if q_rate else q_count.get("provenance_risk"),
        "field_state": q_rate.get("state") if q_rate else q_count.get("state"),
    }
    for share_name, key in (("IllDefinedCauseShare", "ill_defined_cause_share"), ("InvalidICDShare", "invalid_icd_share")):
        t = bundle.tensor(share_name)
        if t is not None and "value" in t.columns:
            vals = [v for v in t["value"].to_list() if v is not None]
            if vals:
                quality[key] = float(sum(vals) / len(vals))

    rc = bundle.run_config()
    provenance = {
        "run_dir": str(bundle.run_dir),
        "execution_scale": rc.get("execution_scale"),
        "execution_stage": rc.get("execution_stage"),
        "population_mode": rc.get("population_mode"),
        "source": "DATASUS SIM-DO (underlying cause) + IBGE/SIDRA population denominator",
        "field_count": bundle.v_fields.height,
        "cause_assignment": "PegaSUS σ_C curated-cause partition (icd_curated_groups.yaml)",
    }

    scope = "National (all 27 UFs, municipality resolution)" if rc.get("execution_scale") == "national" else str(rc.get("execution_scale") or "unknown")

    return CuratedCauseReport(
        cause_group=cause_group,
        icd_label=icd_label,
        scope=scope,
        years=years,
        total_deaths=total_deaths,
        deaths_by_year=deaths_by_year,
        municipalities_with_deaths=int(munis_with_deaths),
        national_rate_per_100k_by_year=national_rate_by_year,
        by_uf=by_uf,
        spatial=spatial,
        quality=quality,
        provenance=provenance,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------

def _fmt(x: Any, nd: int = 1) -> str:
    if x is None:
        return "—"
    if isinstance(x, float):
        return f"{x:,.{nd}f}"
    return html.escape(str(x))


def _sparkline(points: list[float], *, width: int = 220, height: int = 40) -> str:
    if not points:
        return ""
    lo, hi = min(points), max(points)
    span = (hi - lo) or 1.0
    n = len(points)
    step = width / max(1, n - 1)
    coords = [
        (i * step, height - (p - lo) / span * (height - 6) - 3)
        for i, p in enumerate(points)
    ]
    path = " ".join(f"{'M' if i == 0 else 'L'}{x:.1f},{y:.1f}" for i, (x, y) in enumerate(coords))
    dots = "".join(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.4" fill="#0b6"/>' for x, y in coords)
    return (
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
        f'<path d="{path}" fill="none" stroke="#0b6" stroke-width="2"/>{dots}</svg>'
    )


def render_curated_cause_html(report: CuratedCauseReport) -> str:
    r = report
    yr_rows = "".join(
        f"<tr><td>{y}</td><td>{_fmt(d,0)}</td><td>{_fmt(dict(r.national_rate_per_100k_by_year).get(y))}</td></tr>"
        for y, d in r.deaths_by_year
    )
    uf_rows = "".join(
        f"<tr><td>{html.escape(u['uf'])}</td><td>{_fmt(u['deaths'],0)}</td><td>{_fmt(u['rate_per_100k'])}</td></tr>"
        for u in r.by_uf[:15]
    )
    spark = _sparkline([d for _, d in r.deaths_by_year])
    warn_html = ""
    if r.warnings:
        warn_html = "<div class='warn'><b>Caveats:</b><ul>" + "".join(f"<li>{html.escape(w)}</li>" for w in r.warnings) + "</ul></div>"
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>{html.escape(r.icd_label)} — PegaSUS</title>
<style>
body{{font:15px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;max-width:900px;margin:2rem auto;padding:0 1rem;color:#1a1a1a}}
h1{{font-size:1.5rem;margin-bottom:.2rem}} h2{{font-size:1.1rem;margin-top:2rem;border-bottom:2px solid #0b6;padding-bottom:.2rem}}
.sub{{color:#666;margin-top:0}} table{{border-collapse:collapse;width:100%;margin:.6rem 0}}
th,td{{text-align:left;padding:.35rem .6rem;border-bottom:1px solid #eee}} th{{color:#555;font-weight:600}}
td:nth-child(n+2),th:nth-child(n+2){{text-align:right;font-variant-numeric:tabular-nums}}
.kpis{{display:flex;gap:1.5rem;flex-wrap:wrap;margin:1rem 0}} .kpi{{background:#f6f9f7;border:1px solid #dceee4;border-radius:8px;padding:.7rem 1rem}}
.kpi b{{display:block;font-size:1.4rem;color:#0b6}} .kpi span{{color:#666;font-size:.85rem}}
.warn{{background:#fff8e6;border:1px solid #f0d98a;border-radius:8px;padding:.5rem 1rem;margin-top:1rem;font-size:.9rem}}
footer{{margin-top:2.5rem;color:#888;font-size:.82rem;border-top:1px solid #eee;padding-top:.8rem}}
</style></head><body>
<h1>{html.escape(r.icd_label)}</h1>
<p class="sub">{html.escape(r.scope)} · {r.years[0]}–{r.years[1]} · produced by PegaSUS from DATASUS + IBGE/SIDRA</p>
<div class="kpis">
  <div class="kpi"><b>{_fmt(r.total_deaths,0)}</b><span>total deaths, window</span></div>
  <div class="kpi"><b>{_fmt(r.deaths_by_year[-1][1],0) if r.deaths_by_year else '—'}</b><span>deaths in {r.years[1]}</span></div>
  <div class="kpi"><b>{_fmt(dict(r.national_rate_per_100k_by_year).get(r.years[1]))}</b><span>rate /100k ({r.years[1]})</span></div>
  <div class="kpi"><b>{r.municipalities_with_deaths:,}</b><span>municipalities affected</span></div>
</div>
<h2>Temporal trend {spark}</h2>
<table><tr><th>Year</th><th>Deaths</th><th>Rate /100k</th></tr>{yr_rows}</table>
<h2>Spatial distribution</h2>
<p>Spatial autocorrelation (Moran's I): <b>{_fmt(r.spatial.get('moran_i'),3)}</b> — {html.escape(str(r.spatial.get('interpretation') or 'n/a'))}.</p>
<table><tr><th>UF</th><th>Deaths (window)</th><th>Rate /100k</th></tr>{uf_rows}</table>
<h2>Data quality &amp; uncertainty</h2>
<table>
<tr><th>Metric</th><th>Value</th></tr>
<tr><td>Coefficient of variation (CV)</td><td>{_fmt(r.quality.get('cv'),3)}</td></tr>
<tr><td>Effective sample size (n_eff)</td><td>{_fmt(r.quality.get('n_eff'),0)}</td></tr>
<tr><td>Ill-defined cause share (chapter R)</td><td>{_fmt((r.quality.get('ill_defined_cause_share') or 0)*100,2)}%</td></tr>
<tr><td>Invalid ICD share</td><td>{_fmt((r.quality.get('invalid_icd_share') or 0)*100,2)}%</td></tr>
<tr><td>Provenance risk</td><td>{_fmt(r.quality.get('provenance_risk'),3)}</td></tr>
<tr><td>Field state</td><td>{html.escape(str(r.quality.get('field_state') or '—'))}</td></tr>
</table>
{warn_html}
<footer>
Cause assignment: {html.escape(r.provenance.get('cause_assignment') or '')}.
Denominator: {html.escape(r.provenance.get('source') or '')}.
Scale={html.escape(str(r.provenance.get('execution_scale')))}, stage={html.escape(str(r.provenance.get('execution_stage')))},
population_mode={html.escape(str(r.provenance.get('population_mode')))}, fields={r.provenance.get('field_count')}.
All figures are values PegaSUS emitted or legal aggregations of its additive measures — no epidemiology recomputed in the reporter.
</footer>
</body></html>"""


def write_curated_cause_report(
    run_dir: str | Path,
    *,
    cause_group: str = "neoplasm_pancreas",
    icd_label: str = "Malignant neoplasm of pancreas (ICD-10 C25)",
    out_html: str | Path | None = None,
    out_json: str | Path | None = None,
) -> CuratedCauseReport:
    report = build_curated_cause_report(run_dir, cause_group=cause_group, icd_label=icd_label)
    if out_html is not None:
        Path(out_html).write_text(render_curated_cause_html(report), encoding="utf-8")
    if out_json is not None:
        Path(out_json).write_text(json.dumps(report.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return report


__all__ = [
    "CuratedCauseReport",
    "CuratedCauseReportError",
    "build_curated_cause_report",
    "render_curated_cause_html",
    "write_curated_cause_report",
]
