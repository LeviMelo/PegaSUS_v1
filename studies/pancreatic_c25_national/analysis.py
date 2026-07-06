"""Exhaustive national C25 (pancreatic cancer) epidemiology analysis.

Leverages PegaSUS outputs: FAL-POP national population tensor (denominators), the compiled
Q-tensor + EFG (state diagnostics), the LDO Hypotheses (inference), and the PegaSUS-normalized
SIM-DO mortality events. Computes crude + directly age-standardized rates across every stratum,
spatial + data-quality + comparative structure, and the inference payload. Dumps a structured
JSON + granular parquet tables consumed by the HTML report."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import polars as pl

import sys
sys.path.insert(0, "src")
from pegasus.measurement.age_standardization import standardize_grouped, load_reference_population, directly_standardized_rate

ROOT = Path("C:/Users/Galaxy/LEVI/PegaSUS")
SP = Path("C:/Users/Galaxy/AppData/Local/Temp/claude/C--Users-Galaxy-LEVI-PegaSUS/45c58f56-c614-422b-a2c7-b22ff017ef77/scratchpad")
OUT = SP / "c25_study"; OUT.mkdir(exist_ok=True)
RUN = ROOT / "data/runs/national_c25_poptensor"
YEARS = [2021, 2022]
RACE = {"1": "branca", "2": "preta", "3": "amarela", "4": "parda", "5": "indigena"}
RACE_PT = {"branca": "White", "preta": "Black", "amarela": "Asian-desc.", "parda": "Mixed (Parda)", "indigena": "Indigenous"}
REGION = {"1": "North", "2": "Northeast", "3": "Southeast", "4": "South", "5": "Center-West"}
UF = {"11":"RO","12":"AC","13":"AM","14":"RR","15":"PA","16":"AP","17":"TO","21":"MA","22":"PI","23":"CE","24":"RN","25":"PB","26":"PE","27":"AL","28":"SE","29":"BA","31":"MG","32":"ES","33":"RJ","35":"SP","41":"PR","42":"SC","43":"RS","50":"MS","51":"MT","52":"GO","53":"DF"}
res: dict = {"meta": {"years": YEARS, "cause": "C25 (malignant neoplasm of pancreas)", "source": "SIM-DO via PegaSUS; denominators FAL-POP national population tensor"}}

# ---------- deaths ----------
sim = pl.scan_parquet(ROOT / "data/normalized/national/SIM-DO__processed_events.parquet").filter(pl.col("year").is_in(YEARS))
icd = pl.col("underlying_icd_raw").cast(pl.Utf8)
def cause(expr, pref):  # startswith any of pref (list)
    e = None
    for p in pref:
        c = expr.str.starts_with(p)
        e = c if e is None else (e | c)
    return e
deaths_all = sim.select(pl.len()).collect().item()
c25 = sim.filter(icd.str.starts_with("C25")).with_columns(
    race_name=pl.col("race_color_admin").cast(pl.Utf8).replace_strict(RACE, default=None),
    age_i=pl.col("age_years").cast(pl.Float64).floor().cast(pl.Int64),
    reg=pl.col("mun_residence_cod6").cast(pl.Utf8).str.slice(0,1).replace_strict(REGION, default=None),
    uf=pl.col("mun_residence_cod6").cast(pl.Utf8).str.slice(0,2).replace_strict(UF, default=None),
).collect()
n_c25 = c25.height
res["headline"] = {"c25_deaths": n_c25, "all_deaths": deaths_all, "c25_pct_of_all_deaths": round(100*n_c25/deaths_all,3)}

# comparison causes
digest = sim.filter(cause(icd, [f"C{n}" for n in range(15,27)])).select(pl.len()).collect().item()
neopl = sim.filter(icd.str.starts_with("C") | (icd.str.starts_with("D") & (icd.str.slice(1,2).cast(pl.Int32,strict=False) < 49))).select(pl.len()).collect().item()
res["comparative"] = {"c25": n_c25, "digestive_cancer_C15_C26": digest, "all_neoplasm_C00_D48": neopl, "all_cause": deaths_all,
                      "c25_pct_of_digestive_ca": round(100*n_c25/digest,2), "c25_pct_of_neoplasm": round(100*n_c25/neopl,2)}

# ---------- population (FAL-POP) ----------
pop = (pl.scan_parquet(SP / "national_population_tensor.parquet")
       .filter(pl.col("year").is_in(YEARS))
       .with_columns(age_i=pl.when(pl.col("age_group")=="age_100_plus").then(100)
                          .otherwise(pl.col("age_group").str.replace("age_","").cast(pl.Int64, strict=False)),
                     reg=pl.col("municipality_cod6").cast(pl.Utf8).str.slice(0,1).replace_strict(REGION, default=None),
                     uf=pl.col("municipality_cod6").cast(pl.Utf8).str.slice(0,2).replace_strict(UF, default=None))
       .select("year","municipality_cod6","age_i","sex","race","value","reg","uf")).collect()
py_total = pop["value"].sum()  # person-years (summed over both years)
res["denominator"] = {"person_years_total": float(py_total), "mean_annual_pop": float(py_total/len(YEARS))}

# ---------- helper: deaths-by-age + pop-by-age for a stratum key ----------
def asr_by(strata: list[str], death_keys: list[str], pop_keys: list[str], ref="who_world_2000_2025"):
    d = c25.group_by(strata + ["age_i"]).agg(pl.len().alias("deaths"))
    n = pop.group_by([pop_keys[i] for i in range(len(strata))] + ["age_i"]).agg(pl.col("value").sum().alias("pop")) if strata else pop.group_by("age_i").agg(pl.col("value").sum().alias("pop"))
    # rename pop strata cols to match death strata
    for i,k in enumerate(strata):
        n = n.rename({pop_keys[i]: k}) if pop_keys[i]!=k else n
    j = d.join(n, on=strata+["age_i"], how="full", coalesce=True).with_columns(pl.col("deaths").fill_null(0), pl.col("pop").fill_null(0.0))
    return standardize_grouped(j, age_col="age_i", deaths_col="deaths", pop_col="pop", by=strata, reference=ref, root=str(ROOT))

# national ASMR (WHO + Brazil)
who = asr_by([], [], [])
br = asr_by([], [], [], ref="brazil_2010_census")
res["national_rates"] = {
    "crude_per_100k_yr": round(who["crude"][0],3),
    "asmr_who_per_100k": round(who["asr"][0],3), "asmr_who_ci": [round(who["ci_low"][0],3), round(who["ci_high"][0],3)],
    "asmr_brazil2010_per_100k": round(br["asr"][0],3),
}

# by sex
sx = asr_by(["sex"], ["sex"], ["sex"]).sort("sex")
res["by_sex"] = [{"sex": r["sex"], "deaths": r["deaths"], "crude": round(r["crude"],3), "asmr_who": round(r["asr"],3), "ci": [round(r["ci_low"],3), round(r["ci_high"],3)]} for r in sx.iter_rows(named=True)]
mf = {r["sex"]: r["asr"] for r in sx.iter_rows(named=True)}
res["sex_ratio_M_F_asmr"] = round(mf.get("male",0)/mf["female"],3) if mf.get("female") else None

# by race
c25r = c25.filter(pl.col("race_name").is_not_null())
rc = asr_by(["race_name"], ["race_name"], ["race"]).sort("asr", descending=True) if True else None
# race: death strata col 'race_name', pop strata col 'race'
d = c25r.group_by(["race_name","age_i"]).agg(pl.len().alias("deaths"))
n = pop.group_by(["race","age_i"]).agg(pl.col("value").sum().alias("pop")).rename({"race":"race_name"})
j = d.join(n, on=["race_name","age_i"], how="full", coalesce=True).with_columns(pl.col("deaths").fill_null(0), pl.col("pop").fill_null(0.0))
rc = standardize_grouped(j, age_col="age_i", deaths_col="deaths", pop_col="pop", by=["race_name"], reference="who_world_2000_2025", root=str(ROOT)).sort("asr", descending=True)
res["by_race"] = [{"race": RACE_PT.get(r["race_name"], r["race_name"]), "deaths": r["deaths"], "crude": round(r["crude"],3), "asmr_who": round(r["asr"],3), "ci":[round(r["ci_low"],3),round(r["ci_high"],3)]} for r in rc.iter_rows(named=True)]

# by age group (age-specific rates)
ag_lbls = load_reference_population(root=str(ROOT))
d_age = c25.group_by("age_i").agg(pl.len().alias("deaths"))
n_age = pop.group_by("age_i").agg(pl.col("value").sum().alias("pop"))
ja = d_age.join(n_age, on="age_i", how="full", coalesce=True).with_columns(pl.col("deaths").fill_null(0), pl.col("pop").fill_null(0.0))
ref = load_reference_population(root=str(ROOT)); gi = ref.age_group_index(ja["age_i"].fill_null(0).to_numpy().astype(float))
ja = ja.with_columns(pl.Series("g", gi)).group_by("g").agg(pl.col("deaths").sum(), pl.col("pop").sum()).sort("g")
res["by_age_group"] = [{"age_group": ref.group_labels[int(r["g"])], "deaths": int(r["deaths"]), "person_years": float(r["pop"]), "rate_per_100k": round(1e5*r["deaths"]/r["pop"],2) if r["pop"]>0 else None} for r in ja.iter_rows(named=True)]

# by region + UF
for key, keyname in [("reg","by_region"), ("uf","by_uf")]:
    d = c25.filter(pl.col(key).is_not_null()).group_by([key,"age_i"]).agg(pl.len().alias("deaths"))
    n = pop.filter(pl.col(key).is_not_null()).group_by([key,"age_i"]).agg(pl.col("value").sum().alias("pop"))
    j = d.join(n, on=[key,"age_i"], how="full", coalesce=True).with_columns(pl.col("deaths").fill_null(0), pl.col("pop").fill_null(0.0))
    st = standardize_grouped(j, age_col="age_i", deaths_col="deaths", pop_col="pop", by=[key], reference="who_world_2000_2025", root=str(ROOT)).sort("asr", descending=True)
    res[keyname] = [{key: r[key], "deaths": r["deaths"], "crude": round(r["crude"],3), "asmr_who": round(r["asr"],3), "ci":[round(r["ci_low"],3),round(r["ci_high"],3)]} for r in st.iter_rows(named=True)]

# temporal (per year crude)
tmp = []
for y in YEARS:
    dd = c25.filter(pl.col("year")==y).height
    nn = pop.filter(pl.col("year")==y)["value"].sum()
    tmp.append({"year": y, "deaths": dd, "crude_per_100k": round(1e5*dd/nn,3)})
res["temporal"] = tmp

# place of death + facility
pod = c25.group_by("place_of_death").agg(pl.len().alias("n")).sort("n", descending=True)
res["place_of_death"] = [{"place": str(r["place_of_death"]), "deaths": r["n"], "pct": round(100*r["n"]/n_c25,2)} for r in pod.iter_rows(named=True)][:12]

# data quality
res["data_quality"] = {
    "missing_race_pct": round(100*c25.filter(pl.col("race_name").is_null()).height/n_c25,3),
    "missing_age_pct": round(100*c25.filter(pl.col("age_i").is_null()).height/n_c25,3),
    "missing_sex_pct": round(100*c25.filter(pl.col("sex").is_null()).height/n_c25,3),
    "missing_muni_pct": round(100*c25.filter(pl.col("reg").is_null()).height/n_c25,3),
}

# municipality granular (crude rate; require pop>0)
dm = c25.group_by("mun_residence_cod6").agg(pl.len().alias("deaths"))
nm = pop.group_by("municipality_cod6").agg(pl.col("value").sum().alias("pop")).rename({"municipality_cod6":"mun_residence_cod6"})
muni = dm.join(nm, on="mun_residence_cod6", how="inner").with_columns(crude=(1e5*pl.col("deaths")/pl.col("pop"))).sort("deaths", descending=True)
muni.write_parquet(OUT / "c25_by_municipality.parquet")
res["municipality"] = {"n_municipalities_with_deaths": muni.height, "n_total_munis_with_pop": nm.height,
                       "top10_by_deaths": [{"cod6": r["mun_residence_cod6"], "deaths": r["deaths"], "crude": round(r["crude"],2)} for r in muni.head(10).iter_rows(named=True)]}

# ---------- PegaSUS system outputs ----------
qt = pl.read_parquet(RUN / "Q_tensor.parquet")
res["qtensor"] = {
    "n_fields": qt.height,
    "state_counts": dict(zip(qt["state"].value_counts()["state"].to_list(), qt["state"].value_counts()["count"].to_list())),
    "moran_i_summary": {"mean": round(float(qt["moran_i"].drop_nulls().mean() or 0),4), "max": round(float(qt["moran_i"].drop_nulls().max() or 0),4)},
    "mean_n_eff": round(float(qt["n_eff"].drop_nulls().mean() or 0),1),
}
try:
    hyp = pl.read_parquet(RUN / "Hypotheses.parquet")
    res["ldo"] = {"n_link_records": hyp.height, "columns": hyp.columns[:20]}
    if hyp.height and "certification_status" in hyp.columns:
        res["ldo"]["certification"] = dict(zip(hyp["certification_status"].value_counts()["certification_status"].to_list(), hyp["certification_status"].value_counts()["count"].to_list()))
    hyp.write_parquet(OUT / "ldo_hypotheses.parquet")
except Exception as e:
    res["ldo"] = {"error": str(e)}

(OUT / "c25_results.json").write_text(json.dumps(res, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
print("=== C25 ANALYSIS COMPLETE ===")
print(json.dumps({k: res[k] for k in ("headline","national_rates","sex_ratio_M_F_asmr","comparative","data_quality")}, indent=1, ensure_ascii=False))
print("by_sex:", res["by_sex"]); print("by_race:", res["by_race"])
print("by_region:", res["by_region"])
print("wrote:", OUT)
