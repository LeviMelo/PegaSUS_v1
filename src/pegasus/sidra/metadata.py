from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl

from pegasus.sidra.api import SidraClient
from pegasus.sidra.schemas import SIDRAMetadata, SIDRATableMetadata


def fixture_sidra_metadata() -> SIDRAMetadata:
    """Deterministic SIDRA metadata fixture for local planner/hash tests."""
    table = SIDRATableMetadata(
        table_id="9606",
        name="Populacao residente",
        variables=["93"],
        periods=["2022"],
        locality_levels=["N6"],
        localities_by_level={"N6": ["2704302", "2700300"]},
        classifications={
            "2": ["6794"],
            "86": ["95251"],
            "287": ["100362"],
        },
        units_by_variable={"93": "Pessoas"},
    )
    return SIDRAMetadata(tables={"9606": table})


def load_table_seed(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows


def table_ids_from_seed(path: str | Path) -> list[str]:
    ids: set[str] = set()
    for row in load_table_seed(path):
        for key in ["table_id", "table_code", "agregado", "id"]:
            if key in row and row[key] is not None:
                ids.add(str(row[key]))
                break
    return sorted(ids, key=lambda x: int(x) if x.isdigit() else x)


def _as_list(payload: Any) -> list[Any]:
    if payload is None:
        return []
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ["items", "resultados", "periodos", "localidades"]:
            value = payload.get(key)
            if isinstance(value, list):
                return value
    return []


def _id(item: Any) -> str | None:
    if item is None:
        return None
    if isinstance(item, (str, int)):
        return str(item)
    if isinstance(item, dict):
        for key in ["id", "codigo", "cod", "periodo", "localidade"]:
            if key in item and item[key] is not None:
                return str(item[key])
    return None


def _name(item: Any) -> str:
    if isinstance(item, dict):
        for key in ["nome", "name", "descricao", "label"]:
            if key in item and item[key] is not None:
                return str(item[key])
    return ""


def _first_metadata_obj(metadata_json: Any) -> dict[str, Any]:
    if isinstance(metadata_json, list):
        for item in metadata_json:
            if isinstance(item, dict):
                return item
        return {}
    if isinstance(metadata_json, dict):
        return metadata_json
    return {}


def _extract_variables(meta: dict[str, Any]) -> tuple[list[str], dict[str, str | None]]:
    candidates = meta.get("variaveis") or meta.get("variables") or []
    variables: list[str] = []
    units: dict[str, str | None] = {}
    for item in _as_list(candidates):
        vid = _id(item)
        if vid is None:
            continue
        variables.append(vid)
        if isinstance(item, dict):
            units[vid] = (
                item.get("unidade")
                or item.get("unit")
                or item.get("unidadeMedida")
                or item.get("medida")
            )
        else:
            units[vid] = None
    return variables, units


def _extract_classifications(meta: dict[str, Any]) -> dict[str, list[str]]:
    classifications: dict[str, list[str]] = {}
    candidates = meta.get("classificacoes") or meta.get("classifications") or []
    for cls in _as_list(candidates):
        if not isinstance(cls, dict):
            continue
        cid = _id(cls)
        if cid is None:
            continue
        cats = []
        for cat in _as_list(cls.get("categorias") or cls.get("categories") or []):
            cat_id = _id(cat)
            if cat_id is not None:
                cats.append(cat_id)
        classifications[cid] = cats
    return classifications


def _extract_periods(periods_json: Any) -> list[str]:
    out: list[str] = []
    for item in _as_list(periods_json):
        pid = _id(item)
        if pid is not None:
            out.append(pid)
    return sorted(set(out), key=lambda x: x)


def _extract_localities(localities_json: Any) -> list[str]:
    out: list[str] = []
    for item in _as_list(localities_json):
        lid = _id(item)
        if lid is not None:
            out.append(lid)
    return sorted(set(out), key=lambda x: x)


def normalize_official_table_metadata(
    *,
    table_id: str,
    metadata_json: Any,
    periods_json: Any,
    localities_json: Any,
    locality_level: str,
) -> SIDRATableMetadata:
    meta = _first_metadata_obj(metadata_json)
    variables, units = _extract_variables(meta)
    classifications = _extract_classifications(meta)
    periods = _extract_periods(periods_json)
    localities = _extract_localities(localities_json)

    if not variables:
        raise ValueError(f"SIDRA metadata for table {table_id} has no variables.")
    if not periods:
        raise ValueError(f"SIDRA periods for table {table_id} are empty.")
    if not localities:
        raise ValueError(f"SIDRA localities for table {table_id} at {locality_level} are empty.")

    return SIDRATableMetadata(
        table_id=table_id,
        name=str(meta.get("nome") or meta.get("name") or f"SIDRA table {table_id}"),
        variables=variables,
        periods=periods,
        locality_levels=[locality_level],
        localities_by_level={locality_level: localities},
        classifications=classifications,
        units_by_variable=units,
    )


def fetch_official_metadata(
    *,
    table_ids: list[str],
    client: SidraClient,
    locality_level: str = "N6",
    raw_dir: str | Path = "data/metadata/sidra/raw",
) -> SIDRAMetadata:
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    tables: dict[str, SIDRATableMetadata] = {}

    for table_id in table_ids:
        meta = client.metadata(table_id)
        periods = client.periods(table_id)
        localities = client.localities(table_id, locality_level)

        (raw_dir / f"{table_id}.metadata.json").write_text(
            json.dumps(meta.payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (raw_dir / f"{table_id}.periods.json").write_text(
            json.dumps(periods.payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (raw_dir / f"{table_id}.localities.{locality_level}.json").write_text(
            json.dumps(localities.payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        if meta.status_code >= 400:
            raise RuntimeError(f"SIDRA metadata request failed for {table_id}: HTTP {meta.status_code}")
        if periods.status_code >= 400:
            raise RuntimeError(f"SIDRA periods request failed for {table_id}: HTTP {periods.status_code}")
        if localities.status_code >= 400:
            raise RuntimeError(f"SIDRA localities request failed for {table_id}: HTTP {localities.status_code}")

        table = normalize_official_table_metadata(
            table_id=table_id,
            metadata_json=meta.payload,
            periods_json=periods.payload,
            localities_json=localities.payload,
            locality_level=locality_level,
        )
        tables[table_id] = table

    return SIDRAMetadata(tables=tables)


def write_normalized_metadata_tables(
    metadata: SIDRAMetadata,
    *,
    output_dir: str | Path,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tables = []
    variables = []
    classifications = []
    categories = []
    periods = []
    localities = []

    for table in metadata.tables.values():
        tables.append({"table_id": table.table_id, "name": table.name})

        for variable in table.variables:
            variables.append(
                {
                    "table_id": table.table_id,
                    "variable_id": variable,
                    "unit": table.units_by_variable.get(variable),
                }
            )

        for period in table.periods:
            periods.append({"table_id": table.table_id, "period": period})

        for level, locs in table.localities_by_level.items():
            for loc in locs:
                localities.append(
                    {
                        "table_id": table.table_id,
                        "locality_level": level,
                        "locality_id": loc,
                    }
                )

        for cls_id, cats in table.classifications.items():
            classifications.append(
                {
                    "table_id": table.table_id,
                    "classification_id": cls_id,
                }
            )
            for cat in cats:
                categories.append(
                    {
                        "table_id": table.table_id,
                        "classification_id": cls_id,
                        "category_id": cat,
                    }
                )

    outputs = {
        "sidra_tables": output_dir / "sidra_tables.parquet",
        "sidra_variables": output_dir / "sidra_variables.parquet",
        "sidra_classifications": output_dir / "sidra_classifications.parquet",
        "sidra_categories": output_dir / "sidra_categories.parquet",
        "sidra_periods": output_dir / "sidra_periods.parquet",
        "sidra_localities": output_dir / "sidra_localities.parquet",
    }

    pl.DataFrame(tables).write_parquet(outputs["sidra_tables"])
    pl.DataFrame(variables).write_parquet(outputs["sidra_variables"])
    pl.DataFrame(classifications).write_parquet(outputs["sidra_classifications"])
    pl.DataFrame(categories).write_parquet(outputs["sidra_categories"])
    pl.DataFrame(periods).write_parquet(outputs["sidra_periods"])
    pl.DataFrame(localities).write_parquet(outputs["sidra_localities"])

    return outputs


def read_normalized_metadata_tables(input_dir: str | Path) -> SIDRAMetadata:
    input_dir = Path(input_dir)

    required = [
        "sidra_tables.parquet",
        "sidra_variables.parquet",
        "sidra_classifications.parquet",
        "sidra_categories.parquet",
        "sidra_periods.parquet",
        "sidra_localities.parquet",
    ]
    missing = [name for name in required if not (input_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing normalized SIDRA metadata tables: {missing}")

    tables_df = pl.read_parquet(input_dir / "sidra_tables.parquet")
    vars_df = pl.read_parquet(input_dir / "sidra_variables.parquet")
    cls_df = pl.read_parquet(input_dir / "sidra_classifications.parquet")
    cat_df = pl.read_parquet(input_dir / "sidra_categories.parquet")
    periods_df = pl.read_parquet(input_dir / "sidra_periods.parquet")
    locs_df = pl.read_parquet(input_dir / "sidra_localities.parquet")

    tables: dict[str, SIDRATableMetadata] = {}

    for row in tables_df.to_dicts():
        tid = str(row["table_id"])
        table_vars = vars_df.filter(pl.col("table_id") == tid).to_dicts()
        table_periods = periods_df.filter(pl.col("table_id") == tid)["period"].cast(pl.Utf8).to_list()
        table_locs = locs_df.filter(pl.col("table_id") == tid).to_dicts()

        levels: dict[str, list[str]] = {}
        for loc in table_locs:
            levels.setdefault(str(loc["locality_level"]), []).append(str(loc["locality_id"]))

        classifications: dict[str, list[str]] = {}
        for cls_row in cls_df.filter(pl.col("table_id") == tid).to_dicts():
            cid = str(cls_row["classification_id"])
            cats = (
                cat_df
                .filter((pl.col("table_id") == tid) & (pl.col("classification_id") == cid))
                ["category_id"]
                .cast(pl.Utf8)
                .to_list()
            )
            classifications[cid] = cats

        tables[tid] = SIDRATableMetadata(
            table_id=tid,
            name=str(row["name"]),
            variables=[str(x["variable_id"]) for x in table_vars],
            periods=[str(x) for x in table_periods],
            locality_levels=sorted(levels),
            localities_by_level={k: sorted(v) for k, v in levels.items()},
            classifications=classifications,
            units_by_variable={str(x["variable_id"]): x.get("unit") for x in table_vars},
        )

    return SIDRAMetadata(tables=tables)



def metadata_dir_hash(input_dir: str | Path) -> str:
    import hashlib

    input_dir = Path(input_dir)
    required = [
        "sidra_tables.parquet",
        "sidra_variables.parquet",
        "sidra_classifications.parquet",
        "sidra_categories.parquet",
        "sidra_periods.parquet",
        "sidra_localities.parquet",
    ]

    missing = [name for name in required if not (input_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Cannot hash SIDRA metadata directory; missing files: {missing}")

    h = hashlib.sha256()
    for name in required:
        path = input_dir / name
        h.update(name.encode("utf-8"))
        h.update(b"\0")
        h.update(path.read_bytes())
        h.update(b"\0")

    return h.hexdigest()
