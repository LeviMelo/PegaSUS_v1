from __future__ import annotations

import json
from functools import lru_cache
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


def _fetch_one_table_metadata(
    table_id: str, *, client: SidraClient, locality_level: str, raw_dir: Path
) -> tuple[str, SIDRATableMetadata | None, str | None]:
    """Fetch+normalize one table's metadata. Returns ``(id, metadata|None, error|None)``
    -- a per-table transient failure (after the client's own retry/backoff) is isolated
    as an error string, never a raised exception that would sink the whole batch."""
    try:
        meta = client.metadata(table_id)
        periods = client.periods(table_id)
        localities = client.localities(table_id, locality_level)
        (raw_dir / f"{table_id}.metadata.json").write_text(json.dumps(meta.payload, indent=2, ensure_ascii=False), encoding="utf-8")
        (raw_dir / f"{table_id}.periods.json").write_text(json.dumps(periods.payload, indent=2, ensure_ascii=False), encoding="utf-8")
        (raw_dir / f"{table_id}.localities.{locality_level}.json").write_text(json.dumps(localities.payload, indent=2, ensure_ascii=False), encoding="utf-8")
        for name, resp in (("metadata", meta), ("periods", periods), ("localities", localities)):
            if resp.status_code >= 400:
                return table_id, None, f"{name} HTTP {resp.status_code}"
        table = normalize_official_table_metadata(
            table_id=table_id,
            metadata_json=meta.payload,
            periods_json=periods.payload,
            localities_json=localities.payload,
            locality_level=locality_level,
        )
        return table_id, table, None
    except Exception as exc:
        return table_id, None, f"{type(exc).__name__}: {exc}"


def fetch_official_metadata(
    *,
    table_ids: list[str],
    client: SidraClient,
    locality_level: str = "N6",
    raw_dir: str | Path = "data/metadata/sidra/raw",
    required_table_ids: set[str] | frozenset[str] | None = None,
    max_workers: int = 8,
) -> SIDRAMetadata:
    """Fetch official metadata for many tables concurrently.

    Each table's transient failures are retried by the client (`retry_status_codes`
    incl. 599). A table that still fails is *isolated* (skipped, not fatal) so one
    dead/renamed table never aborts a 94-table compendium fetch -- unless it is in
    ``required_table_ids`` (e.g. the population denominator), which must succeed.
    """
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    tables: dict[str, SIDRATableMetadata] = {}
    errors: dict[str, str] = {}
    workers = max(1, min(int(max_workers), len(table_ids))) if table_ids else 1
    if workers <= 1:
        outcomes = [_fetch_one_table_metadata(t, client=client, locality_level=locality_level, raw_dir=raw_dir) for t in table_ids]
    else:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=workers) as pool:
            outcomes = list(pool.map(
                lambda t: _fetch_one_table_metadata(t, client=client, locality_level=locality_level, raw_dir=raw_dir),
                table_ids,
            ))
    for table_id, table, error in outcomes:
        if table is not None:
            tables[table_id] = table
        else:
            errors[table_id] = error or "unknown"

    required = set(required_table_ids or ())
    missing_required = sorted(required & set(errors))
    if missing_required:
        raise RuntimeError(
            f"SIDRA metadata fetch failed for required tables {missing_required}: "
            + "; ".join(f"{t}={errors[t]}" for t in missing_required)
        )
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


_METADATA_TABLE_FILES = (
    "sidra_tables.parquet",
    "sidra_variables.parquet",
    "sidra_classifications.parquet",
    "sidra_categories.parquet",
    "sidra_periods.parquet",
    "sidra_localities.parquet",
)


def read_normalized_metadata_tables(input_dir: str | Path) -> SIDRAMetadata:
    """Parse the normalized SIDRA metadata tables into a :class:`SIDRAMetadata`.

    Memoized by directory + mtime: a national run calls this ~135× (per UF × per
    acquirer) on identical inputs, and each call re-read 6 parquets and rebuilt every
    ``SIDRATableMetadata`` — a live stack showed the acquire stalled here. The mtime key
    auto-invalidates after ``write_normalized_metadata_tables``. Treat the result as
    read-only (it is shared)."""
    input_dir = Path(input_dir)
    missing = [name for name in _METADATA_TABLE_FILES if not (input_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing normalized SIDRA metadata tables: {missing}")
    mtime_ns = max((input_dir / name).stat().st_mtime_ns for name in _METADATA_TABLE_FILES)
    return _read_normalized_metadata_cached(str(input_dir), mtime_ns)


def _group_rows_by(df: pl.DataFrame, key: str) -> dict[str, list[dict[str, Any]]]:
    """One pass: partition ``df`` rows by ``str(key)``, preserving original row order within each
    group. Replaces a per-table ``.filter(pl.col(key) == tid)`` (which rescans the whole frame per
    table -> O(tables x rows)); here each frame is scanned once -> O(rows)."""
    out: dict[str, list[dict[str, Any]]] = {}
    for r in df.iter_rows(named=True):
        out.setdefault(str(r[key]), []).append(r)
    return out


@lru_cache(maxsize=8)
def _read_normalized_metadata_cached(input_dir_str: str, _mtime_ns: int) -> SIDRAMetadata:
    input_dir = Path(input_dir_str)
    tables_df = pl.read_parquet(input_dir / "sidra_tables.parquet")
    vars_df = pl.read_parquet(input_dir / "sidra_variables.parquet")
    cls_df = pl.read_parquet(input_dir / "sidra_classifications.parquet")
    cat_df = pl.read_parquet(input_dir / "sidra_categories.parquet")
    periods_df = pl.read_parquet(input_dir / "sidra_periods.parquet")
    locs_df = pl.read_parquet(input_dir / "sidra_localities.parquet")

    # Pre-group each frame by table_id once (O(rows)) instead of re-filtering per table (was
    # O(tables x rows) full-frame scans, hundreds of passes for a ~94-table compendium). Categories
    # are keyed by (table_id, classification_id). Row order within each group is preserved, so the
    # assembled SIDRATableMetadata is byte-identical to the old per-table .filter() version.
    vars_by_table = _group_rows_by(vars_df, "table_id")
    periods_by_table = _group_rows_by(periods_df, "table_id")
    locs_by_table = _group_rows_by(locs_df, "table_id")
    cls_by_table = _group_rows_by(cls_df, "table_id")
    cats_by_table_cls: dict[tuple[str, str], list[str]] = {}
    for r in cat_df.iter_rows(named=True):
        cats_by_table_cls.setdefault((str(r["table_id"]), str(r["classification_id"])), []).append(str(r["category_id"]))

    tables: dict[str, SIDRATableMetadata] = {}

    for row in tables_df.to_dicts():
        tid = str(row["table_id"])
        table_vars = vars_by_table.get(tid, [])
        table_periods = [str(x["period"]) for x in periods_by_table.get(tid, [])]
        table_locs = locs_by_table.get(tid, [])

        levels: dict[str, list[str]] = {}
        for loc in table_locs:
            levels.setdefault(str(loc["locality_level"]), []).append(str(loc["locality_id"]))

        classifications: dict[str, list[str]] = {}
        for cls_row in cls_by_table.get(tid, []):
            cid = str(cls_row["classification_id"])
            classifications[cid] = cats_by_table_cls.get((tid, cid), [])

        tables[tid] = SIDRATableMetadata(
            table_id=tid,
            name=str(row["name"]),
            variables=[str(x["variable_id"]) for x in table_vars],
            periods=table_periods,
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
