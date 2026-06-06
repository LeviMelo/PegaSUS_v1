from __future__ import annotations

import hashlib
import json
from pathlib import Path
import textwrap

import yaml

ROOT = Path.cwd()


def write(path: str, content: str) -> None:
    target = ROOT / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(textwrap.dedent(content).lstrip(), encoding="utf-8", newline="\n")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def patch_sidra_views_registry() -> None:
    reg_dir = ROOT / "config" / "registries"
    view_path = reg_dir / "sidra_views.yaml"
    manifest_path = reg_dir / "registry_manifest.yaml"

    payload = {
        "schema_version": "1.0",
        "registry_version": "v1.0",
        "created_at": "2026-06-06",
        "updated_at": "2026-06-06",
        "provenance": "Slice 2B SIDRA extraction view registry. Manual entries are extraction-policy declarations, not official metadata truth.",
        "entries": [
            {
                "id": "population_9606_total_2022_alagoas_smoke",
                "status": "stable",
                "description": "Smoke extraction view for table 9606 resident population, total sex/race/age category, 2022, Maceió N6.",
                "warnings": [
                    "view_is_extraction_policy_not_metadata_truth",
                    "total_categories_for_denominator_anchor_only",
                ],
                "table_id": "9606",
                "variables": ["93"],
                "periods": ["2022"],
                "locality_level": "N6",
                "localities": ["2704302"],
                "classifications": {
                    "86": ["95251"],
                    "2": ["6794"],
                    "287": ["100362"],
                },
                "view_mode": "flat",
                "category_policy": "total_only",
                "modeling_role": "population_denominator_anchor",
            }
        ],
    }

    with view_path.open("w", encoding="utf-8", newline="\n") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)

    if manifest_path.exists():
        manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
        registries = manifest.setdefault("registries", {})
        registries["sidra_views"] = {
            "path": "sidra_views.yaml",
            "schema_version": "1.0",
            "sha256": sha256_file(view_path),
        }
        manifest_path.write_text(
            yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
            newline="\n",
        )


def main() -> None:
    write("src/pegasus/sidra/cache.py", r'''
    from __future__ import annotations

    import json
    import time
    from dataclasses import dataclass
    from datetime import datetime, timezone
    from pathlib import Path
    from typing import Any

    from pegasus.core.hashing import content_hash


    def utc_now() -> str:
        return datetime.now(timezone.utc).isoformat()


    def stable_request_hash(*, url: str, params: dict[str, Any] | None = None) -> str:
        return content_hash(
            {
                "url": url,
                "params": params or {},
            }
        )


    def sha256_text(text: str) -> str:
        import hashlib

        return hashlib.sha256(text.encode("utf-8")).hexdigest()


    @dataclass(frozen=True)
    class CacheRead:
        hit: bool
        payload: Any | None
        sidecar: dict[str, Any] | None
        payload_path: Path
        sidecar_path: Path


    class SidraJsonCache:
        def __init__(self, root: str | Path = "data/cache/sidra") -> None:
            self.root = Path(root)
            self.root.mkdir(parents=True, exist_ok=True)

        def paths(self, *, namespace: str, request_hash: str) -> tuple[Path, Path]:
            directory = self.root / namespace
            directory.mkdir(parents=True, exist_ok=True)
            return directory / f"{request_hash}.json", directory / f"{request_hash}.sidecar.json"

        def read(
            self,
            *,
            namespace: str,
            url: str,
            params: dict[str, Any] | None = None,
        ) -> CacheRead:
            request_hash = stable_request_hash(url=url, params=params)
            payload_path, sidecar_path = self.paths(namespace=namespace, request_hash=request_hash)

            if not payload_path.exists() or not sidecar_path.exists():
                return CacheRead(False, None, None, payload_path, sidecar_path)

            try:
                return CacheRead(
                    True,
                    json.loads(payload_path.read_text(encoding="utf-8")),
                    json.loads(sidecar_path.read_text(encoding="utf-8")),
                    payload_path,
                    sidecar_path,
                )
            except json.JSONDecodeError:
                return CacheRead(False, None, None, payload_path, sidecar_path)

        def write(
            self,
            *,
            namespace: str,
            url: str,
            params: dict[str, Any] | None,
            payload: Any,
            status_code: int,
            attempt: int,
            seconds: float,
        ) -> tuple[Path, Path, dict[str, Any]]:
            request_hash = stable_request_hash(url=url, params=params)
            payload_path, sidecar_path = self.paths(namespace=namespace, request_hash=request_hash)

            raw_text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
            payload_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            sidecar = {
                "request_hash": request_hash,
                "url": url,
                "params": params or {},
                "status_code": status_code,
                "fetched_at": utc_now(),
                "attempt": attempt,
                "seconds": round(seconds, 6),
                "bytes": len(raw_text.encode("utf-8")),
                "sha256": sha256_text(raw_text),
                "payload_path": str(payload_path),
            }
            sidecar_path.write_text(
                json.dumps(sidecar, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            return payload_path, sidecar_path, sidecar
    ''')

    write("src/pegasus/sidra/api.py", r'''
    from __future__ import annotations

    import json
    import time
    import urllib.error
    import urllib.parse
    import urllib.request
    from dataclasses import dataclass, field
    from pathlib import Path
    from typing import Any, Callable

    from pegasus.core.config import load_yaml
    from pegasus.sidra.cache import SidraJsonCache
    from pegasus.sidra.schemas import SIDRAChunk


    DEFAULT_BASE_URL = "https://servicodados.ibge.gov.br/api/v3/agregados"

    Transport = Callable[[str, dict[str, Any], int], tuple[int, Any]]


    @dataclass(frozen=True)
    class SidraClientConfig:
        base_url: str = DEFAULT_BASE_URL
        view_mode: str = "flat"
        timeout_seconds: int = 60
        retry_status_codes: tuple[int, ...] = (429, 500, 502, 503, 504)
        max_retries: int = 5
        backoff_initial_seconds: float = 0.25
        backoff_max_seconds: float = 10.0

        @classmethod
        def from_mapping(cls, payload: dict[str, Any]) -> "SidraClientConfig":
            return cls(
                base_url=str(payload.get("base_url", DEFAULT_BASE_URL)),
                view_mode=str(payload.get("view_mode", "flat")),
                timeout_seconds=int(payload.get("timeout_seconds", 60)),
                retry_status_codes=tuple(int(x) for x in payload.get("retry_status_codes", [429, 500, 502, 503, 504])),
                max_retries=int(payload.get("max_retries", 5)),
                backoff_initial_seconds=float(payload.get("backoff_initial_seconds", 0.25)),
                backoff_max_seconds=float(payload.get("backoff_max_seconds", 10.0)),
            )

        @classmethod
        def from_file(cls, path: str | Path = "config/sidra.yaml") -> "SidraClientConfig":
            data = load_yaml(path)
            return cls.from_mapping(data.get("sidra", data))


    @dataclass(frozen=True)
    class SidraResponse:
        url: str
        params: dict[str, Any]
        payload: Any
        status_code: int
        from_cache: bool
        attempt: int
        seconds: float
        sidecar: dict[str, Any] | None = None


    def _default_transport(url: str, params: dict[str, Any], timeout_seconds: int) -> tuple[int, Any]:
        query = urllib.parse.urlencode(params, doseq=False)
        full_url = f"{url}?{query}" if query else url
        request = urllib.request.Request(full_url, headers={"User-Agent": "PegaSUS/0.1 SIDRA client"})

        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                status = int(response.status)
                raw = response.read().decode("utf-8")
                return status, json.loads(raw)
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="replace")
            try:
                payload: Any = json.loads(raw)
            except json.JSONDecodeError:
                payload = {"error": raw}
            return int(exc.code), payload


    def classification_expr(classifications: dict[str, list[str]] | None) -> str | None:
        if not classifications:
            return None
        parts = []
        for cls_id in sorted(classifications, key=lambda x: int(x) if str(x).isdigit() else str(x)):
            categories = classifications[cls_id]
            parts.append(f"{cls_id}[{','.join(str(x) for x in categories)}]")
        return "|".join(parts)


    def locality_expr(level: str, localities: list[str] | str) -> str:
        if isinstance(localities, str):
            return localities

        if len(localities) == 1 and localities[0] in {"BR", "N1", "N2", "N3", "N6"}:
            return localities[0]

        return f"{level}[{','.join(str(x) for x in localities)}]"


    def pipe_expr(values: list[str]) -> str:
        return "|".join(str(x) for x in values)


    class SidraClient:
        def __init__(
            self,
            *,
            config: SidraClientConfig | None = None,
            cache: SidraJsonCache | None = None,
            transport: Transport | None = None,
        ) -> None:
            self.config = config or SidraClientConfig.from_file()
            self.cache = cache or SidraJsonCache()
            self.transport = transport or _default_transport

        def endpoint(self, suffix: str = "") -> str:
            base = self.config.base_url.rstrip("/")
            suffix = suffix.strip("/")
            return base if not suffix else f"{base}/{suffix}"

        def catalog(self, **params: Any) -> SidraResponse:
            return self.get_json(self.endpoint(), params={k: v for k, v in params.items() if v is not None}, namespace="http")

        def metadata(self, table_code: str) -> SidraResponse:
            return self.get_json(self.endpoint(f"{table_code}/metadados"), namespace="http")

        def periods(self, table_code: str) -> SidraResponse:
            return self.get_json(self.endpoint(f"{table_code}/periodos"), namespace="http")

        def localities(self, table_code: str, level: str) -> SidraResponse:
            return self.get_json(self.endpoint(f"{table_code}/localidades/{level}"), namespace="http")

        def values(
            self,
            *,
            table_code: str,
            periods: list[str],
            variables: list[str],
            localities: list[str] | str,
            locality_level: str,
            classifications: dict[str, list[str]] | None = None,
            view: str | None = None,
        ) -> SidraResponse:
            params: dict[str, Any] = {
                "localidades": locality_expr(locality_level, localities),
                "view": view or self.config.view_mode,
            }
            cls_expr = classification_expr(classifications)
            if cls_expr:
                params["classificacao"] = cls_expr

            return self.get_json(
                self.endpoint(f"{table_code}/periodos/{pipe_expr(periods)}/variaveis/{pipe_expr(variables)}"),
                params=params,
                namespace="values",
            )

        def values_from_chunk(self, chunk: SIDRAChunk, *, view: str | None = None) -> SidraResponse:
            return self.values(
                table_code=chunk.table_id,
                periods=chunk.periods,
                variables=chunk.variables,
                localities=chunk.localities,
                locality_level=chunk.locality_level,
                classifications=chunk.classifications,
                view=view,
            )

        def ping(self) -> SidraResponse:
            return self.catalog(nivel="N1")

        def get_json(
            self,
            url: str,
            *,
            params: dict[str, Any] | None = None,
            namespace: str,
            use_cache: bool = True,
        ) -> SidraResponse:
            params = params or {}
            if use_cache:
                cached = self.cache.read(namespace=namespace, url=url, params=params)
                if cached.hit:
                    return SidraResponse(
                        url=url,
                        params=params,
                        payload=cached.payload,
                        status_code=int((cached.sidecar or {}).get("status_code", 200)),
                        from_cache=True,
                        attempt=int((cached.sidecar or {}).get("attempt", 0)),
                        seconds=0.0,
                        sidecar=cached.sidecar,
                    )

            last_status = 0
            last_payload: Any = None
            start_all = time.time()

            for attempt in range(1, self.config.max_retries + 2):
                started = time.time()
                try:
                    status, payload = self.transport(url, params, self.config.timeout_seconds)
                except Exception as exc:
                    status = 599
                    payload = {"error": str(exc)}

                seconds = time.time() - started
                last_status = status
                last_payload = payload

                if status < 400 or status not in self.config.retry_status_codes or attempt > self.config.max_retries:
                    _, _, sidecar = self.cache.write(
                        namespace=namespace,
                        url=url,
                        params=params,
                        payload=payload,
                        status_code=status,
                        attempt=attempt,
                        seconds=seconds,
                    )
                    return SidraResponse(
                        url=url,
                        params=params,
                        payload=payload,
                        status_code=status,
                        from_cache=False,
                        attempt=attempt,
                        seconds=round(time.time() - start_all, 6),
                        sidecar=sidecar,
                    )

                sleep_s = min(
                    self.config.backoff_max_seconds,
                    self.config.backoff_initial_seconds * (2 ** (attempt - 1)),
                )
                time.sleep(sleep_s)

            # Defensive fallback; loop always returns.
            return SidraResponse(
                url=url,
                params=params,
                payload=last_payload,
                status_code=last_status,
                from_cache=False,
                attempt=self.config.max_retries + 1,
                seconds=round(time.time() - start_all, 6),
            )
    ''')

    write("src/pegasus/sidra/metadata.py", r'''
    from __future__ import annotations

    import json
    from pathlib import Path
    from typing import Any

    import polars as pl

    from pegasus.core.hashing import content_hash
    from pegasus.sidra.api import SidraClient
    from pegasus.sidra.schemas import SIDRAMetadata, SIDRATableMetadata


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


    def fixture_sidra_metadata() -> SIDRAMetadata:
        table = SIDRATableMetadata(
            table_id="9606",
            name="Population by municipality, period and classification fixture",
            variables=["93"],
            periods=["2022", "2023"],
            locality_levels=["N6"],
            localities_by_level={
                "N6": ["270030", "270430", "270770"],
            },
            classifications={
                "2": ["0", "1", "2"],
                "58": ["0", "1", "2", "3", "4", "5", "9"],
                "287": ["0", "93070", "93084", "100000"],
            },
            units_by_variable={"93": "persons"},
        )
        return SIDRAMetadata(tables={"9606": table})


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
    ''')

    write("src/pegasus/sidra/registry.py", r'''
    from __future__ import annotations

    from pathlib import Path
    from typing import Any

    import yaml

    from pegasus.sidra.schemas import SIDRARequest


    def load_sidra_view_registry(path: str | Path = "config/registries/sidra_views.yaml") -> dict[str, Any]:
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        entries = payload.get("entries", [])
        return {str(entry["id"]): entry for entry in entries if "id" in entry}


    def request_from_view(view_id: str, *, registry_path: str | Path = "config/registries/sidra_views.yaml") -> SIDRARequest:
        registry = load_sidra_view_registry(registry_path)
        if view_id not in registry:
            raise KeyError(f"SIDRA view not found in registry: {view_id}")

        entry = registry[view_id]
        return SIDRARequest(
            table_id=str(entry["table_id"]),
            variables=[str(x) for x in entry.get("variables", [])],
            periods=[str(x) for x in entry.get("periods", [])],
            locality_level=str(entry["locality_level"]),
            localities=[str(x) for x in entry.get("localities", [])],
            classifications={str(k): [str(x) for x in v] for k, v in entry.get("classifications", {}).items()},
        )
    ''')

    write("src/pegasus/sidra/normalize.py", r'''
    from __future__ import annotations

    import json
    import re
    from typing import Any

    from pegasus.sidra.facts import normalize_flat_records_to_facts
    from pegasus.sidra.schemas import SIDRAFactRow


    def _is_header_row(row: dict[str, Any]) -> bool:
        if row.get("header_row") is True:
            return True
        value = str(row.get("V", row.get("value", ""))).strip().casefold()
        # In flat SIDRA responses the first record can contain labels/descriptors.
        descriptor_tokens = {"valor", "value", "v"}
        return value in descriptor_tokens


    def _classification_pairs_from_flat(row: dict[str, Any]) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
        classification_pairs: list[tuple[str, str]] = []
        category_pairs: list[tuple[str, str]] = []

        # Support explicit fixture-like tuples first.
        if "classification_tuple" in row or "category_tuple" in row:
            raw_cls = row.get("classification_tuple") or []
            raw_cat = row.get("category_tuple") or []
            return (
                [tuple(map(str, x)) for x in raw_cls],
                [tuple(map(str, x)) for x in raw_cat],
            )

        # Generic flat parser: D3C/D3N, D4C/D4N... are usually classification/category
        # dimensions after period/locality/variable descriptors. We keep the pair as
        # source-coded because official column ordering varies by table.
        for key in sorted(row.keys()):
            m = re.fullmatch(r"D(\d+)C", key)
            if not m:
                continue
            idx = int(m.group(1))
            if idx <= 2:
                continue
            code = row.get(key)
            label = row.get(f"D{idx}N")
            if code is None:
                continue
            classification_pairs.append((f"D{idx}", str(label or "")))
            category_pairs.append((f"D{idx}", str(code)))

        return classification_pairs, category_pairs


    def flat_response_to_records(payload: Any, *, table_id: str, chunk_request: dict[str, Any]) -> list[dict[str, Any]]:
        if isinstance(payload, dict):
            rows = payload.get("data") or payload.get("resultados") or payload.get("values") or []
        elif isinstance(payload, list):
            rows = payload
        else:
            rows = []

        records: list[dict[str, Any]] = []
        for i, row in enumerate(rows):
            if not isinstance(row, dict):
                continue

            if i == 0 and _is_header_row(row):
                continue
            if _is_header_row(row):
                continue

            cls_pairs, cat_pairs = _classification_pairs_from_flat(row)

            record = {
                "table_id": str(row.get("table_id") or row.get("agregado") or table_id),
                "variable_id": str(row.get("variable_id") or row.get("variable") or row.get("MC") or row.get("D1C") or (chunk_request.get("variables") or [""])[0]),
                "period": str(row.get("period") or row.get("periodo") or row.get("D2C") or (chunk_request.get("periods") or [""])[0]),
                "locality_level": str(row.get("locality_level") or row.get("NC") or chunk_request.get("locality_level") or ""),
                "locality_id": str(row.get("locality_id") or row.get("localidade") or row.get("NN") or ""),
                "classification_tuple": cls_pairs,
                "category_tuple": cat_pairs,
                "value": row.get("value", row.get("V")),
                "header_row": False,
            }

            # Normalized fixture-style records should preserve their explicit fields.
            for key in [
                "variable_id",
                "period",
                "locality_level",
                "locality_id",
                "classification_tuple",
                "category_tuple",
                "value",
            ]:
                if key in row:
                    record[key] = row[key]

            records.append(record)

        return records


    def normalize_sidra_payload_to_facts(
        payload: Any,
        *,
        table_id: str,
        request_hash: str,
        metadata_hash: str,
        chunk_request: dict[str, Any],
        unit_by_variable: dict[str, str | None] | None = None,
        fetched_at: str | None = None,
    ) -> list[SIDRAFactRow]:
        records = flat_response_to_records(payload, table_id=table_id, chunk_request=chunk_request)
        return normalize_flat_records_to_facts(
            records,
            table_id=table_id,
            request_hash=request_hash,
            metadata_hash=metadata_hash,
            unit_by_variable=unit_by_variable,
            fetched_at=fetched_at,
        )
    ''')

    write("src/pegasus/sidra/extract.py", r'''
    from __future__ import annotations

    import json
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from pathlib import Path
    from typing import Any

    from pydantic import BaseModel, ConfigDict

    from pegasus.core.hashing import content_hash
    from pegasus.sidra.api import SidraClient
    from pegasus.sidra.facts import write_facts_parquet
    from pegasus.sidra.normalize import normalize_sidra_payload_to_facts
    from pegasus.sidra.schemas import SIDRAChunk


    class SIDRAChunkResult(BaseModel):
        model_config = ConfigDict(extra="forbid")

        chunk_id: str
        table_id: str
        status: str
        status_code: int
        raw_path: str | None
        facts_path: str | None
        row_count: int
        request_hash: str
        error_message: str | None = None


    def write_chunk_plan(chunks: list[SIDRAChunk], *, output_path: str | Path) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps([c.model_dump(mode="json") for c in chunks], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return output_path


    def read_chunk_plan(path: str | Path) -> list[SIDRAChunk]:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("SIDRA chunk plan must be a JSON list.")
        return [SIDRAChunk.model_validate(x) for x in payload]


    def _raw_chunk_path(chunk: SIDRAChunk, *, raw_dir: str | Path) -> Path:
        return Path(raw_dir) / f"{chunk.chunk_id}.json"


    def _facts_chunk_path(chunk: SIDRAChunk, *, facts_root: str | Path) -> Path:
        return Path(facts_root) / chunk.table_id / f"{chunk.chunk_id}.parquet"


    def extract_one_chunk(
        chunk: SIDRAChunk,
        *,
        client: SidraClient,
        raw_dir: str | Path = "data/raw/sidra/chunks",
        facts_root: str | Path = "data/processed/sidra/facts",
        metadata_hash: str = "metadata_unset",
        unit_by_variable: dict[str, str | None] | None = None,
    ) -> SIDRAChunkResult:
        raw_path = _raw_chunk_path(chunk, raw_dir=raw_dir)
        facts_path = _facts_chunk_path(chunk, facts_root=facts_root)
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        facts_path.parent.mkdir(parents=True, exist_ok=True)

        response = client.values_from_chunk(chunk)
        request_hash = content_hash(chunk.model_dump(mode="json"))

        raw_path.write_text(
            json.dumps(
                {
                    "chunk": chunk.model_dump(mode="json"),
                    "status_code": response.status_code,
                    "from_cache": response.from_cache,
                    "sidecar": response.sidecar,
                    "payload": response.payload,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        if response.status_code >= 400:
            return SIDRAChunkResult(
                chunk_id=chunk.chunk_id,
                table_id=chunk.table_id,
                status="failed",
                status_code=response.status_code,
                raw_path=str(raw_path),
                facts_path=None,
                row_count=0,
                request_hash=request_hash,
                error_message=f"SIDRA HTTP status {response.status_code}",
            )

        facts = normalize_sidra_payload_to_facts(
            response.payload,
            table_id=chunk.table_id,
            request_hash=request_hash,
            metadata_hash=metadata_hash,
            chunk_request=chunk.request_params,
            unit_by_variable=unit_by_variable,
            fetched_at=(response.sidecar or {}).get("fetched_at"),
        )
        write_facts_parquet(facts, output_path=facts_path)

        return SIDRAChunkResult(
            chunk_id=chunk.chunk_id,
            table_id=chunk.table_id,
            status="success",
            status_code=response.status_code,
            raw_path=str(raw_path),
            facts_path=str(facts_path),
            row_count=len(facts),
            request_hash=request_hash,
        )


    def extract_chunk_plan(
        chunks: list[SIDRAChunk],
        *,
        client: SidraClient,
        concurrency: int = 4,
        raw_dir: str | Path = "data/raw/sidra/chunks",
        facts_root: str | Path = "data/processed/sidra/facts",
        metadata_hash: str = "metadata_unset",
        unit_by_variable: dict[str, str | None] | None = None,
    ) -> list[SIDRAChunkResult]:
        if concurrency <= 1:
            return [
                extract_one_chunk(
                    chunk,
                    client=client,
                    raw_dir=raw_dir,
                    facts_root=facts_root,
                    metadata_hash=metadata_hash,
                    unit_by_variable=unit_by_variable,
                )
                for chunk in chunks
            ]

        results: list[SIDRAChunkResult] = []
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {
                executor.submit(
                    extract_one_chunk,
                    chunk,
                    client=client,
                    raw_dir=raw_dir,
                    facts_root=facts_root,
                    metadata_hash=metadata_hash,
                    unit_by_variable=unit_by_variable,
                ): chunk
                for chunk in chunks
            }
            for future in as_completed(futures):
                results.append(future.result())

        return sorted(results, key=lambda r: r.chunk_id)


    def write_extraction_log(results: list[SIDRAChunkResult], *, output_path: str | Path) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps([r.model_dump(mode="json") for r in results], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return output_path
    ''')

    write("src/pegasus/cli.py", r'''
    from __future__ import annotations

    import importlib.util
    import json
    import shutil
    import sys
    from pathlib import Path

    import typer
    from rich import print

    from pegasus.core.config import load_yaml, validate_config_tree
    from pegasus.core.paths import ensure_data_lake
    from pegasus.datasus.cache import DatasusCache
    from pegasus.datasus.manifests import (
        build_datasus_manifests,
        load_datasus_config,
        read_request_manifest,
        write_request_manifest,
    )
    from pegasus.datasus.normalize import normalize_sim_do_events
    from pegasus.datasus.profile import profile_table
    from pegasus.datasus.schema_compare import compare_profiles
    from pegasus.datasus.subprocess import DatasusConfig, fetch_datasus_chunk
    from pegasus.output.bundle import create_empty_output_bundle
    from pegasus.output.validate import validate_output_bundle
    from pegasus.registries.validators import validate_registry_tree
    from pegasus.sidra.api import SidraClient, SidraClientConfig
    from pegasus.sidra.extract import extract_chunk_plan, read_chunk_plan, write_chunk_plan, write_extraction_log
    from pegasus.sidra.facts import normalize_fixture_json_to_facts
    from pegasus.sidra.metadata import (
        fetch_official_metadata,
        fixture_sidra_metadata,
        read_normalized_metadata_tables,
        table_ids_from_seed,
        write_normalized_metadata_tables,
    )
    from pegasus.sidra.plan import plan_sidra_chunks
    from pegasus.sidra.registry import request_from_view
    from pegasus.sidra.schemas import SIDRARequest
    from pegasus.workflows.build_efg import build_sim_fixture_efg_run

    app = typer.Typer(no_args_is_help=True)
    registries_app = typer.Typer(no_args_is_help=True)
    sidra_app = typer.Typer(no_args_is_help=True)
    datasus_app = typer.Typer(no_args_is_help=True)
    efg_app = typer.Typer(no_args_is_help=True)

    app.add_typer(registries_app, name="registries")
    app.add_typer(sidra_app, name="sidra")
    app.add_typer(datasus_app, name="datasus")
    app.add_typer(efg_app, name="efg")


    def _fail(errors: list[str]) -> None:
        for error in errors:
            print(f"[red]ERROR[/red] {error}")
        raise typer.Exit(1)


    def _sidra_runtime_config() -> dict:
        data = load_yaml("config/sidra.yaml")
        return data.get("sidra", data)


    @app.command()
    def init() -> None:
        ensure_data_lake(".")
        run_dir = create_empty_output_bundle(Path("data/runs/slice0_empty"))
        print(f"[green]initialized[/green] data lake and scaffold run: {run_dir}")


    @app.command("validate-config")
    def validate_config() -> None:
        errors = validate_config_tree(".")
        if errors:
            _fail(errors)
        print("[green]config valid[/green]")


    @registries_app.command("validate")
    def validate_registries() -> None:
        errors = validate_registry_tree("config/registries")
        if errors:
            _fail(errors)
        print("[green]registries valid[/green]")


    @app.command("validate-run")
    def validate_run(run: Path = typer.Option(..., "--run")) -> None:
        result = validate_output_bundle(run_dir=str(run))
        if not result.ok:
            _fail(result.errors)
        print("[green]run bundle valid[/green]")


    @app.command()
    def doctor() -> None:
        checks: dict[str, str] = {}
        checks["python"] = sys.version.split()[0]

        for mod in ["duckdb", "polars", "pyarrow", "pydantic", "typer", "yaml"]:
            checks[mod] = "ok" if importlib.util.find_spec(mod) else "missing"

        torch_spec = importlib.util.find_spec("torch")
        if torch_spec:
            import torch

            checks["torch"] = getattr(torch, "__version__", "ok")
            checks["torch_cuda_available"] = str(torch.cuda.is_available())
            checks["cuda_device_name"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
        else:
            checks["torch"] = "missing"
            checks["torch_cuda_available"] = "False"
            checks["cuda_device_name"] = "none"

        checks["Rscript"] = shutil.which("Rscript") or "missing"
        checks["microdatasus"] = "unchecked_without_rscript" if checks["Rscript"] == "missing" else "check_with_R_script"
        checks["read.dbc"] = "unchecked_without_rscript" if checks["Rscript"] == "missing" else "check_with_R_script"

        checks["data_write"] = "ok"
        try:
            ensure_data_lake(".")
        except Exception as exc:
            checks["data_write"] = f"failed: {exc}"

        reg_errors = validate_registry_tree("config/registries")
        checks["registry_schema"] = "ok" if not reg_errors else f"{len(reg_errors)} errors"

        try:
            client = SidraClient(
                config=SidraClientConfig.from_mapping(
                    {
                        **_sidra_runtime_config(),
                        "timeout_seconds": 5,
                        "max_retries": 0,
                    }
                )
            )
            response = client.ping()
            checks["sidra_network"] = "ok" if response.status_code < 400 else f"HTTP {response.status_code}"
        except Exception as exc:
            checks["sidra_network"] = f"failed: {exc}"

        for key, value in checks.items():
            color = "green" if value not in {"missing", "False"} and not str(value).startswith("failed") else "yellow"
            print(f"[{color}]{key}[/] {value}")

        print("[yellow]doctor is light: it checks connectivity but does not run heavy ingestion.[/yellow]")


    @datasus_app.command("ingest")
    def datasus_ingest(
        system: str = typer.Option(..., "--system"),
        uf: str = typer.Option(..., "--uf"),
        years: str = typer.Option(..., "--years"),
        dry_run: bool = typer.Option(False, "--dry-run", help="Plan and persist manifests without invoking R."),
    ) -> None:
        cfg_payload = load_datasus_config()
        cfg = DatasusConfig.from_mapping(cfg_payload)
        manifests = build_datasus_manifests(system=system, uf=uf, years=years, config=cfg_payload)

        cache = DatasusCache()
        blocked = False
        failed = False

        for manifest in manifests:
            planned_path = write_request_manifest(manifest)
            print(f"[cyan]planned[/cyan] {manifest.system} {manifest.uf} {manifest.year_start}: {planned_path}")

            if dry_run:
                continue

            executed = fetch_datasus_chunk(
                manifest,
                config=cfg,
                cache=cache,
                timeout_seconds=cfg.r_timeout_seconds,
                heartbeat_timeout_seconds=cfg.heartbeat_timeout_seconds,
            )
            executed_path = write_request_manifest(executed)

            if executed.status == "success":
                print(f"[green]success[/green] {executed.system} {executed.uf} {executed.year_start}: {executed_path}")
            elif executed.status == "blocked":
                blocked = True
                print(f"[yellow]blocked[/yellow] {executed.system} {executed.uf} {executed.year_start}: {executed.error_message}")
            else:
                failed = True
                print(f"[red]{executed.status}[/red] {executed.system} {executed.uf} {executed.year_start}: {executed.error_message}")

        if failed:
            raise typer.Exit(1)
        if blocked:
            raise typer.Exit(2)


    @datasus_app.command("profile")
    def datasus_profile(manifest: Path = typer.Option(..., "--manifest")) -> None:
        request = read_request_manifest(manifest)

        raw_path = Path(request.raw_path)
        processed_path = Path(request.processed_path)

        if not raw_path.exists() or not processed_path.exists():
            print("[yellow]blocked[/yellow] raw/processed artifacts are missing; cannot profile this manifest.")
            raise typer.Exit(2)

        raw_profile_path = Path("data/metadata/datasus/profiles") / request.system / request.request_hash / "raw_profile.json"
        processed_profile_path = Path("data/metadata/datasus/profiles") / request.system / request.request_hash / "processed_profile.json"
        compare_path = Path("data/metadata/datasus/schema_compare") / request.system / request.request_hash / "schema_compare.json"

        raw_profile = profile_table(raw_path, output_path=raw_profile_path)
        processed_profile = profile_table(processed_path, output_path=processed_profile_path)
        compare_profiles(raw_profile, processed_profile, output_path=compare_path)

        print(f"[green]raw profile[/green] {raw_profile_path}")
        print(f"[green]processed profile[/green] {processed_profile_path}")
        print(f"[green]schema comparison[/green] {compare_path}")


    @datasus_app.command("normalize-sim")
    def datasus_normalize_sim(
        input_path: Path = typer.Option(..., "--input"),
        output_path: Path = typer.Option(..., "--output"),
        source_manifest_hash: str = typer.Option("fixture", "--source-manifest-hash"),
    ) -> None:
        result = normalize_sim_do_events(
            input_path=input_path,
            output_path=output_path,
            source_manifest_hash=source_manifest_hash,
        )
        print(f"[green]sim normalized[/green] rows={result['row_count']} output={result['output_path']}")


    @efg_app.command("build-sim-fixture")
    def efg_build_sim_fixture(
        sim_events: Path = typer.Option(..., "--sim-events"),
        run_dir: Path = typer.Option(..., "--run-dir"),
    ) -> None:
        output = build_sim_fixture_efg_run(
            sim_events_path=sim_events,
            run_dir=run_dir,
        )
        result = validate_output_bundle(run_dir=str(output))
        if not result.ok:
            _fail(result.errors)
        print(f"[green]sim fixture EFG bundle valid[/green] {output}")


    @sidra_app.command("metadata-fixture")
    def sidra_metadata_fixture(
        output_dir: Path = typer.Option(Path("data/metadata/sidra/normalized"), "--output-dir"),
    ) -> None:
        metadata = fixture_sidra_metadata()
        outputs = write_normalized_metadata_tables(metadata, output_dir=output_dir)
        for name, path in outputs.items():
            print(f"[green]{name}[/green] {path}")


    @sidra_app.command("plan-fixture")
    def sidra_plan_fixture(
        output: Path = typer.Option(Path("data/manifests/sidra/fixture_plan.json"), "--output"),
        max_cells: int = typer.Option(49900, "--max-cells"),
    ) -> None:
        metadata = fixture_sidra_metadata()
        table = metadata.tables["9606"]
        request = SIDRARequest(
            table_id="9606",
            variables=table.variables,
            periods=table.periods,
            locality_level="N6",
            localities=table.localities_by_level["N6"],
            classifications=table.classifications,
        )
        chunks = plan_sidra_chunks(request, metadata, max_cells_per_request=max_cells)
        write_chunk_plan(chunks, output_path=output)
        print(f"[green]planned[/green] chunks={len(chunks)} output={output}")


    @sidra_app.command("normalize-fixture")
    def sidra_normalize_fixture(
        input_path: Path = typer.Option(..., "--input"),
        output_path: Path = typer.Option(Path("data/processed/sidra/facts/9606/fixture.parquet"), "--output"),
    ) -> None:
        output = normalize_fixture_json_to_facts(
            input_path=input_path,
            output_path=output_path,
            table_id="9606",
            unit_by_variable={"93": "persons"},
        )
        print(f"[green]sidra facts normalized[/green] {output}")


    @sidra_app.command("metadata")
    def sidra_metadata(
        tables: Path = typer.Option(..., "--tables"),
        level: str = typer.Option("N6", "--level"),
        output_dir: Path = typer.Option(Path("data/metadata/sidra/normalized"), "--output-dir"),
        raw_dir: Path = typer.Option(Path("data/metadata/sidra/raw"), "--raw-dir"),
    ) -> None:
        table_ids = table_ids_from_seed(tables)
        if not table_ids:
            print("[red]ERROR[/red] no SIDRA table IDs found in seed.")
            raise typer.Exit(1)

        client = SidraClient()
        metadata = fetch_official_metadata(
            table_ids=table_ids,
            client=client,
            locality_level=level,
            raw_dir=raw_dir,
        )
        outputs = write_normalized_metadata_tables(metadata, output_dir=output_dir)

        print(f"[green]metadata rebuilt[/green] tables={len(table_ids)}")
        for name, path in outputs.items():
            print(f"[green]{name}[/green] {path}")


    @sidra_app.command("plan")
    def sidra_plan(
        view: str = typer.Option(..., "--view"),
        metadata_dir: Path = typer.Option(Path("data/metadata/sidra/normalized"), "--metadata-dir"),
        output: Path | None = typer.Option(None, "--output"),
    ) -> None:
        sidra_cfg = _sidra_runtime_config()
        metadata = read_normalized_metadata_tables(metadata_dir)
        request = request_from_view(view)
        chunks = plan_sidra_chunks(
            request,
            metadata,
            max_cells_per_request=int(sidra_cfg.get("max_cells_per_request", 49900)),
        )
        output = output or Path("data/manifests/sidra") / f"{view}.json"
        write_chunk_plan(chunks, output_path=output)

        print(f"[green]planned[/green] view={view} chunks={len(chunks)} output={output}")


    @sidra_app.command("extract")
    def sidra_extract(
        plan: Path = typer.Option(..., "--plan"),
        dry_run: bool = typer.Option(False, "--dry-run"),
        concurrency: int | None = typer.Option(None, "--concurrency"),
        log_path: Path | None = typer.Option(None, "--log"),
    ) -> None:
        sidra_cfg = _sidra_runtime_config()
        chunks = read_chunk_plan(plan)

        if dry_run:
            total = sum(c.estimated_cells for c in chunks)
            print(f"[cyan]dry-run[/cyan] chunks={len(chunks)} estimated_cells={total}")
            return

        client = SidraClient()
        results = extract_chunk_plan(
            chunks,
            client=client,
            concurrency=concurrency or int(sidra_cfg.get("concurrency", 4)),
        )
        log_path = log_path or Path("data/diagnostics/sidra") / f"{plan.stem}.extraction_log.json"
        write_extraction_log(results, output_path=log_path)

        failures = [r for r in results if r.status != "success"]
        print(f"[green]extract complete[/green] chunks={len(results)} failures={len(failures)} log={log_path}")
        if failures:
            raise typer.Exit(1)


    @app.command()
    def compile(intent: Path = typer.Option(..., "--intent")) -> None:
        print(f"[yellow]blocked[/yellow] compile workflow requires later slices. Intent: {intent}")
        raise typer.Exit(2)
    ''')

    write("tests/unit/test_sidra_cache_api.py", r'''
    from pathlib import Path

    from pegasus.sidra.api import SidraClient, SidraClientConfig, classification_expr, locality_expr
    from pegasus.sidra.cache import SidraJsonCache


    def test_sidra_expression_builders():
        assert classification_expr({"86": ["2776", "2777"], "2": ["4", "5"]}) == "2[4,5]|86[2776,2777]"
        assert locality_expr("N6", ["2704302", "2700300"]) == "N6[2704302,2700300]"


    def test_sidra_client_cache_hit(tmp_path: Path):
        calls = {"n": 0}

        def transport(url, params, timeout):
            calls["n"] += 1
            return 200, [{"ok": True, "params": params}]

        client = SidraClient(
            config=SidraClientConfig(max_retries=0),
            cache=SidraJsonCache(tmp_path),
            transport=transport,
        )

        first = client.values(
            table_code="9606",
            periods=["2022"],
            variables=["93"],
            localities=["2704302"],
            locality_level="N6",
            classifications={"86": ["95251"]},
        )
        second = client.values(
            table_code="9606",
            periods=["2022"],
            variables=["93"],
            localities=["2704302"],
            locality_level="N6",
            classifications={"86": ["95251"]},
        )

        assert first.from_cache is False
        assert second.from_cache is True
        assert calls["n"] == 1


    def test_sidra_client_retries_transient_status(tmp_path: Path):
        calls = {"n": 0}

        def transport(url, params, timeout):
            calls["n"] += 1
            if calls["n"] == 1:
                return 500, {"error": "server"}
            return 200, [{"ok": True}]

        client = SidraClient(
            config=SidraClientConfig(max_retries=1, backoff_initial_seconds=0, backoff_max_seconds=0),
            cache=SidraJsonCache(tmp_path),
            transport=transport,
        )

        response = client.metadata("9606")
        assert response.status_code == 200
        assert response.attempt == 2
    ''')

    write("tests/unit/test_sidra_metadata_official_normalizer.py", r'''
    from pathlib import Path

    import polars as pl

    from pegasus.sidra.metadata import (
        normalize_official_table_metadata,
        read_normalized_metadata_tables,
        write_normalized_metadata_tables,
    )


    def test_official_metadata_normalizer_shape(tmp_path: Path):
        metadata_json = {
            "id": "9606",
            "nome": "População residente por cor, sexo e idade",
            "variaveis": [{"id": "93", "nome": "População residente", "unidade": "Pessoas"}],
            "classificacoes": [
                {"id": "86", "nome": "Cor ou raça", "categorias": [{"id": "95251", "nome": "Total"}, {"id": "2776", "nome": "Branca"}]},
                {"id": "2", "nome": "Sexo", "categorias": [{"id": "6794", "nome": "Total"}, {"id": "4", "nome": "Homens"}]},
            ],
        }
        periods_json = [{"id": "2022", "nome": "2022"}]
        localities_json = [{"id": "2704302", "nome": "Maceió"}]

        table = normalize_official_table_metadata(
            table_id="9606",
            metadata_json=metadata_json,
            periods_json=periods_json,
            localities_json=localities_json,
            locality_level="N6",
        )

        assert table.table_id == "9606"
        assert table.variables == ["93"]
        assert table.classifications["86"] == ["95251", "2776"]
        assert table.localities_by_level["N6"] == ["2704302"]


    def test_metadata_parquet_roundtrip(tmp_path: Path):
        metadata_json = {
            "id": "9606",
            "nome": "População residente por cor, sexo e idade",
            "variaveis": [{"id": "93", "nome": "População residente", "unidade": "Pessoas"}],
            "classificacoes": [{"id": "86", "nome": "Cor ou raça", "categorias": [{"id": "95251", "nome": "Total"}]}],
        }
        table = normalize_official_table_metadata(
            table_id="9606",
            metadata_json=metadata_json,
            periods_json=[{"id": "2022"}],
            localities_json=[{"id": "2704302"}],
            locality_level="N6",
        )
        from pegasus.sidra.schemas import SIDRAMetadata
        metadata = SIDRAMetadata(tables={"9606": table})

        write_normalized_metadata_tables(metadata, output_dir=tmp_path)
        loaded = read_normalized_metadata_tables(tmp_path)

        assert loaded.tables["9606"].variables == ["93"]
        assert loaded.tables["9606"].localities_by_level["N6"] == ["2704302"]
    ''')

    write("tests/unit/test_sidra_extract.py", r'''
    from pathlib import Path

    import polars as pl

    from pegasus.sidra.api import SidraClient, SidraClientConfig
    from pegasus.sidra.cache import SidraJsonCache
    from pegasus.sidra.extract import extract_one_chunk
    from pegasus.sidra.schemas import SIDRAChunk


    def test_extract_one_chunk_writes_raw_and_facts(tmp_path: Path):
        payload = [
            {"V": "Valor", "D1C": "Período"},
            {
                "table_id": "9606",
                "variable_id": "93",
                "period": "2022",
                "locality_level": "N6",
                "locality_id": "2704302",
                "classification_tuple": [["86", "Cor ou raça"]],
                "category_tuple": [["86", "95251"]],
                "value": "1025360",
            },
        ]

        def transport(url, params, timeout):
            return 200, payload

        client = SidraClient(
            config=SidraClientConfig(max_retries=0),
            cache=SidraJsonCache(tmp_path / "cache"),
            transport=transport,
        )

        chunk = SIDRAChunk(
            chunk_id="chunk123",
            table_id="9606",
            variables=["93"],
            periods=["2022"],
            locality_level="N6",
            localities=["2704302"],
            classifications={"86": ["95251"]},
            estimated_cells=1,
            request_url="unused",
            request_params={
                "table_id": "9606",
                "variables": ["93"],
                "periods": ["2022"],
                "locality_level": "N6",
                "localities": ["2704302"],
                "classifications": {"86": ["95251"]},
            },
        )

        result = extract_one_chunk(
            chunk,
            client=client,
            raw_dir=tmp_path / "raw",
            facts_root=tmp_path / "facts",
            metadata_hash="metadata_hash",
            unit_by_variable={"93": "persons"},
        )

        assert result.status == "success"
        assert result.row_count == 1
        assert Path(result.raw_path).exists()
        assert Path(result.facts_path).exists()

        df = pl.read_parquet(result.facts_path)
        assert df.height == 1
        assert df.row(0, named=True)["value_numeric"] == 1025360.0
    ''')

    write("tests/unit/test_sidra_registry_view.py", r'''
    from pegasus.sidra.registry import request_from_view


    def test_sidra_view_registry_population_smoke_view():
        request = request_from_view("population_9606_total_2022_alagoas_smoke")
        assert request.table_id == "9606"
        assert request.variables == ["93"]
        assert request.periods == ["2022"]
        assert request.locality_level == "N6"
        assert request.localities == ["2704302"]
        assert request.classifications["86"] == ["95251"]
    ''')

    patch_sidra_views_registry()
    print("Applied Slice 2B: SIDRA HTTP client, cache, official metadata rebuild, extraction executor, real CLI commands, and tests.")


if __name__ == "__main__":
    main()