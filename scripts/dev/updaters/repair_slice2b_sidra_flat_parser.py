from __future__ import annotations

from pathlib import Path
import textwrap

ROOT = Path.cwd()


def write(path: str, content: str) -> None:
    target = ROOT / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(textwrap.dedent(content).lstrip(), encoding="utf-8", newline="\n")


def main() -> None:
    write("src/pegasus/sidra/facts.py", r'''
    from __future__ import annotations

    import json
    from datetime import datetime, timezone
    from pathlib import Path
    from typing import Any

    import polars as pl

    from pegasus.core.hashing import content_hash
    from pegasus.sidra.schemas import SIDRAFactRow


    def utc_now() -> str:
        return datetime.now(timezone.utc).isoformat()


    def classify_sidra_value(raw: Any) -> tuple[float | None, str]:
        if raw is None:
            return None, "blank"

        text = str(raw).strip()
        if text == "":
            return None, "blank"

        if text in {"-", "—", "–"}:
            return None, "dash_zero_or_nil"

        lowered = text.casefold()
        if lowered in {"x", "...", "na", "n/a"}:
            return None, "not_available"

        if lowered in {"..", "…"}:
            return None, "suppressed_or_unidentified"

        normalized = text.replace(".", "").replace(",", ".") if "," in text else text

        try:
            return float(normalized), "numeric"
        except ValueError:
            return None, "non_numeric_symbol"


    def _tuple_from_pairs(value: Any) -> tuple[tuple[str, str], ...]:
        if value is None:
            return tuple()
        if isinstance(value, tuple):
            return tuple(tuple(map(str, x)) for x in value)
        if isinstance(value, list):
            return tuple(tuple(map(str, x)) for x in value)
        if isinstance(value, dict):
            return tuple((str(k), str(v)) for k, v in sorted(value.items()))
        return tuple()


    def normalize_flat_records_to_facts(
        records: list[dict[str, Any]],
        *,
        table_id: str,
        request_hash: str,
        metadata_hash: str,
        unit_by_variable: dict[str, str | None] | None = None,
        fetched_at: str | None = None,
    ) -> list[SIDRAFactRow]:
        unit_by_variable = unit_by_variable or {}
        fetched_at = fetched_at or utc_now()
        facts: list[SIDRAFactRow] = []

        for record in records:
            if record.get("header_row") is True:
                status = "header_row"
                value_numeric = None
            else:
                value_numeric, status = classify_sidra_value(record.get("value"))

            variable_id = str(record.get("variable_id") or record.get("variable") or "")
            if not variable_id:
                continue

            unit = unit_by_variable.get(variable_id)
            if unit is None and record.get("unit") is not None:
                unit = str(record.get("unit"))

            facts.append(
                SIDRAFactRow(
                    table_id=str(record.get("table_id") or table_id),
                    variable_id=variable_id,
                    period=str(record.get("period") or ""),
                    locality_level=str(record.get("locality_level") or ""),
                    locality_id=str(record.get("locality_id") or ""),
                    classification_tuple=_tuple_from_pairs(record.get("classification_tuple")),
                    category_tuple=_tuple_from_pairs(record.get("category_tuple")),
                    value_raw=None if record.get("value") is None else str(record.get("value")),
                    value_numeric=value_numeric,
                    value_status=status,
                    unit=unit,
                    request_hash=request_hash,
                    metadata_hash=metadata_hash,
                    fetched_at=fetched_at,
                )
            )

        return facts


    def facts_to_frame(facts: list[SIDRAFactRow]) -> pl.DataFrame:
        rows = []
        for fact in facts:
            row = fact.model_dump(mode="json")
            row["classification_tuple"] = json.dumps(row["classification_tuple"], ensure_ascii=False)
            row["category_tuple"] = json.dumps(row["category_tuple"], ensure_ascii=False)
            rows.append(row)

        if not rows:
            return pl.DataFrame(
                schema={
                    "table_id": pl.Utf8,
                    "variable_id": pl.Utf8,
                    "period": pl.Utf8,
                    "locality_level": pl.Utf8,
                    "locality_id": pl.Utf8,
                    "classification_tuple": pl.Utf8,
                    "category_tuple": pl.Utf8,
                    "value_raw": pl.Utf8,
                    "value_numeric": pl.Float64,
                    "value_status": pl.Utf8,
                    "unit": pl.Utf8,
                    "request_hash": pl.Utf8,
                    "metadata_hash": pl.Utf8,
                    "fetched_at": pl.Utf8,
                }
            )
        return pl.DataFrame(rows)


    def write_facts_parquet(
        facts: list[SIDRAFactRow],
        *,
        output_path: str | Path,
    ) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        facts_to_frame(facts).write_parquet(output_path)
        return output_path


    def normalize_fixture_json_to_facts(
        *,
        input_path: str | Path,
        output_path: str | Path,
        table_id: str,
        unit_by_variable: dict[str, str | None] | None = None,
    ) -> Path:
        input_path = Path(input_path)
        records = json.loads(input_path.read_text(encoding="utf-8"))
        if not isinstance(records, list):
            raise ValueError("SIDRA fixture JSON must contain a list of flat records.")

        request_hash = content_hash({"fixture": str(input_path), "table_id": table_id})
        metadata_hash = content_hash({"fixture_metadata": table_id, "unit_by_variable": unit_by_variable or {}})

        facts = normalize_flat_records_to_facts(
            records,
            table_id=table_id,
            request_hash=request_hash,
            metadata_hash=metadata_hash,
            unit_by_variable=unit_by_variable,
        )
        return write_facts_parquet(facts, output_path=output_path)
    ''')

    write("src/pegasus/sidra/normalize.py", r'''
    from __future__ import annotations

    import re
    from typing import Any

    from pegasus.sidra.facts import normalize_flat_records_to_facts
    from pegasus.sidra.schemas import SIDRAFactRow


    def _is_header_row(row: dict[str, Any]) -> bool:
        if row.get("header_row") is True:
            return True

        value = str(row.get("V", row.get("value", ""))).strip().casefold()
        if value in {"valor", "value", "v"}:
            return True

        return str(row.get("D1C", "")).casefold() in {
            "município (código)",
            "municipio (codigo)",
            "localidade (código)",
            "localidade (codigo)",
        }


    def _sidra_level_from_nc(value: Any, fallback: str | None = None) -> str:
        if value is None:
            return fallback or ""

        text = str(value).strip()
        if text.upper().startswith("N"):
            return text.upper()

        if text.isdigit():
            return f"N{text}"

        return fallback or text


    def _ordered_classification_ids(chunk_request: dict[str, Any]) -> list[str]:
        classifications = chunk_request.get("classifications") or {}
        ids = [str(x) for x in classifications.keys()]
        return sorted(ids, key=lambda x: int(x) if x.isdigit() else x)


    def _header_dimension_names(rows: list[dict[str, Any]]) -> dict[int, str]:
        for row in rows:
            if not isinstance(row, dict):
                continue
            if not _is_header_row(row):
                continue

            names: dict[int, str] = {}
            for key, value in row.items():
                m = re.fullmatch(r"D(\d+)N", str(key))
                if m and value is not None:
                    names[int(m.group(1))] = str(value)
            return names

        return {}


    def _classification_pairs_from_flat(
        row: dict[str, Any],
        *,
        chunk_request: dict[str, Any],
        header_names: dict[int, str],
    ) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
        if "classification_tuple" in row or "category_tuple" in row:
            raw_cls = row.get("classification_tuple") or []
            raw_cat = row.get("category_tuple") or []
            return (
                [tuple(map(str, x)) for x in raw_cls],
                [tuple(map(str, x)) for x in raw_cat],
            )

        class_ids = _ordered_classification_ids(chunk_request)
        classification_pairs: list[tuple[str, str]] = []
        category_pairs: list[tuple[str, str]] = []

        # Real SIDRA view=flat layout, as observed:
        # D1 = locality, D2 = period, D3 = variable, D4+ = classifications.
        for offset, cls_id in enumerate(class_ids):
            dim_idx = 4 + offset
            code = row.get(f"D{dim_idx}C")
            if code is None:
                continue

            classification_name = header_names.get(dim_idx, "")
            classification_pairs.append((cls_id, classification_name))
            category_pairs.append((cls_id, str(code)))

        return classification_pairs, category_pairs


    def flat_response_to_records(payload: Any, *, table_id: str, chunk_request: dict[str, Any]) -> list[dict[str, Any]]:
        if isinstance(payload, dict):
            rows = payload.get("data") or payload.get("resultados") or payload.get("values") or []
        elif isinstance(payload, list):
            rows = payload
        else:
            rows = []

        rows = [row for row in rows if isinstance(row, dict)]
        header_names = _header_dimension_names(rows)

        records: list[dict[str, Any]] = []

        for row in rows:
            if _is_header_row(row):
                continue

            cls_pairs, cat_pairs = _classification_pairs_from_flat(
                row,
                chunk_request=chunk_request,
                header_names=header_names,
            )

            record = {
                "table_id": str(row.get("table_id") or row.get("agregado") or table_id),

                # Real SIDRA flat response:
                # MC/MN are measurement-unit code/name. Variable is D3C/D3N.
                "variable_id": str(row.get("variable_id") or row.get("variable") or row.get("D3C") or (chunk_request.get("variables") or [""])[0]),

                # Real SIDRA flat response:
                # D2C/D2N are period code/name.
                "period": str(row.get("period") or row.get("periodo") or row.get("D2C") or (chunk_request.get("periods") or [""])[0]),

                # Real SIDRA flat response:
                # NC is territorial-level code such as 6. Convert to N6.
                "locality_level": str(row.get("locality_level") or _sidra_level_from_nc(row.get("NC"), chunk_request.get("locality_level"))),

                # Real SIDRA flat response:
                # D1C is locality ID. NN is territorial-level name, not locality.
                "locality_id": str(row.get("locality_id") or row.get("localidade") or row.get("D1C") or ""),

                "classification_tuple": cls_pairs,
                "category_tuple": cat_pairs,
                "value": row.get("value", row.get("V")),
                "unit": row.get("unit") or row.get("MN"),
                "header_row": False,
            }

            # Preserve explicit fixture-style records.
            for key in [
                "variable_id",
                "period",
                "locality_level",
                "locality_id",
                "classification_tuple",
                "category_tuple",
                "value",
                "unit",
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

    write("tests/unit/test_sidra_flat_real_shape.py", r'''
    import json

    from pegasus.sidra.normalize import flat_response_to_records, normalize_sidra_payload_to_facts


    LIVE_SHAPE_PAYLOAD = [
        {
            "NC": "Nível Territorial (Código)",
            "NN": "Nível Territorial",
            "MC": "Unidade de Medida (Código)",
            "MN": "Unidade de Medida",
            "V": "Valor",
            "D1C": "Município (Código)",
            "D1N": "Município",
            "D2C": "Ano (Código)",
            "D2N": "Ano",
            "D3C": "Variável (Código)",
            "D3N": "Variável",
            "D4C": "Sexo (Código)",
            "D4N": "Sexo",
            "D5C": "Cor ou raça (Código)",
            "D5N": "Cor ou raça",
            "D6C": "Idade (Código)",
            "D6N": "Idade",
        },
        {
            "NC": "6",
            "NN": "Município",
            "MC": "45",
            "MN": "Pessoas",
            "V": "957916",
            "D1C": "2704302",
            "D1N": "Maceió (AL)",
            "D2C": "2022",
            "D2N": "2022",
            "D3C": "93",
            "D3N": "População residente",
            "D4C": "6794",
            "D4N": "Total",
            "D5C": "95251",
            "D5N": "Total",
            "D6C": "100362",
            "D6N": "Total",
        },
    ]


    CHUNK_REQUEST = {
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
    }


    def test_real_sidra_flat_shape_maps_core_olap_fields():
        records = flat_response_to_records(
            LIVE_SHAPE_PAYLOAD,
            table_id="9606",
            chunk_request=CHUNK_REQUEST,
        )

        assert len(records) == 1
        row = records[0]

        assert row["table_id"] == "9606"
        assert row["variable_id"] == "93"
        assert row["period"] == "2022"
        assert row["locality_level"] == "N6"
        assert row["locality_id"] == "2704302"
        assert row["value"] == "957916"
        assert row["unit"] == "Pessoas"

        assert row["classification_tuple"] == [
            ("2", "Sexo"),
            ("86", "Cor ou raça"),
            ("287", "Idade"),
        ]
        assert row["category_tuple"] == [
            ("2", "6794"),
            ("86", "95251"),
            ("287", "100362"),
        ]


    def test_real_sidra_flat_shape_normalizes_to_fact():
        facts = normalize_sidra_payload_to_facts(
            LIVE_SHAPE_PAYLOAD,
            table_id="9606",
            request_hash="request_hash",
            metadata_hash="metadata_hash",
            chunk_request=CHUNK_REQUEST,
            unit_by_variable=None,
            fetched_at="2026-06-06T00:00:00+00:00",
        )

        assert len(facts) == 1
        fact = facts[0]

        assert fact.table_id == "9606"
        assert fact.variable_id == "93"
        assert fact.period == "2022"
        assert fact.locality_level == "N6"
        assert fact.locality_id == "2704302"
        assert fact.value_raw == "957916"
        assert fact.value_numeric == 957916.0
        assert fact.value_status == "numeric"
        assert fact.unit == "Pessoas"
        assert fact.classification_tuple == (
            ("2", "Sexo"),
            ("86", "Cor ou raça"),
            ("287", "Idade"),
        )
        assert fact.category_tuple == (
            ("2", "6794"),
            ("86", "95251"),
            ("287", "100362"),
        )
    ''')

    write("tests/unit/test_sidra_extract.py", r'''
    from pathlib import Path

    import polars as pl

    from pegasus.sidra.api import SidraClient, SidraClientConfig
    from pegasus.sidra.cache import SidraJsonCache
    from pegasus.sidra.extract import extract_one_chunk
    from pegasus.sidra.schemas import SIDRAChunk


    def test_extract_one_chunk_writes_raw_and_facts_from_real_flat_shape(tmp_path: Path):
        payload = [
            {
                "NC": "Nível Territorial (Código)",
                "NN": "Nível Territorial",
                "MC": "Unidade de Medida (Código)",
                "MN": "Unidade de Medida",
                "V": "Valor",
                "D1C": "Município (Código)",
                "D1N": "Município",
                "D2C": "Ano (Código)",
                "D2N": "Ano",
                "D3C": "Variável (Código)",
                "D3N": "Variável",
                "D4C": "Sexo (Código)",
                "D4N": "Sexo",
                "D5C": "Cor ou raça (Código)",
                "D5N": "Cor ou raça",
                "D6C": "Idade (Código)",
                "D6N": "Idade",
            },
            {
                "NC": "6",
                "NN": "Município",
                "MC": "45",
                "MN": "Pessoas",
                "V": "957916",
                "D1C": "2704302",
                "D1N": "Maceió (AL)",
                "D2C": "2022",
                "D2N": "2022",
                "D3C": "93",
                "D3N": "População residente",
                "D4C": "6794",
                "D4N": "Total",
                "D5C": "95251",
                "D5N": "Total",
                "D6C": "100362",
                "D6N": "Total",
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
            classifications={"86": ["95251"], "2": ["6794"], "287": ["100362"]},
            estimated_cells=1,
            request_url="unused",
            request_params={
                "table_id": "9606",
                "variables": ["93"],
                "periods": ["2022"],
                "locality_level": "N6",
                "localities": ["2704302"],
                "classifications": {"86": ["95251"], "2": ["6794"], "287": ["100362"]},
            },
        )

        result = extract_one_chunk(
            chunk,
            client=client,
            raw_dir=tmp_path / "raw",
            facts_root=tmp_path / "facts",
            metadata_hash="metadata_hash",
            unit_by_variable=None,
        )

        assert result.status == "success"
        assert result.row_count == 1
        assert Path(result.raw_path).exists()
        assert Path(result.facts_path).exists()

        df = pl.read_parquet(result.facts_path)
        assert df.height == 1

        row = df.row(0, named=True)
        assert row["table_id"] == "9606"
        assert row["variable_id"] == "93"
        assert row["period"] == "2022"
        assert row["locality_level"] == "N6"
        assert row["locality_id"] == "2704302"
        assert row["value_numeric"] == 957916.0
        assert row["unit"] == "Pessoas"
        assert '"2","Sexo"' in row["classification_tuple"].replace(" ", "")
        assert '"2","6794"' in row["category_tuple"].replace(" ", "")
    ''')

    write("scripts/dev/audit_slice2b_sidra_outputs.py", r'''
    from __future__ import annotations

    import json
    from pathlib import Path

    import polars as pl


    def read_json(path: Path):
        return json.loads(path.read_text(encoding="utf-8"))


    def main() -> None:
        log_path = Path("data/diagnostics/sidra/population_9606_total_2022_alagoas_smoke.extraction_log.json")
        plan_path = Path("data/manifests/sidra/population_9606_total_2022_alagoas_smoke.json")

        if not log_path.exists():
            raise SystemExit(f"Missing extraction log: {log_path}")
        if not plan_path.exists():
            raise SystemExit(f"Missing plan: {plan_path}")

        plan = read_json(plan_path)
        log = read_json(log_path)

        print("\nPLAN")
        print(json.dumps(plan, indent=2, ensure_ascii=False)[:4000])

        print("\nLOG")
        print(json.dumps(log, indent=2, ensure_ascii=False)[:4000])

        if len(log) != 1:
            raise AssertionError(f"Expected exactly one smoke chunk result; got {len(log)}")

        result = log[0]
        if result["status"] != "success":
            raise AssertionError(f"Smoke extraction was not successful: {result}")

        raw_path = Path(result["raw_path"])
        facts_path = Path(result["facts_path"])

        if not raw_path.exists():
            raise AssertionError(f"Missing raw chunk file: {raw_path}")
        if not facts_path.exists():
            raise AssertionError(f"Missing facts parquet: {facts_path}")

        raw = read_json(raw_path)
        payload = raw.get("payload")

        print("\nRAW PAYLOAD FIRST 2 ROWS")
        if isinstance(payload, list):
            for row in payload[:2]:
                print(json.dumps(row, indent=2, ensure_ascii=False)[:4000])
        else:
            print(json.dumps(payload, indent=2, ensure_ascii=False)[:4000])

        df = pl.read_parquet(facts_path)

        print("\nFACTS SCHEMA")
        print(df.schema)

        print("\nFACTS ROWS")
        print(df)

        if df.height != 1:
            raise AssertionError(f"Expected exactly one normalized fact row; got {df.height}")

        row = df.row(0, named=True)

        expected = {
            "table_id": "9606",
            "variable_id": "93",
            "period": "2022",
            "locality_level": "N6",
            "locality_id": "2704302",
            "value_status": "numeric",
            "unit": "Pessoas",
        }

        hard_failures = []
        for key, value in expected.items():
            if row.get(key) != value:
                hard_failures.append((key, value, row.get(key)))

        if row.get("value_numeric") is None:
            hard_failures.append(("value_numeric", "non-null population value", None))

        cat_tuple = row.get("category_tuple")
        cls_tuple = row.get("classification_tuple")

        if not cat_tuple or cat_tuple in {"[]", "null"}:
            hard_failures.append(("category_tuple", "nonempty total race/sex/age categories", cat_tuple))

        if not cls_tuple or cls_tuple in {"[]", "null"}:
            hard_failures.append(("classification_tuple", "nonempty classification metadata", cls_tuple))

        for expected_fragment in ['"2","6794"', '"86","95251"', '"287","100362"']:
            compact = str(cat_tuple).replace(" ", "")
            if expected_fragment not in compact:
                hard_failures.append(("category_tuple", f"contains {expected_fragment}", cat_tuple))

        if hard_failures:
            print("\nAUDIT FAILURES")
            for key, expected_value, actual_value in hard_failures:
                print(f"- {key}: expected {expected_value!r}; got {actual_value!r}")
            raise SystemExit(1)

        print("\nAUDIT PASSED: live SIDRA smoke fact has correct long-form semantics.")


    if __name__ == "__main__":
        main()
    ''')

    print("Repaired Slice 2B SIDRA real flat parser and tests.")


if __name__ == "__main__":
    main()