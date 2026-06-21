from __future__ import annotations

import re
import shutil
from datetime import datetime
from pathlib import Path

ROOT = Path.cwd()
BACKUP_ROOT = ROOT / ".codex-tmp" / "alagoas_state_support_backups" / datetime.now().strftime("%Y%m%d_%H%M%S")

COMPILE = ROOT / "src" / "pegasus" / "workflows" / "compile.py"
RUNTIME = ROOT / "scripts" / "dev" / "audits" / "actual_data_smoke_runtime.py"
TEST = ROOT / "tests" / "unit" / "test_alagoas_state_actual_support.py"


def read(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(path)
    return path.read_text(encoding="utf-8")


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        backup = BACKUP_ROOT / path.relative_to(ROOT)
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, backup)
    path.write_text(text, encoding="utf-8")


def patch_compile() -> None:
    text = read(COMPILE)

    pattern = re.compile(
        r"def _smoke_municipality_cod6\(intent: UserIntent\) -> str:\n"
        r"(?:    .*\n)+?"
        r"(?=\n\ndef )",
        re.MULTILINE,
    )

    replacement = '''def _intent_municipality_filter_cod6(intent: UserIntent) -> str | None:
    """Resolve the optional DATASUS municipality filter for compile inputs.

    Smoke runs remain single-municipality runs. State runs over a UF deliberately
    return None so SIM/SINASC are not filtered down to one municipality.
    This is a state-level aggregate support, not yet a per-municipality panel.
    """
    if intent.geography.level != "municipality":
        raise ValueError("Current compile supports geography.level='municipality' only.")

    if intent.execution_scale == "smoke":
        if len(intent.geography.codes) != 1:
            raise ValueError("Compile smoke requires exactly one municipality code.")
        cod6 = ibge_cod7_to_datasus_cod6(intent.geography.codes[0], strict=True)
        if cod6 is None:
            raise ValueError(f"Could not resolve municipality DATASUS cod6 for {intent.geography.codes[0]!r}.")
        return cod6

    if intent.execution_scale == "state":
        if intent.geography.codes:
            raise ValueError(
                "State compile currently expects geography.codes=[] and geography.uf=[<UF>]. "
                "Explicit multi-code municipal subsets require panel support and are not implemented in this slice."
            )
        if len(intent.geography.uf) != 1:
            raise ValueError("State compile requires exactly one UF in geography.uf.")
        return None

    raise ValueError(
        "Compile currently supports execution_scale='smoke' or execution_scale='state'. "
        f"Received {intent.execution_scale!r}."
    )


def _smoke_municipality_cod6(intent: UserIntent) -> str:
    """Legacy strict helper retained for tests and smoke-only callers."""
    cod6 = _intent_municipality_filter_cod6(intent)
    if cod6 is None:
        raise ValueError("Compile smoke helper received a non-smoke/state-wide intent.")
    return cod6
'''

    text, count = pattern.subn(replacement, text, count=1)
    if count != 1 and "_intent_municipality_filter_cod6" not in text:
        raise RuntimeError("Could not replace _smoke_municipality_cod6 block in compile.py.")

    old = "    municipality_cod6 = _smoke_municipality_cod6(intent)\n"
    new = (
        "    municipality_cod6 = _intent_municipality_filter_cod6(intent)\n"
        "    if municipality_cod6 is None and str(intent.race_tensor_mode) != \"decoupled\":\n"
        "        raise ValueError(\"State-level compile currently supports race_tensor_mode='decoupled' only.\")\n"
    )
    if old in text:
        text = text.replace(old, new, 1)
    elif "municipality_cod6 = _intent_municipality_filter_cod6(intent)" not in text:
        raise RuntimeError("Could not replace municipality_cod6 resolver call in compile.py.")

    write(COMPILE, text)


def patch_actual_smoke_runtime() -> None:
    text = read(RUNTIME)

    if "import polars as pl" not in text:
        text = text.replace("from typing import Any\n", "from typing import Any\n\nimport polars as pl\n", 1)

    old = '''    write_facts_parquet(facts, output_path=path)
    municipality_count = len({str(row.get("D1C")) for row in response.payload[1:] if isinstance(row, dict) and row.get("D1C")})
    return path, bool(response.from_cache or response.status_code < 400), municipality_count
'''

    new = '''    write_facts_parquet(facts, output_path=path)

    df = pl.read_parquet(path)
    if "locality_id" not in df.columns:
        raise RuntimeError("SIDRA population facts lack locality_id after normalization.")

    if localities == ["all"]:
        # SIDRA N6/all returns a municipality-level Brazil-wide payload. For the
        # current state-level compiler contract, collapse AL municipalities into
        # one official AL anchor while retaining municipality_count for the grid
        # audit gate. The full municipal panel is a later EFG/Q tensor expansion.
        al = df.filter(
            (pl.col("locality_id").cast(pl.Utf8).str.starts_with("27"))
            & (pl.col("value_status").cast(pl.Utf8) == "numeric")
            & pl.col("value_numeric").is_not_null()
        )
        municipality_count = int(al.select(pl.col("locality_id").n_unique()).item()) if al.height else 0
        if municipality_count <= 1:
            raise RuntimeError(f"SIDRA N6/all did not yield multi-municipality AL support: {municipality_count}")

        total = float(al.select(pl.col("value_numeric").sum()).item())
        row = dict(al.head(1).to_dicts()[0])
        row.update(
            {
                "locality_level": "N3",
                "locality_id": "27",
                "value_numeric": total,
                "value_raw": str(int(total)) if total.is_integer() else str(total),
                "request_hash": content_hash({**request, "aggregation": "AL_N6_sum_to_UF"}),
                "metadata_hash": content_hash({"table": "9606", "official": True, "aggregation": "AL_N6_sum_to_UF"}),
            }
        )
        pl.DataFrame([row], infer_schema_length=None).write_parquet(path)
    else:
        municipality_count = int(df.select(pl.col("locality_id").n_unique()).item()) if df.height else 0

    return path, bool(response.from_cache or response.status_code < 400), municipality_count
'''

    if old in text:
        text = text.replace(old, new, 1)
    elif "AL_N6_sum_to_UF" not in text:
        raise RuntimeError(
            "Could not find the expected _sidra_facts write/return block. "
            "Open scripts/dev/audits/actual_data_smoke_runtime.py and patch _sidra_facts manually."
        )

    write(RUNTIME, text)


def write_tests() -> None:
    text = '''from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from pegasus.core.schemas import UserIntent
from pegasus.workflows.compile import _intent_municipality_filter_cod6, _smoke_municipality_cod6


def test_compile_support_resolver_allows_alagoas_state_without_municipality_filter() -> None:
    payload = json.loads(Path("config/intents/alagoas_2022_actual_grid_smoke.json").read_text(encoding="utf-8"))
    intent = UserIntent.model_validate(payload)

    assert intent.execution_scale == "state"
    assert intent.geography.uf == ["AL"]
    assert intent.geography.codes == []
    assert _intent_municipality_filter_cod6(intent) is None


def test_compile_support_resolver_preserves_maceio_smoke_filter() -> None:
    payload = json.loads(Path("config/intents/alagoas_maceio_2022_actual_smoke.json").read_text(encoding="utf-8"))
    intent = UserIntent.model_validate(payload)

    assert _intent_municipality_filter_cod6(intent) == "270430"
    assert _smoke_municipality_cod6(intent) == "270430"


def test_actual_smoke_grid_classifier_requires_multi_municipality_support() -> None:
    path = Path("scripts/dev/audits/actual_data_smoke_runtime.py")
    spec = importlib.util.spec_from_file_location("actual_data_smoke_runtime", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    payload = {
        "compile_source_mode": "materialized_external",
        "output_validator_ok": True,
        "dashboard_read_only_ok": True,
        "mandatory_fields_present": True,
        "efg_fields_nonempty": True,
        "efg_edges_nonempty": True,
        "q_tensor_nonempty": True,
        "fixture_semantics_present": [],
        "municipality_count": 102,
    }

    assert module.classify_actual_smoke(payload, grid=True) == "actual_data_validated"

    payload["municipality_count"] = 1
    assert module.classify_actual_smoke(payload, grid=True) == "source_partial"
'''
    write(TEST, text)


def main() -> None:
    patch_compile()
    patch_actual_smoke_runtime()
    write_tests()
    print("Enabled Alagoas state-level actual-source compile support.")
    print(f"Backups: {BACKUP_ROOT}")


if __name__ == "__main__":
    main()
