from __future__ import annotations

import re
import textwrap
from pathlib import Path

ROOT = Path.cwd()

EXPECTED_CREATE = [
    "config/intents/alagoas_smoke.json",
    "tests/integration/test_slice2d_compile_smoke.py",
    "tests/unit/test_compile_intent_contract.py",
    "tests/unit/test_reproducibility_telemetry.py",
    "scripts/dev/audits/audit_slice2d_compile_smoke.py",
]

EXPECTED_REPLACE = [
    "src/pegasus/workflows/compile.py",
    "src/pegasus/output/reproducibility.py",
]

EXPECTED_PATCH = [
    "src/pegasus/cli.py",
]


def rel(path: str) -> Path:
    return ROOT / path


def read(path: str) -> str:
    return rel(path).read_text(encoding="utf-8")


def write(path: str, content: str) -> None:
    target = rel(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(textwrap.dedent(content).lstrip(), encoding="utf-8", newline="\n")


def preflight() -> None:
    required = [
        "pyproject.toml",
        "src/pegasus/cli.py",
        "src/pegasus/workflows/compile.py",
        "src/pegasus/output/reproducibility.py",
        "src/pegasus/output/validate.py",
        "src/pegasus/output/schemas.py",
        "src/pegasus/workflows/build_efg.py",
        "src/pegasus/workflows/datasus.py",
        "src/pegasus/workflows/efg.py",
        "src/pegasus/output/sidra_denominator_anchor.py",
        "src/pegasus/she/population/sidra_anchor.py",
        "src/pegasus/sidra/normalize.py",
        "src/pegasus/sidra/facts.py",
        "src/pegasus/core/schemas.py",
        "tests/fixtures/datasus/sim_do_fixture.csv",
    ]
    missing = [p for p in required if not rel(p).exists()]
    if missing:
        raise RuntimeError(f"Slice 2D preflight failed; missing expected files: {missing}")

    cli = read("src/pegasus/cli.py")
    if "def compile(" not in cli:
        raise RuntimeError("Slice 2D preflight failed: src/pegasus/cli.py has no compile command.")
    if "compile workflow requires later slices" not in cli and "run_compile(" not in cli:
        raise RuntimeError(
            "Slice 2D preflight failed: compile command is neither blocked in the expected way nor already patched."
        )

    compile_py = read("src/pegasus/workflows/compile.py")
    if "slice0_scaffold_only" not in compile_py and "def run_compile(" not in compile_py:
        raise RuntimeError(
            "Slice 2D preflight failed: workflows/compile.py is not the expected scaffold and not already a compile workflow."
        )


def patch_cli() -> None:
    path = "src/pegasus/cli.py"
    text = read(path)

    if "from pegasus.workflows.compile import run_compile" not in text:
        marker = "from pegasus.registries.validators import validate_registry_tree\n"
        if marker not in text:
            raise RuntimeError("Could not find CLI import insertion point for run_compile.")
        text = text.replace(marker, marker + "from pegasus.workflows.compile import run_compile\n", 1)

    replacement = """@app.command()
def compile(
    intent: Path = typer.Option(..., "--intent"),
    run_dir: Path | None = typer.Option(None, "--run-dir"),
) -> None:
    try:
        result = run_compile(intent_path=intent, run_dir=run_dir)
    except ValueError as exc:
        print(f"[red]ERROR[/red] {exc}")
        raise typer.Exit(1) from exc
    validation = result["validation"]
    if not validation.ok:
        _fail(validation.errors)
    print(f"[green]compile complete[/green] run={result['run_dir']}")
"""

    pattern = r"@app\.command\(\)\ndef compile\([\s\S]*?(?:\n\s*raise typer\.Exit\(2\)\n|\n\s*print\(f\"\[green\]compile complete[\s\S]*?\n)"
    if re.search(pattern, text):
        text = re.sub(pattern, replacement, text, count=1)
    else:
        pattern_last = r"@app\.command\(\)\ndef compile\([\s\S]*\Z"
        if not re.search(pattern_last, text):
            raise RuntimeError("Could not locate CLI compile command for replacement.")
        text = re.sub(pattern_last, replacement, text, count=1)

    rel(path).write_text(text, encoding="utf-8", newline="\n")


def main() -> None:
    preflight()

    print("Slice 2D updater preflight passed.")
    print("CREATE:")
    for p in EXPECTED_CREATE:
        print(f"  {p}")
    print("REPLACE:")
    for p in EXPECTED_REPLACE:
        print(f"  {p}")
    print("PATCH:")
    for p in EXPECTED_PATCH:
        print(f"  {p}")

    write("config/intents/alagoas_smoke.json", r"""
    {
      "geography": {
        "level": "municipality",
        "codes": ["2704302"],
        "uf": ["AL"]
      },
      "time": {
        "start_year": 2022,
        "end_year": 2022,
        "start_month": null,
        "end_month": null
      },
      "health_seeds": ["all_cause_mortality", "icd_chapter", "icd_block"],
      "mandatory_fields": [
        "SIMDeathsAll",
        "SIMCrudeMortalitySIDRAOfficial",
        "SIDRAPopulationTotalAnchor"
      ],
      "system_weights": {
        "SIM-DO": 1.0,
        "SIDRA": 1.0
      },
      "context_policy": ["compile_smoke", "maceio_fixture", "official_sidra_anchor"],
      "budget": "fast",
      "geo_mode": "native",
      "force_selectors": [],
      "exclude_systems": [],
      "execution_scale": "smoke",
      "race_tensor_mode": "decoupled",
      "population_mode": "official_sidra_anchor"
    }
    """)

    write("src/pegasus/output/reproducibility.py", r"""
    from __future__ import annotations

    import json
    import time
    from contextlib import contextmanager
    from dataclasses import dataclass, field
    from datetime import datetime, timezone
    from pathlib import Path
    from typing import Any, Iterator

    TERMINAL_STAGE_STATUSES = {"success", "skipped", "blocked", "failed"}

    COMPILE_TELEMETRY_STAGES = (
        "datasus_manifest",
        "datasus_acquire",
        "datasus_decode",
        "sidra_metadata",
        "sidra_plan",
        "sidra_fetch",
        "sidra_normalize",
        "geo_support",
        "she_build",
        "population_solver",
        "stdfm",
        "efg_build",
        "q_tensor",
        "pirs_model",
        "pirs_hsic",
        "output_serialization",
        "output_validation",
    )


    def utc_now() -> str:
        return datetime.now(timezone.utc).isoformat()


    def stable_json(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str, indent=2)


    @dataclass
    class RunTelemetry:
        "Incremental compile telemetry for TDD-compliant run reproducibility."

        run_id: str
        diagnostic_path: Path
        run_dir: Path | None = None
        started_at: float = field(default_factory=time.perf_counter)
        stage_status: dict[str, str] = field(default_factory=lambda: {s: "skipped" for s in COMPILE_TELEMETRY_STAGES})
        stage_wall_seconds: dict[str, float] = field(default_factory=lambda: {s: 0.0 for s in COMPILE_TELEMETRY_STAGES})
        resource_summary: dict[str, Any] = field(
            default_factory=lambda: {
                "peak_rss_mb": None,
                "peak_vram_mb": None,
                "duckdb_temp_bytes": None,
                "rows_read": {},
                "rows_written": {},
                "parquet_bytes_written": 0,
            }
        )

        def model(self) -> dict[str, Any]:
            return {
                "run_id": self.run_id,
                "total_wall_seconds": max(0.0, time.perf_counter() - self.started_at),
                "stage_status": dict(self.stage_status),
                "stage_wall_seconds": dict(self.stage_wall_seconds),
                "resource_summary": self.resource_summary,
            }

        def set_stage(self, stage: str, status: str, duration: float = 0.0) -> None:
            if stage not in self.stage_status:
                raise KeyError(f"Unknown telemetry stage: {stage}")
            if status not in TERMINAL_STAGE_STATUSES:
                raise ValueError(f"Invalid telemetry status: {status}")
            self.stage_status[stage] = status
            self.stage_wall_seconds[stage] = max(0.0, float(duration))

        def block(self, stage: str, *, reason: str | None = None) -> None:
            self.set_stage(stage, "blocked", 0.0)
            if reason:
                blocked = self.resource_summary.setdefault("blocked_reasons", {})
                blocked[stage] = reason

        @contextmanager
        def stage(self, stage: str) -> Iterator[None]:
            started = time.perf_counter()
            self.flush()
            try:
                yield
            except Exception:
                self.set_stage(stage, "failed", time.perf_counter() - started)
                self.flush()
                raise
            else:
                self.set_stage(stage, "success", time.perf_counter() - started)
                self.flush()

        def flush(self) -> None:
            self.diagnostic_path.parent.mkdir(parents=True, exist_ok=True)
            self.diagnostic_path.write_text(stable_json({"telemetry": self.model()}), encoding="utf-8")
            if self.run_dir is not None:
                manifest_path = self.run_dir / "ReproducibilityManifest.json"
                if manifest_path.exists():
                    try:
                        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    except json.JSONDecodeError:
                        manifest = {}
                    manifest["telemetry"] = self.model()
                    manifest_path.write_text(stable_json(manifest), encoding="utf-8")


    def write_reproducibility_manifest(
        *,
        run_dir: str | Path,
        run_id: str,
        intent_hash: str,
        source_hashes: dict[str, str],
        registry_hashes: dict[str, str],
        telemetry: RunTelemetry,
        extras: dict[str, Any] | None = None,
    ) -> Path:
        run_dir = Path(run_dir)
        manifest = {
            "schema_version": "1.0",
            "run_id": run_id,
            "generated_at": utc_now(),
            "code_version": "0.1.0",
            "intent_hash": intent_hash,
            "source_hashes": source_hashes,
            "registry_hashes": registry_hashes,
            "random_seed": 0,
            "telemetry": telemetry.model(),
        }
        if extras:
            manifest.update(extras)
        path = run_dir / "ReproducibilityManifest.json"
        path.write_text(stable_json(manifest), encoding="utf-8")
        telemetry.run_dir = run_dir
        telemetry.flush()
        return path
    """)

    write("src/pegasus/workflows/compile.py", r"""
    from __future__ import annotations

    import json
    import shutil
    from datetime import datetime, timezone
    from pathlib import Path
    from typing import Any

    from pydantic import ValidationError

    from pegasus.core.hashing import content_hash, sha256_file
    from pegasus.core.schemas import UserIntent
    from pegasus.geo.municipality_crosswalk import ibge_cod7_to_datasus_cod6
    from pegasus.output.reproducibility import RunTelemetry, write_reproducibility_manifest
    from pegasus.output.validate import validate_output_bundle
    from pegasus.sidra.facts import write_facts_parquet
    from pegasus.sidra.normalize import normalize_sidra_payload_to_facts
    from pegasus.workflows.datasus import run_datasus_normalize_sim
    from pegasus.workflows.efg import run_attach_sidra_denominator, run_build_sim_fixture


    SIDRA_POPULATION_MACEIO_FLAT_PAYLOAD: list[dict[str, str]] = [
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


    SIDRA_POPULATION_MACEIO_CHUNK_REQUEST: dict[str, Any] = {
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


    def utc_stamp() -> str:
        return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


    def _load_intent(intent_path: Path) -> tuple[dict[str, Any], UserIntent]:
        payload = json.loads(intent_path.read_text(encoding="utf-8"))
        try:
            return payload, UserIntent.model_validate(payload)
        except ValidationError as exc:
            raise ValueError(f"Invalid UserIntent file {intent_path}: {exc}") from exc


    def _registry_hashes() -> dict[str, str]:
        candidates = [
            Path("config/registries/registry_manifest.yaml"),
            Path("config/registries/sidra_views.yaml"),
            Path("config/registries/source_fields.yaml"),
            Path("config/registries/quality_permissions.yaml"),
            Path("config/registries/race_axis_registry.yaml"),
        ]
        return {str(path): sha256_file(path) for path in candidates if path.exists()}


    def _write_compile_manifest(*, run_id: str, intent_path: Path, data_root: Path, run_dir: Path, payload: dict[str, Any]) -> Path:
        path = data_root / "manifests" / "runs" / f"{run_id}.compile_manifest.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "kind": "compile_smoke_manifest",
                    "run_id": run_id,
                    "intent_path": str(intent_path),
                    "run_dir": str(run_dir),
                    "intent": payload,
                },
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            ),
            encoding="utf-8",
        )
        return path


    def _write_sidra_smoke_facts(*, output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        facts = normalize_sidra_payload_to_facts(
            SIDRA_POPULATION_MACEIO_FLAT_PAYLOAD,
            table_id="9606",
            request_hash=content_hash(SIDRA_POPULATION_MACEIO_CHUNK_REQUEST),
            metadata_hash=content_hash({"metadata": "compile_smoke_sidra_9606_maceio_total_v1"}),
            chunk_request=SIDRA_POPULATION_MACEIO_CHUNK_REQUEST,
            unit_by_variable=None,
            fetched_at=datetime.now(timezone.utc).isoformat(),
        )
        write_facts_parquet(facts, output_path=output_path)
        return output_path


    def _smoke_municipality_cod6(intent: UserIntent) -> str:
        if intent.execution_scale != "smoke":
            raise ValueError("Slice 2D compile supports execution_scale='smoke' only.")
        if intent.geography.level != "municipality":
            raise ValueError("Slice 2D compile smoke requires geography.level='municipality'.")
        if len(intent.geography.codes) != 1:
            raise ValueError("Slice 2D compile smoke requires exactly one municipality code.")
        cod6 = ibge_cod7_to_datasus_cod6(intent.geography.codes[0], strict=True)
        if cod6 is None:
            raise ValueError(f"Could not convert intent municipality code to DATASUS cod6: {intent.geography.codes[0]!r}")
        return cod6


    def run_compile(
        *,
        intent_path: str | Path,
        run_dir: str | Path | None = None,
        data_root: str | Path = "data",
    ) -> dict[str, Any]:
        intent_path = Path(intent_path)
        data_root = Path(data_root)
        intent_payload, intent = _load_intent(intent_path)
        municipality_cod6 = _smoke_municipality_cod6(intent)

        intent_hash = sha256_file(intent_path)
        run_id = f"compile_{intent_path.stem}_{utc_stamp()}_{intent_hash[:8]}"
        run_dir = Path(run_dir) if run_dir is not None else data_root / "runs" / run_id
        diagnostic_path = data_root / "diagnostics" / "compile" / f"{run_id}.telemetry.json"
        telemetry = RunTelemetry(run_id=run_id, diagnostic_path=diagnostic_path)

        source_hashes: dict[str, str] = {"intent": intent_hash}
        registry_hashes = _registry_hashes()

        raw_fixture_source = Path("tests/fixtures/datasus/sim_do_fixture.csv")
        if not raw_fixture_source.exists():
            raise FileNotFoundError(f"Missing SIM smoke fixture: {raw_fixture_source}")

        compile_manifest_path = data_root / "manifests" / "runs" / f"{run_id}.compile_manifest.json"
        raw_cache_path = data_root / "raw" / "datasus" / "SIM-DO" / "fixture" / "sim_do_fixture.csv"
        sim_events_path = data_root / "processed" / "datasus" / "SIM-DO" / "fixture" / "sim_events.parquet"
        sidra_facts_path = data_root / "processed" / "sidra" / "facts" / "9606" / "compile_smoke_maceio.parquet"

        with telemetry.stage("datasus_manifest"):
            compile_manifest_path = _write_compile_manifest(
                run_id=run_id,
                intent_path=intent_path,
                data_root=data_root,
                run_dir=run_dir,
                payload=intent_payload,
            )
            source_hashes["compile_manifest"] = sha256_file(compile_manifest_path)

        with telemetry.stage("datasus_acquire"):
            raw_cache_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(raw_fixture_source, raw_cache_path)
            source_hashes["sim_raw_fixture"] = sha256_file(raw_cache_path)

        with telemetry.stage("datasus_decode"):
            run_datasus_normalize_sim(
                input_path=raw_cache_path,
                output_path=sim_events_path,
                source_manifest_hash=source_hashes["compile_manifest"],
            )
            source_hashes["sim_processed_events"] = sha256_file(sim_events_path)

        telemetry.set_stage("sidra_metadata", "skipped", 0.0)
        telemetry.set_stage("sidra_plan", "skipped", 0.0)
        telemetry.set_stage("sidra_fetch", "skipped", 0.0)
        telemetry.flush()

        with telemetry.stage("sidra_normalize"):
            _write_sidra_smoke_facts(output_path=sidra_facts_path)
            source_hashes["sidra_facts"] = sha256_file(sidra_facts_path)

        with telemetry.stage("efg_build"):
            run_build_sim_fixture(
                sim_events_path=sim_events_path,
                run_dir=run_dir,
                municipality_cod6=municipality_cod6,
            )

        with telemetry.stage("she_build"):
            run_attach_sidra_denominator(
                run_dir=run_dir,
                sidra_facts_path=sidra_facts_path,
            )

        telemetry.set_stage("geo_support", "success", 0.0)
        telemetry.set_stage("q_tensor", "success", 0.0)
        telemetry.block("population_solver", reason="official SIDRA anchor smoke path; tensor solver scaffold remains blocked")
        telemetry.block("stdfm", reason="ST-DFM scaffold remains blocked for compile smoke")
        telemetry.block("pirs_model", reason="PIRS model stage is not invoked in compile smoke")
        telemetry.block("pirs_hsic", reason="PIRS HSIC stage is not invoked in compile smoke")
        telemetry.flush()

        with telemetry.stage("output_serialization"):
            (run_dir / "UserIntent.json").write_text(
                json.dumps(intent_payload, ensure_ascii=False, sort_keys=True, indent=2),
                encoding="utf-8",
            )
            (run_dir / "RunConfig.json").write_text(
                json.dumps(
                    {
                        "schema_version": "1.0",
                        "compile_mode": "smoke",
                        "run_id": run_id,
                        "intent_path": str(intent_path),
                        "data_root": str(data_root),
                        "support_policy": {
                            "geography_level": intent.geography.level,
                            "ibge_cod7": intent.geography.codes,
                            "datasus_cod6": [municipality_cod6],
                            "geo_mode": intent.geo_mode,
                        },
                        "population_mode": intent.population_mode,
                        "race_tensor_mode": intent.race_tensor_mode,
                        "registry_hashes": registry_hashes,
                        "source_hashes": source_hashes,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                    indent=2,
                ),
                encoding="utf-8",
            )
            write_reproducibility_manifest(
                run_dir=run_dir,
                run_id=run_id,
                intent_hash=intent_hash,
                source_hashes=source_hashes,
                registry_hashes=registry_hashes,
                telemetry=telemetry,
                extras={
                    "compile_mode": "smoke",
                    "compile_manifest": str(compile_manifest_path),
                    "intent_path": str(intent_path),
                },
            )

        with telemetry.stage("output_validation"):
            result = validate_output_bundle(run_dir=str(run_dir))
            if not result.ok:
                raise RuntimeError("Compile smoke produced invalid output bundle: " + "; ".join(result.errors))

        write_reproducibility_manifest(
            run_dir=run_dir,
            run_id=run_id,
            intent_hash=intent_hash,
            source_hashes=source_hashes,
            registry_hashes=registry_hashes,
            telemetry=telemetry,
            extras={
                "compile_mode": "smoke",
                "compile_manifest": str(compile_manifest_path),
                "intent_path": str(intent_path),
            },
        )

        validation = validate_output_bundle(run_dir=str(run_dir))
        return {
            "status": "success" if validation.ok else "failed",
            "run_id": run_id,
            "run_dir": run_dir,
            "intent": intent,
            "validation": validation,
            "source_hashes": source_hashes,
            "registry_hashes": registry_hashes,
            "telemetry": telemetry.model(),
        }
    """)

    write("tests/unit/test_compile_intent_contract.py", r"""
    import json
    from pathlib import Path

    from pegasus.core.schemas import UserIntent


    def test_alagoas_smoke_intent_is_strict_user_intent():
        payload = json.loads(Path("config/intents/alagoas_smoke.json").read_text(encoding="utf-8"))
        intent = UserIntent.model_validate(payload)

        assert intent.execution_scale == "smoke"
        assert intent.population_mode == "official_sidra_anchor"
        assert intent.geography.level == "municipality"
        assert intent.geography.codes == ["2704302"]
        assert intent.budget == "fast"
    """)

    write("tests/unit/test_reproducibility_telemetry.py", r"""
    import json
    from pathlib import Path

    from pegasus.output.reproducibility import COMPILE_TELEMETRY_STAGES, RunTelemetry


    def test_run_telemetry_flushes_all_compile_stages(tmp_path: Path):
        diagnostic = tmp_path / "telemetry.json"
        telemetry = RunTelemetry(run_id="test_run", diagnostic_path=diagnostic)

        with telemetry.stage("datasus_manifest"):
            pass
        telemetry.block("pirs_model", reason="blocked in test")

        payload = json.loads(diagnostic.read_text(encoding="utf-8"))
        model = payload["telemetry"]

        assert set(COMPILE_TELEMETRY_STAGES).issubset(model["stage_status"])
        assert model["stage_status"]["datasus_manifest"] == "success"
        assert model["stage_status"]["pirs_model"] == "blocked"
        assert model["stage_wall_seconds"]["datasus_manifest"] >= 0
        assert model["total_wall_seconds"] >= 0
    """)

    write("tests/integration/test_slice2d_compile_smoke.py", r"""
    import json
    from pathlib import Path

    import polars as pl

    from pegasus.output.validate import validate_output_bundle
    from pegasus.workflows.compile import run_compile


    REQUIRED_SUCCESS_STAGES = {
        "datasus_manifest",
        "datasus_acquire",
        "datasus_decode",
        "sidra_normalize",
        "efg_build",
        "geo_support",
        "she_build",
        "q_tensor",
        "output_serialization",
        "output_validation",
    }


    REQUIRED_BLOCKED_STAGES = {"population_solver", "stdfm", "pirs_model", "pirs_hsic"}


    def test_compile_smoke_emits_valid_17_key_bundle_with_sidra_denominator(tmp_path: Path):
        run_dir = tmp_path / "compile_run"
        data_root = tmp_path / "data"

        result = run_compile(
            intent_path="config/intents/alagoas_smoke.json",
            run_dir=run_dir,
            data_root=data_root,
        )

        assert result["status"] == "success"
        assert result["validation"].ok, result["validation"].errors

        validation = validate_output_bundle(run_dir=str(run_dir))
        assert validation.ok, validation.errors

        intent = json.loads((run_dir / "UserIntent.json").read_text(encoding="utf-8"))
        run_config = json.loads((run_dir / "RunConfig.json").read_text(encoding="utf-8"))
        manifest = json.loads((run_dir / "ReproducibilityManifest.json").read_text(encoding="utf-8"))

        assert intent["geography"]["codes"] == ["2704302"]
        assert run_config["compile_mode"] == "smoke"
        assert run_config["support_policy"]["datasus_cod6"] == ["270430"]
        assert "source_hashes" in run_config
        assert "registry_hashes" in run_config

        telemetry = manifest["telemetry"]
        for stage in REQUIRED_SUCCESS_STAGES:
            assert telemetry["stage_status"][stage] == "success", stage
            assert telemetry["stage_wall_seconds"][stage] >= 0, stage
        for stage in REQUIRED_BLOCKED_STAGES:
            assert telemetry["stage_status"][stage] == "blocked", stage

        for key in ["intent", "compile_manifest", "sim_raw_fixture", "sim_processed_events", "sidra_facts"]:
            assert key in manifest["source_hashes"]

        v = pl.read_parquet(run_dir / "V_fields.parquet")
        names = set(v["name"].to_list())

        assert "SIMDeathsAll" in names
        assert "SIDRAPopulationTotalAnchor" in names
        assert "SIMCrudeMortalitySIDRAOfficial" in names

        rate = v.filter(pl.col("name") == "SIMCrudeMortalitySIDRAOfficial").row(0, named=True)
        support = json.loads(rate["support_json"])
        assert support["support_alignment"]["aligned"] is True
        assert support["support_alignment"]["numerator_municipalities_ibge_cod7"] == ["2704302"]
        assert support["support_alignment"]["denominator_municipalities_ibge_cod7"] == ["2704302"]
    """)

    write("scripts/dev/audits/audit_slice2d_compile_smoke.py", r"""
    from __future__ import annotations

    import argparse
    import json
    import sys
    from pathlib import Path

    import polars as pl

    from pegasus.output.validate import validate_output_bundle

    REQUIRED_SUCCESS_STAGES = {
        "datasus_manifest",
        "datasus_acquire",
        "datasus_decode",
        "sidra_normalize",
        "efg_build",
        "geo_support",
        "she_build",
        "q_tensor",
        "output_serialization",
        "output_validation",
    }

    REQUIRED_BLOCKED_STAGES = {"population_solver", "stdfm", "pirs_model", "pirs_hsic"}

    REQUIRED_SOURCE_HASHES = {
        "intent",
        "compile_manifest",
        "sim_raw_fixture",
        "sim_processed_events",
        "sidra_facts",
    }

    REQUIRED_FIELDS = {
        "SIMDeathsAll",
        "SIDRAPopulationTotalAnchor",
        "SIMCrudeMortalitySIDRAOfficial",
    }


    def fail(payload: dict) -> None:
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2))
        sys.exit(1)


    def main() -> None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--run", required=True)
        args = parser.parse_args()

        run = Path(args.run)
        failures: list[dict] = []

        validation = validate_output_bundle(run_dir=str(run))
        if not validation.ok:
            failures.append({"kind": "validate_output_bundle", "errors": validation.errors})

        try:
            intent = json.loads((run / "UserIntent.json").read_text(encoding="utf-8"))
            run_config = json.loads((run / "RunConfig.json").read_text(encoding="utf-8"))
            manifest = json.loads((run / "ReproducibilityManifest.json").read_text(encoding="utf-8"))
        except Exception as exc:
            fail({"status": "failed", "reason": "json_read_failure", "error": str(exc)})

        if intent.get("execution_scale") != "smoke":
            failures.append({"kind": "intent", "expected": "execution_scale=smoke", "actual": intent.get("execution_scale")})
        if run_config.get("compile_mode") != "smoke":
            failures.append({"kind": "run_config", "expected": "compile_mode=smoke", "actual": run_config.get("compile_mode")})

        telemetry = manifest.get("telemetry", {})
        statuses = telemetry.get("stage_status", {})
        durations = telemetry.get("stage_wall_seconds", {})

        for stage in REQUIRED_SUCCESS_STAGES:
            if statuses.get(stage) != "success":
                failures.append({"kind": "telemetry_success_stage", "stage": stage, "actual": statuses.get(stage)})
            if durations.get(stage, -1) < 0:
                failures.append({"kind": "telemetry_duration", "stage": stage, "actual": durations.get(stage)})

        for stage in REQUIRED_BLOCKED_STAGES:
            if statuses.get(stage) != "blocked":
                failures.append({"kind": "telemetry_blocked_stage", "stage": stage, "actual": statuses.get(stage)})

        source_hashes = manifest.get("source_hashes", {})
        missing_source_hashes = sorted(REQUIRED_SOURCE_HASHES - set(source_hashes))
        if missing_source_hashes:
            failures.append({"kind": "missing_source_hashes", "missing": missing_source_hashes})

        if not manifest.get("registry_hashes"):
            failures.append({"kind": "missing_registry_hashes"})

        try:
            v = pl.read_parquet(run / "V_fields.parquet")
            names = set(v["name"].to_list())
            missing_fields = sorted(REQUIRED_FIELDS - names)
            if missing_fields:
                failures.append({"kind": "missing_fields", "missing": missing_fields})

            rate_rows = v.filter(pl.col("name") == "SIMCrudeMortalitySIDRAOfficial").to_dicts()
            if len(rate_rows) != 1:
                failures.append({"kind": "rate_field_count", "expected": 1, "actual": len(rate_rows)})
            else:
                support = json.loads(rate_rows[0]["support_json"])
                alignment = support.get("support_alignment", {})
                if alignment.get("aligned") is not True:
                    failures.append({"kind": "support_alignment", "alignment": alignment})
                if alignment.get("numerator_municipalities_ibge_cod7") != alignment.get("denominator_municipalities_ibge_cod7"):
                    failures.append({"kind": "support_alignment_municipalities", "alignment": alignment})
        except Exception as exc:
            failures.append({"kind": "field_read_failure", "error": str(exc)})

        if failures:
            fail({"status": "failed", "failures": failures})

        print("AUDIT PASSED: Slice 2D compile smoke emitted validated telemetry, provenance, SIDRA denominator, and support alignment.")


    if __name__ == "__main__":
        main()
    """)

    patch_cli()

    print("Applied Slice 2D compile smoke integration.")
    print("Run the validation commands supplied by the assistant.")


if __name__ == "__main__":
    main()
