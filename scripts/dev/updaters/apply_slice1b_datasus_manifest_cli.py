from __future__ import annotations

from pathlib import Path
import textwrap

ROOT = Path.cwd()


def write(path: str, content: str) -> None:
    target = ROOT / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(textwrap.dedent(content).lstrip(), encoding="utf-8", newline="\n")


def main() -> None:
    write("src/pegasus/datasus/manifests.py", r'''
    from __future__ import annotations

    import json
    import re
    from datetime import datetime, timezone
    from pathlib import Path
    from typing import Any

    from pegasus.core.config import load_yaml
    from pegasus.core.hashing import content_hash
    from pegasus.core.schemas import DATASUSRequestManifest


    ALLOWED_SYSTEMS = {"SIM-DO", "SINASC", "SIH-RD", "CNES-ST"}
    UF_RE = re.compile(r"^[A-Z]{2}$")


    def utc_now() -> str:
        return datetime.now(timezone.utc).isoformat()


    def normalize_system(system: str) -> str:
        value = system.strip().upper()
        aliases = {
            "SIM": "SIM-DO",
            "SIM_DO": "SIM-DO",
            "SIM-DO": "SIM-DO",
            "DO": "SIM-DO",
            "SINASC": "SINASC",
            "SIH": "SIH-RD",
            "SIH_RD": "SIH-RD",
            "SIH-RD": "SIH-RD",
            "RD": "SIH-RD",
            "CNES": "CNES-ST",
            "CNES_ST": "CNES-ST",
            "CNES-ST": "CNES-ST",
            "ST": "CNES-ST",
        }
        if value not in aliases:
            raise ValueError(f"Unsupported DATASUS system: {system}")
        return aliases[value]


    def normalize_uf(uf: str) -> str:
        value = uf.strip().upper()
        if not UF_RE.fullmatch(value):
            raise ValueError(f"Invalid UF code: {uf}")
        return value


    def parse_years(years: str) -> list[int]:
        result: set[int] = set()

        for part in years.split(","):
            token = part.strip()
            if not token:
                continue

            if "-" in token:
                left, right = [x.strip() for x in token.split("-", 1)]
                start = int(left)
                end = int(right)
                if end < start:
                    raise ValueError(f"Invalid descending year range: {token}")
                result.update(range(start, end + 1))
            else:
                result.add(int(token))

        if not result:
            raise ValueError("No years were parsed.")

        for year in result:
            if year < 1970 or year > 2100:
                raise ValueError(f"Suspicious DATASUS year: {year}")

        return sorted(result)


    def period_label(
        *,
        year_start: int,
        year_end: int,
        month_start: int | None = None,
        month_end: int | None = None,
    ) -> str:
        ms = "NA" if month_start is None else f"{month_start:02d}"
        me = "NA" if month_end is None else f"{month_end:02d}"
        return f"{year_start}_{ms}__{year_end}_{me}"


    def load_datasus_config(path: str | Path = "config/datasus.yaml") -> dict[str, Any]:
        data = load_yaml(path)
        return data.get("datasus", data)


    def _request_identity(
        *,
        system: str,
        uf: str,
        year_start: int,
        year_end: int,
        month_start: int | None,
        month_end: int | None,
        config: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "schema_version": "1.0",
            "backend": config.get("backend", "microdatasus"),
            "system": system,
            "uf": uf,
            "year_start": year_start,
            "year_end": year_end,
            "month_start": month_start,
            "month_end": month_end,
            "information_system": system,
            "fetch_function": "fetch_datasus",
            "process_function": "process_datasus",
        }


    def build_datasus_request_manifest(
        *,
        system: str,
        uf: str,
        year_start: int,
        year_end: int,
        month_start: int | None = None,
        month_end: int | None = None,
        config: dict[str, Any] | None = None,
        data_root: str | Path = "data",
    ) -> DATASUSRequestManifest:
        config = config or load_datasus_config()
        system = normalize_system(system)
        uf = normalize_uf(uf)

        identity = _request_identity(
            system=system,
            uf=uf,
            year_start=year_start,
            year_end=year_end,
            month_start=month_start,
            month_end=month_end,
            config=config,
        )
        request_hash = content_hash(identity)
        period = period_label(
            year_start=year_start,
            year_end=year_end,
            month_start=month_start,
            month_end=month_end,
        )

        data_root = Path(data_root)

        raw_dir = data_root / "raw" / "datasus" / system / f"uf={uf}" / f"period={period}" / request_hash
        processed_dir = data_root / "processed" / "datasus" / system / f"uf={uf}" / f"period={period}" / request_hash

        now = utc_now()

        return DATASUSRequestManifest(
            system=system,
            uf=uf,
            year_start=year_start,
            month_start=month_start,
            year_end=year_end,
            month_end=month_end,
            information_system=identity["information_system"],
            fetch_function=identity["fetch_function"],
            process_function=identity["process_function"],
            raw_path=str(raw_dir / "raw.rds"),
            processed_path=str(processed_dir / "processed.parquet"),
            raw_sha256="",
            processed_sha256="",
            row_counts={},
            column_lists={},
            started_at=now,
            ended_at=now,
            duration_seconds=0.0,
            rscript_path=str(config.get("rscript_path", "Rscript")),
            r_version=None,
            microdatasus_version=None,
            read_dbc_version=None,
            stdout_path=str(raw_dir / "stdout.log"),
            stderr_path=str(raw_dir / "stderr.log"),
            heartbeat_path=str(raw_dir / "heartbeat.json"),
            exit_code=41,
            status="blocked",
            error_message="not_executed",
            request_hash=request_hash,
        )


    def build_datasus_manifests(
        *,
        system: str,
        uf: str,
        years: str,
        config: dict[str, Any] | None = None,
        data_root: str | Path = "data",
    ) -> list[DATASUSRequestManifest]:
        config = config or load_datasus_config()
        system = normalize_system(system)
        uf = normalize_uf(uf)
        parsed_years = parse_years(years)

        # Slice 1B implements deterministic UF × year manifests.
        # SIH-RD and CNES-ST month-level chunking will be introduced when their
        # source-specific slices require it.
        return [
            build_datasus_request_manifest(
                system=system,
                uf=uf,
                year_start=year,
                year_end=year,
                month_start=None,
                month_end=None,
                config=config,
                data_root=data_root,
            )
            for year in parsed_years
        ]


    def manifest_output_path(
        manifest: DATASUSRequestManifest,
        *,
        root: str | Path = "data/manifests/datasus",
    ) -> Path:
        period = period_label(
            year_start=manifest.year_start,
            year_end=manifest.year_end,
            month_start=manifest.month_start,
            month_end=manifest.month_end,
        )
        return (
            Path(root)
            / manifest.system
            / f"uf={manifest.uf}"
            / f"period={period}"
            / manifest.request_hash
            / "manifest.json"
        )


    def write_request_manifest(
        manifest: DATASUSRequestManifest,
        *,
        root: str | Path = "data/manifests/datasus",
    ) -> Path:
        path = manifest_output_path(manifest, root=root)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(manifest.model_dump(mode="json"), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return path


    def read_request_manifest(path: str | Path) -> DATASUSRequestManifest:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return DATASUSRequestManifest.model_validate(payload)
    ''')

    write("src/pegasus/datasus/subprocess.py", r'''
    from __future__ import annotations

    import json
    import os
    import shutil
    import subprocess
    import time
    from dataclasses import dataclass
    from pathlib import Path
    from typing import Any

    from pegasus.core.config import load_yaml
    from pegasus.core.hashing import sha256_file
    from pegasus.core.schemas import DATASUSRequestManifest
    from pegasus.datasus.cache import DatasusCache
    from pegasus.datasus.manifests import utc_now


    @dataclass(frozen=True)
    class DatasusConfig:
        rscript_path: str = "Rscript"
        r_timeout_seconds: int = 7200
        heartbeat_timeout_seconds: int = 900

        @classmethod
        def from_mapping(cls, payload: dict[str, Any]) -> "DatasusConfig":
            return cls(
                rscript_path=str(payload.get("rscript_path", "Rscript")),
                r_timeout_seconds=int(payload.get("r_timeout_seconds", 7200)),
                heartbeat_timeout_seconds=int(payload.get("heartbeat_timeout_seconds", 900)),
            )

        @classmethod
        def from_file(cls, path: str | Path = "config/datasus.yaml") -> "DatasusConfig":
            data = load_yaml(path)
            return cls.from_mapping(data.get("datasus", data))


    def _duration(started: float) -> float:
        return round(time.time() - started, 6)


    def _finish(
        request: DATASUSRequestManifest,
        *,
        started: float,
        status: str,
        exit_code: int,
        error_message: str | None,
        updates: dict[str, Any] | None = None,
    ) -> DATASUSRequestManifest:
        payload = {
            "status": status,
            "exit_code": exit_code,
            "error_message": error_message,
            "ended_at": utc_now(),
            "duration_seconds": _duration(started),
        }
        if updates:
            payload.update(updates)
        return request.model_copy(update=payload)


    def _kill_process_tree(process: subprocess.Popen) -> None:
        if process.poll() is not None:
            return

        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        else:
            process.kill()


    def _heartbeat_stale(path: Path, timeout_seconds: int) -> bool:
        if not path.exists():
            return False
        return time.time() - path.stat().st_mtime > timeout_seconds


    def fetch_datasus_chunk(
        request: DATASUSRequestManifest,
        *,
        config: DatasusConfig,
        cache: DatasusCache,
        timeout_seconds: int,
        heartbeat_timeout_seconds: int,
    ) -> DATASUSRequestManifest:
        started = time.time()
        request = request.model_copy(update={"started_at": utc_now(), "rscript_path": config.rscript_path})

        rscript = shutil.which(config.rscript_path) or config.rscript_path
        if shutil.which(config.rscript_path) is None and not Path(config.rscript_path).exists():
            return _finish(
                request,
                started=started,
                status="blocked",
                exit_code=41,
                error_message=f"Rscript not found: {config.rscript_path}",
            )

        script = Path(__file__).parent / "r_scripts" / "fetch_process_microdatasus.R"
        if not script.exists():
            return _finish(
                request,
                started=started,
                status="blocked",
                exit_code=11,
                error_message=f"R bridge script missing: {script}",
            )

        raw_path = Path(request.raw_path)
        processed_path = Path(request.processed_path)
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        processed_path.parent.mkdir(parents=True, exist_ok=True)

        heartbeat_path = Path(request.heartbeat_path)
        stdout_path = Path(request.stdout_path)
        stderr_path = Path(request.stderr_path)

        command = [
            rscript,
            str(script),
            "--system",
            request.system,
            "--uf",
            request.uf,
            "--year-start",
            str(request.year_start),
            "--year-end",
            str(request.year_end),
            "--raw-path",
            str(raw_path),
            "--processed-path",
            str(processed_path),
            "--out-dir",
            str(raw_path.parent),
        ]
        if request.month_start is not None:
            command.extend(["--month-start", str(request.month_start)])
        if request.month_end is not None:
            command.extend(["--month-end", str(request.month_end)])

        cache.write_request(request.request_hash, request.model_dump(mode="json"))

        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
            process = subprocess.Popen(command, stdout=stdout, stderr=stderr)
            while process.poll() is None:
                elapsed = time.time() - started
                if elapsed > timeout_seconds or _heartbeat_stale(heartbeat_path, heartbeat_timeout_seconds):
                    _kill_process_tree(process)
                    return _finish(
                        request,
                        started=started,
                        status="timeout",
                        exit_code=41,
                        error_message="R subprocess timed out or heartbeat became stale.",
                    )
                time.sleep(1.0)

        manifest_path = raw_path.parent / "manifest.json"

        if process.returncode != 0:
            return _finish(
                request,
                started=started,
                status="failed",
                exit_code=int(process.returncode or 60),
                error_message=f"R subprocess failed with exit code {process.returncode}.",
            )

        if not manifest_path.exists():
            return _finish(
                request,
                started=started,
                status="failed",
                exit_code=50,
                error_message="R subprocess returned success but manifest.json is missing.",
            )

        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            actual_raw_path = Path(payload.get("raw_path", request.raw_path))
            actual_processed_path = Path(payload.get("processed_path", request.processed_path))

            if not actual_raw_path.exists() or not actual_processed_path.exists():
                return _finish(
                    request,
                    started=started,
                    status="failed",
                    exit_code=50,
                    error_message="R manifest exists but raw or processed artifact is missing.",
                )

            return _finish(
                request,
                started=started,
                status="success",
                exit_code=0,
                error_message=None,
                updates={
                    "raw_path": str(actual_raw_path),
                    "processed_path": str(actual_processed_path),
                    "raw_sha256": sha256_file(actual_raw_path),
                    "processed_sha256": sha256_file(actual_processed_path),
                    "row_counts": payload.get("row_counts", {}),
                    "column_lists": payload.get("column_lists", {}),
                    "r_version": payload.get("r_version"),
                    "microdatasus_version": payload.get("microdatasus_version"),
                    "read_dbc_version": payload.get("read_dbc_version"),
                },
            )
        except Exception as exc:
            return _finish(
                request,
                started=started,
                status="failed",
                exit_code=50,
                error_message=f"Invalid R manifest: {exc}",
            )
    ''')

    write("src/pegasus/cli.py", r'''
    from __future__ import annotations

    import importlib.util
    import shutil
    import sys
    from pathlib import Path

    import typer
    from rich import print

    from pegasus.core.config import validate_config_tree
    from pegasus.core.paths import ensure_data_lake
    from pegasus.datasus.cache import DatasusCache
    from pegasus.datasus.manifests import (
        build_datasus_manifests,
        load_datasus_config,
        read_request_manifest,
        write_request_manifest,
    )
    from pegasus.datasus.profile import profile_table
    from pegasus.datasus.schema_compare import compare_profiles
    from pegasus.datasus.subprocess import DatasusConfig, fetch_datasus_chunk
    from pegasus.output.bundle import create_empty_output_bundle
    from pegasus.output.validate import validate_output_bundle
    from pegasus.registries.validators import validate_registry_tree

    app = typer.Typer(no_args_is_help=True)
    registries_app = typer.Typer(no_args_is_help=True)
    sidra_app = typer.Typer(no_args_is_help=True)
    datasus_app = typer.Typer(no_args_is_help=True)

    app.add_typer(registries_app, name="registries")
    app.add_typer(sidra_app, name="sidra")
    app.add_typer(datasus_app, name="datasus")


    def _fail(errors: list[str]) -> None:
        for error in errors:
            print(f"[red]ERROR[/red] {error}")
        raise typer.Exit(1)


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

        for key, value in checks.items():
            color = "green" if value not in {"missing", "False"} and not str(value).startswith("failed") else "yellow"
            print(f"[{color}]{key}[/] {value}")

        print("[yellow]doctor is still light: no DATASUS/SIDRA ingestion is executed.[/yellow]")


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


    @sidra_app.command("metadata")
    def sidra_metadata(tables: Path = typer.Option(..., "--tables")) -> None:
        print(f"[yellow]blocked[/yellow] SIDRA metadata is Slice 2. Seed received: {tables}")
        raise typer.Exit(2)


    @sidra_app.command("plan")
    def sidra_plan(view: str = typer.Option(..., "--view")) -> None:
        print(f"[yellow]blocked[/yellow] SIDRA planning is Slice 2. View received: {view}")
        raise typer.Exit(2)


    @sidra_app.command("extract")
    def sidra_extract(plan: Path = typer.Option(..., "--plan")) -> None:
        print(f"[yellow]blocked[/yellow] SIDRA extraction is Slice 2. Plan received: {plan}")
        raise typer.Exit(2)


    @app.command()
    def compile(intent: Path = typer.Option(..., "--intent")) -> None:
        print(f"[yellow]blocked[/yellow] compile workflow requires later slices. Intent: {intent}")
        raise typer.Exit(2)
    ''')

    write("tests/unit/test_datasus_manifests.py", r'''
    from pathlib import Path

    from pegasus.datasus.manifests import (
        build_datasus_manifests,
        build_datasus_request_manifest,
        parse_years,
        read_request_manifest,
        write_request_manifest,
    )


    def test_parse_years_single_range_and_list():
        assert parse_years("2022") == [2022]
        assert parse_years("2020-2022") == [2020, 2021, 2022]
        assert parse_years("2020,2022") == [2020, 2022]


    def test_build_sim_manifest_paths_are_content_addressed(tmp_path: Path):
        manifest = build_datasus_request_manifest(
            system="SIM-DO",
            uf="AL",
            year_start=2022,
            year_end=2022,
            config={"rscript_path": "Rscript"},
            data_root=tmp_path / "data",
        )

        assert manifest.system == "SIM-DO"
        assert manifest.uf == "AL"
        assert manifest.year_start == 2022
        assert manifest.raw_path.endswith(f"{manifest.request_hash}\\raw.rds") or manifest.raw_path.endswith(f"{manifest.request_hash}/raw.rds")
        assert "data" in manifest.raw_path
        assert "raw" in manifest.raw_path
        assert "processed" in manifest.processed_path


    def test_build_yearly_manifests_for_range(tmp_path: Path):
        manifests = build_datasus_manifests(
            system="SIM-DO",
            uf="AL",
            years="2021-2022",
            config={"rscript_path": "Rscript"},
            data_root=tmp_path / "data",
        )
        assert [m.year_start for m in manifests] == [2021, 2022]
        assert len({m.request_hash for m in manifests}) == 2


    def test_manifest_write_read_roundtrip(tmp_path: Path):
        manifest = build_datasus_request_manifest(
            system="SIM-DO",
            uf="AL",
            year_start=2022,
            year_end=2022,
            config={"rscript_path": "Rscript"},
            data_root=tmp_path / "data",
        )

        path = write_request_manifest(manifest, root=tmp_path / "manifests")
        loaded = read_request_manifest(path)

        assert loaded.request_hash == manifest.request_hash
        assert loaded.system == "SIM-DO"
    ''')

    write("tests/unit/test_datasus_subprocess_boundary.py", r'''
    from pathlib import Path

    from pegasus.datasus.cache import DatasusCache
    from pegasus.datasus.manifests import build_datasus_request_manifest
    from pegasus.datasus.subprocess import DatasusConfig, fetch_datasus_chunk


    def test_fetch_datasus_chunk_blocks_when_rscript_missing(tmp_path: Path):
        manifest = build_datasus_request_manifest(
            system="SIM-DO",
            uf="AL",
            year_start=2022,
            year_end=2022,
            config={"rscript_path": "__definitely_missing_Rscript__"},
            data_root=tmp_path / "data",
        )

        result = fetch_datasus_chunk(
            manifest,
            config=DatasusConfig(
                rscript_path="__definitely_missing_Rscript__",
                r_timeout_seconds=1,
                heartbeat_timeout_seconds=1,
            ),
            cache=DatasusCache(tmp_path / "cache"),
            timeout_seconds=1,
            heartbeat_timeout_seconds=1,
        )

        assert result.status == "blocked"
        assert result.exit_code == 41
        assert "Rscript not found" in (result.error_message or "")
        assert result.duration_seconds >= 0
    ''')

    write("tests/unit/test_cli_datasus_manifest.py", r'''
    from typer.testing import CliRunner

    from pegasus.cli import app


    def test_datasus_ingest_dry_run_plans_manifest():
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "datasus",
                "ingest",
                "--system",
                "SIM-DO",
                "--uf",
                "AL",
                "--years",
                "2022",
                "--dry-run",
            ],
        )

        assert result.exit_code == 0
        assert "planned" in result.output
        assert "SIM-DO" in result.output
    ''')

    print("Applied Slice 1B: DATASUS manifest builder, subprocess hardening, CLI ingest/profile wiring, and tests.")


if __name__ == "__main__":
    main()