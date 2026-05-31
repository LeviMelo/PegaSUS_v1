from __future__ import annotations

from pathlib import Path
from typing import Literal

from pegasus.datasus.adapters.registry import get_datasus_adapter
from pegasus.datasus.artifacts import DatasusLocalWorkflowManifest, build_success_manifest
from pegasus.datasus.io import read_table, write_json, write_parquet
from pegasus.datasus.profile import profile_dataframe
from pegasus.datasus.schema_compare import compare_profiles


def process_datasus_local_files(
    *,
    source_system: str,
    raw_path: str | Path,
    out_root: str | Path,
    processed_path: str | Path | None = None,
    normalization_input_kind: Literal["raw", "processed"] = "raw",
) -> DatasusLocalWorkflowManifest:
    """Profile, compare, and normalize local DATASUS artifacts via source adapter."""
    adapter = get_datasus_adapter(source_system)

    raw_path = Path(raw_path).resolve()
    processed_path_resolved = None if processed_path is None else Path(processed_path).resolve()
    out_root = Path(out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    raw_df = read_table(raw_path)
    processed_df = read_table(processed_path_resolved) if processed_path_resolved else None

    raw_profile = profile_dataframe(
        raw_df,
        source_system=source_system,
        artifact_kind="raw",
    )
    raw_profile_path = out_root / "raw_profile.json"
    write_json(raw_profile_path, raw_profile)

    processed_profile_path: Path | None = None
    schema_comparison_path: Path | None = None

    if processed_df is not None:
        processed_profile = profile_dataframe(
            processed_df,
            source_system=source_system,
            artifact_kind="processed",
        )
        processed_profile_path = out_root / "processed_profile.json"
        write_json(processed_profile_path, processed_profile)

        comparison = compare_profiles(raw_profile, processed_profile)
        schema_comparison_path = out_root / "schema_comparison.json"
        write_json(schema_comparison_path, comparison)

    if normalization_input_kind == "processed":
        if processed_df is None:
            raise ValueError("normalization_input_kind='processed' requires processed_path.")
        input_df = processed_df
    else:
        input_df = raw_df

    first_pass = adapter.normalize(
        input_df,
        source_manifest_hash="pending_local_workflow_manifest",
    )

    normalized_path = out_root / f"{source_system.lower().replace('-', '_')}_normalized.parquet"
    write_parquet(first_pass.normalized, normalized_path)

    manifest = build_success_manifest(
        source_system=source_system,
        raw_path=raw_path,
        processed_path=processed_path_resolved,
        normalization_input_kind=normalization_input_kind,
        normalizer=f"{adapter.__class__.__name__}.normalize",
        raw_profile_path=raw_profile_path,
        processed_profile_path=processed_profile_path,
        schema_comparison_path=schema_comparison_path,
        normalized_path=normalized_path,
        n_raw_rows=raw_df.height,
        n_processed_rows=None if processed_df is None else processed_df.height,
        n_normalized_rows=first_pass.normalized.height,
        warnings=first_pass.warnings,
    )

    final_pass = adapter.normalize(
        input_df,
        source_manifest_hash=manifest.workflow_id,
    )
    write_parquet(final_pass.normalized, normalized_path)

    manifest_path = out_root / "manifest.json"
    write_json(manifest_path, manifest)

    return manifest


def process_sim_do_local_files(
    *,
    raw_path: str | Path,
    out_root: str | Path,
    processed_path: str | Path | None = None,
    normalization_input_kind: Literal["raw", "processed"] = "raw",
) -> DatasusLocalWorkflowManifest:
    return process_datasus_local_files(
        source_system="SIM-DO",
        raw_path=raw_path,
        processed_path=processed_path,
        out_root=out_root,
        normalization_input_kind=normalization_input_kind,
    )