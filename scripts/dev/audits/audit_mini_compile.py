import json
import shutil
import tempfile
import hashlib
from pathlib import Path
import polars as pl
from pegasus.workflows.compile import run_compile

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def build_mini_audit():
    print("Starting Local Mini-Compile Audit V2...")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        data_dir = tmp / "data"
        raw_dir = data_dir / "raw"
        raw_dir.mkdir(parents=True)
        
        sim_df = pl.DataFrame({"year": [2022, 2022], "mun_residence_cod6": ["270430", "270430"], "death_date": ["2022-01-01", "2022-02-01"], "race_color_admin": ["1", "2"]})
        sinasc_df = pl.DataFrame({"year": [2022, 2022], "mun_residence_cod6": ["270430", "270430"], "birth_date": ["2022-01-05", "2022-02-05"], "low_birth_weight_flag": [False, True]})
        sidra_df = pl.DataFrame({
            "table_id": ["9606", "9606"], "variable_id": ["93", "93"], "period": ["2022", "2022"],
            "locality_level": ["N6", "N6"], "locality_id": ["2704302", "2704302"], 
            "classification_tuple": [json.dumps([("2", "6794"), ("86", "95251"), ("287", "100362")]), json.dumps([("2", "6794"), ("86", "95251"), ("287", "100362")])],
            "category_tuple": [json.dumps([("2", "6794"), ("86", "95251"), ("287", "100362")]), json.dumps([("2", "6794"), ("86", "95251"), ("287", "100362")])],
            "value_raw": ["1000", "1000"], "value_numeric": [1000.0, 1000.0], "value_status": ["numeric", "numeric"], "unit": ["Pessoas", "Pessoas"],
            "request_hash": ["mock", "mock"], "metadata_hash": ["mock", "mock"], "fetched_at": ["2026-01-01", "2026-01-01"]
        })
        
        sim_path = raw_dir / "sim.parquet"
        sinasc_path = raw_dir / "sinasc.parquet"
        sidra_path = raw_dir / "sidra.parquet"
        
        sim_df.write_parquet(sim_path)
        sinasc_df.write_parquet(sinasc_path)
        sidra_df.write_parquet(sidra_path)
        
        manifest_hash = "mock_manifest_hash_12345"
        
        manifest = {
            "schema_version": "1.0",
            "manifest_id": "mini_audit",
            "compile_source_mode": "materialized_external",
            "artifact_count": 3,
            "artifacts": [
                {"path": str(sim_path), "source_system": "SIM-DO", "artifact_role": "processed_events", "provenance_mode": "materialized_external", "content_hash": sha256_file(sim_path), "source_manifest_hash": manifest_hash},
                {"path": str(sinasc_path), "source_system": "SINASC", "artifact_role": "processed_events", "provenance_mode": "materialized_external", "content_hash": sha256_file(sinasc_path), "source_manifest_hash": manifest_hash},
                {"path": str(sidra_path), "source_system": "SIDRA", "artifact_role": "normalized_facts", "provenance_mode": "materialized_external", "content_hash": sha256_file(sidra_path), "source_manifest_hash": manifest_hash}
            ]
        }
        manifest_path = tmp / "source_manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        
        intent = {
            "geography": {"level": "municipality", "codes": ["2704302"], "uf": []},
            "time": {"start_year": 2022, "end_year": 2022},
            "budget": "fast",
            "geo_mode": "native",
            "execution_scale": "smoke",
            "context_policy": ["pirs", "hsic"],
            "mandatory_fields": [],
            "population_mode": "official_sidra_anchor"
        }
        intent_path = tmp / "intent.json"
        intent_path.write_text(json.dumps(intent), encoding="utf-8")
        
        run_dir = tmp / "run_output"
        
        print(f"Executing run_compile()...")
        try:
            result = run_compile(
                intent_path=intent_path,
                run_dir=run_dir,
                data_root=data_dir,
                source_manifest=manifest_path,
                require_materialized_external=True
            )
            
            print(f"\nCompile Status: {result['status']}")
            validation = result.get('validation')
            if validation and validation.ok:
                print("17-Key Bundle Validation: PASSED")
                print("Local Mini-Compile Audit: SUCCESS")
            else:
                print("17-Key Bundle Validation: FAILED")
                if validation:
                    for err in validation.errors:
                        print(f"  - {err}")
        except Exception as e:
            print(f"\nLocal Mini-Compile Audit FAILED with Exception:")
            raise e

if __name__ == "__main__":
    build_mini_audit()
