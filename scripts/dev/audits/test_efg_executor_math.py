import tempfile
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
import polars as pl
from pegasus.efg.executor import _compute_rn_ratio, _compute_bridge_tensor
from pegasus.core.schemas import FieldNode, Lineage
from pegasus.core.enums import MaterializationState, FieldState

def mock_field_node(fid: str, path: str) -> FieldNode:
    return FieldNode(
        id=fid, name=fid, kind="extensive_measure", carrier="mock", unit="count",
        support={}, axes={}, aggregation="additive", role=[], source=[], operator=None,
        provenance=[], state=FieldState.verified, warnings=[], 
        lineage=Lineage(parent_ids=[], operator_type="mock", operator_params={}, registry_versions={}, code_version="1.0"),
        materialization_state=MaterializationState.materialized, path=path
    )

def test_math():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        num_path = tmp / "num.parquet"
        den_path = tmp / "den.parquet"
        out_dir = tmp / "out"
        out_dir.mkdir()
        
        # Create Dummy Data
        # Numerator: 2020/cod1 = 10.0, 2020/cod2 = 50.0
        num_df = pl.DataFrame({"year": [2020, 2020], "municipality_cod6": ["111", "222"], "value": [10.0, 50.0]})
        num_df.write_parquet(num_path)
        
        # Denominator: 2020/cod1 = 100.0, 2020/cod2 = 0.0 (test zero handling)
        den_df = pl.DataFrame({"year": [2020, 2020], "municipality_cod6": ["111", "222"], "value": [100.0, 0.0]})
        den_df.write_parquet(den_path)

        f_num = mock_field_node("num", str(num_path))
        f_den = mock_field_node("den", str(den_path))
        f_rn = mock_field_node("rn", "").model_copy(update={"lineage": Lineage(parent_ids=["num", "den"], operator_type="RN", operator_params={}, registry_versions={}, code_version="1.0")})
        
        parents = {"num": f_num, "den": f_den}
        
        # Test 1: RN Math
        rn_out, _ = _compute_rn_ratio(f_rn, parents, out_dir)
        res_df = pl.read_parquet(rn_out)
        res_dict = {row["municipality_cod6"]: row["value"] for row in res_df.to_dicts()}
        
        assert res_dict["111"] == 0.1, f"RN math failed: expected 0.1, got {res_dict['111']}"
        assert res_dict["222"] is None, f"RN zero handling failed: expected None, got {res_dict['222']}"
        print("RN Math Test: PASSED")

        # Test 2: Bridge Divergence
        f_bridge = mock_field_node("bridge", "").model_copy(update={"kind": "bridge_divergence", "lineage": Lineage(parent_ids=["num", "den"], operator_type="divergence", operator_params={}, registry_versions={}, code_version="1.0")})
        bridge_out, _ = _compute_bridge_tensor(f_bridge, parents, out_dir)
        bridge_df = pl.read_parquet(bridge_out)
        bridge_dict = {row["municipality_cod6"]: row["value"] for row in bridge_df.to_dicts()}
        
        import math
        eps = 1e-9
        expected = math.log((10.0 + eps) / (100.0 + eps))
        assert math.isclose(bridge_dict["111"], expected, rel_tol=1e-5), f"Bridge divergence failed: got {bridge_dict['111']}"
        print("Bridge Divergence Test: PASSED")

        # Test 3: OOM Guard (No intersecting keys)
        bad_den_df = pl.DataFrame({"different_year": [2020], "value": [100.0]})
        bad_den_path = tmp / "bad_den.parquet"
        bad_den_df.write_parquet(bad_den_path)
        f_bad_den = mock_field_node("bad_den", str(bad_den_path))
        parents_bad = {"num": f_num, "bad_den": f_bad_den}
        
        try:
            _compute_rn_ratio(f_rn.model_copy(update={"lineage": Lineage(parent_ids=["num", "bad_den"], operator_type="RN", operator_params={}, registry_versions={}, code_version="1.0")}), parents_bad, out_dir)
            raise AssertionError("OOM Guard failed to raise RuntimeError on cross-join.")
        except RuntimeError as e:
            if "intersecting support axis" in str(e):
                print("OOM Guard Test: PASSED")
            else:
                raise e

if __name__ == "__main__":
    test_math()
