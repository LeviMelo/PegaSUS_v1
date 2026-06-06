from pathlib import Path

from pegasus.core.config import validate_config_tree
from pegasus.output.bundle import create_empty_output_bundle
from pegasus.output.validate import validate_output_bundle
from pegasus.registries.validators import validate_registry_tree


def test_slice0_contract(tmp_path: Path):
    assert not validate_config_tree(".")
    assert not validate_registry_tree("config/registries")
    run_dir = create_empty_output_bundle(tmp_path / "run")
    result = validate_output_bundle(run_dir=str(run_dir))
    assert result.ok, result.errors
