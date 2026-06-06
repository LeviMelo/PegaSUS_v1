from pathlib import Path

from pegasus.sidra.metadata import fixture_sidra_metadata, metadata_dir_hash, write_normalized_metadata_tables


def test_metadata_dir_hash_is_stable_and_nonempty(tmp_path: Path):
    metadata = fixture_sidra_metadata()
    write_normalized_metadata_tables(metadata, output_dir=tmp_path)

    first = metadata_dir_hash(tmp_path)
    second = metadata_dir_hash(tmp_path)

    assert first == second
    assert len(first) == 64
    assert first != "metadata_unset"
