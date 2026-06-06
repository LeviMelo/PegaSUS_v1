from __future__ import annotations

from pathlib import Path
import re
import textwrap

ROOT = Path.cwd()


def write(path: str, content: str) -> None:
    target = ROOT / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(textwrap.dedent(content).lstrip(), encoding="utf-8", newline="\n")


def patch_cli() -> None:
    path = ROOT / "src" / "pegasus" / "cli.py"
    text = path.read_text(encoding="utf-8")

    if "metadata_dir_hash" not in text.split("from pegasus.sidra.metadata import", 1)[1].split(")", 1)[0]:
        text = text.replace(
            "    fixture_sidra_metadata,\n",
            "    fixture_sidra_metadata,\n    metadata_dir_hash,\n",
            1,
        )

    new_function = '''@sidra_app.command("extract")
def sidra_extract(
    plan: Path = typer.Option(..., "--plan"),
    metadata_dir: Path = typer.Option(Path("data/metadata/sidra/normalized"), "--metadata-dir"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    concurrency: int | None = typer.Option(None, "--concurrency"),
    log_path: Path | None = typer.Option(None, "--log"),
) -> None:
    sidra_cfg = _sidra_runtime_config()
    chunks = read_chunk_plan(plan)
    metadata_hash = metadata_dir_hash(metadata_dir)

    if dry_run:
        total = sum(c.estimated_cells for c in chunks)
        print(f"[cyan]dry-run[/cyan] chunks={len(chunks)} estimated_cells={total} metadata_hash={metadata_hash}")
        return

    client = SidraClient()
    results = extract_chunk_plan(
        chunks,
        client=client,
        concurrency=concurrency or int(sidra_cfg.get("concurrency", 4)),
        metadata_hash=metadata_hash,
    )
    log_path = log_path or Path("data/diagnostics/sidra") / f"{plan.stem}.extraction_log.json"
    write_extraction_log(results, output_path=log_path)

    failures = [r for r in results if r.status != "success"]
    print(f"[green]extract complete[/green] chunks={len(results)} failures={len(failures)} log={log_path}")
    if failures:
        raise typer.Exit(1)
'''

    pattern = r'@sidra_app\.command\("extract"\)\ndef sidra_extract\([\s\S]*?\n\n\n@app\.command\(\)'
    replacement = new_function + "\n\n@app.command()"

    text, n = re.subn(pattern, replacement, text, count=1)
    if n != 1:
        raise RuntimeError("Could not replace sidra_extract function by decorator boundary.")

    path.write_text(text, encoding="utf-8", newline="\n")


def patch_audit() -> None:
    path = ROOT / "scripts" / "dev" / "audit_slice2b_sidra_outputs.py"
    text = path.read_text(encoding="utf-8")

    if "content hash of normalized SIDRA metadata" in text:
        return

    marker = '''    for expected_fragment in ['"2","6794"', '"86","95251"', '"287","100362"']:
        compact = str(cat_tuple).replace(" ", "")
        if expected_fragment not in compact:
            hard_failures.append(("category_tuple", f"contains {expected_fragment}", cat_tuple))
'''

    insertion = marker + '''
    metadata_hash = row.get("metadata_hash")
    if metadata_hash in {None, "", "metadata_unset"}:
        hard_failures.append(("metadata_hash", "content hash of normalized SIDRA metadata", metadata_hash))
    elif len(str(metadata_hash)) != 64:
        hard_failures.append(("metadata_hash", "64-character sha256", metadata_hash))
'''

    if marker not in text:
        raise RuntimeError("Could not find audit category_tuple check marker.")

    text = text.replace(marker, insertion, 1)
    path.write_text(text, encoding="utf-8", newline="\n")


def main() -> None:
    patch_cli()
    patch_audit()

    write("tests/unit/test_sidra_metadata_hash.py", r'''
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
    ''')

    print("Resumed Slice 2B metadata_hash patch against actual current cli.py.")


if __name__ == "__main__":
    main()