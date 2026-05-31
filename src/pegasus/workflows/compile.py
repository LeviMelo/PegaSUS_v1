from __future__ import annotations

import uuid
from pathlib import Path

from loguru import logger

from pegasus.core.config import load_config
from pegasus.output.bundle import OutputBundle
from pegasus.registries.loader import load_registry_set
from pegasus.registries.validators import validate_registry_set


def compile_run(intent_path: str | Path, root: str | Path = ".") -> Path:
    """Create the first architecture-valid run bundle.

    This is not yet the full epidemiological compiler. It is the run-bundle
    scaffold that later modules will populate.
    """
    root = Path(root).resolve()
    intent_path = Path(intent_path)

    config = load_config(root)

    registry_set = load_registry_set(root / "config" / "registries")
    validate_registry_set(registry_set)

    run_id = f"run_{uuid.uuid4().hex[:12]}"
    run_dir = root / config.paths.runs / run_id

    logger.info("Creating PegaSUS run bundle: {}", run_dir)

    bundle = OutputBundle(run_dir)
    bundle.initialize()

    if intent_path.exists():
        import json

        with intent_path.open("r", encoding="utf-8") as f:
            intent_payload = json.load(f)
    else:
        intent_payload = {
            "warning": f"Intent file not found at {str(intent_path)}; placeholder intent used."
        }

    bundle.write_user_intent(intent_payload)
    bundle.write_run_config(config.model_dump(mode="json"))
    bundle.write_p_vector({})
    bundle.write_reproducibility_manifest(
        run_id,
        registry_hashes=registry_set.hashes(),
    )
    bundle.validate_minimal()

    logger.info("Run bundle created and validated: {}", run_dir)
    return run_dir