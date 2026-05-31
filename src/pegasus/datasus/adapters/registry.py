from __future__ import annotations

from pegasus.core.exceptions import ConfigurationError
from pegasus.datasus.adapters.base import DatasusSourceAdapter
from pegasus.datasus.adapters.sim_do import SimDoAdapter


_ADAPTERS: dict[str, DatasusSourceAdapter] = {
    "SIM-DO": SimDoAdapter(),
}


def get_datasus_adapter(source_system: str) -> DatasusSourceAdapter:
    adapter = _ADAPTERS.get(source_system)

    if adapter is None:
        raise ConfigurationError(
            f"No DATASUS adapter registered for source system: {source_system}"
        )

    return adapter


def registered_datasus_systems() -> tuple[str, ...]:
    return tuple(sorted(_ADAPTERS))