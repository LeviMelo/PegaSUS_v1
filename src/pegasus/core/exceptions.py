from __future__ import annotations


class PegasusError(Exception):
    """Base class for PegaSUS errors."""


class ConfigurationError(PegasusError):
    """Raised when project configuration is invalid."""


class RegistryError(PegasusError):
    """Raised when registry loading or validation fails."""


class DataContractError(PegasusError):
    """Raised when a data artifact violates its contract."""


class ExternalToolError(PegasusError):
    """Raised when an external tool such as Rscript fails."""


class ModuleBlockedError(PegasusError):
    """Raised when a blueprint-locked module is present but intentionally blocked."""


class IllegalTransformationError(PegasusError):
    """Raised when a transformation violates the legality predicate."""