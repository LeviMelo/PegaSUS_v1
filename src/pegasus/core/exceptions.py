class PegasusError(Exception):
    """Base PegaSUS exception."""


class ConfigError(PegasusError):
    """Configuration failure."""


class RegistryValidationError(PegasusError):
    """Registry validation failure."""


class OutputValidationError(PegasusError):
    """Output bundle validation failure."""


class BlockedModuleError(PegasusError):
    """Raised when an architecturally visible but inactive module is invoked."""

    def __init__(self, *, module: str, reason: str = "blocked_state"):
        self.module = module
        self.reason = reason
        super().__init__(f"{module} is blocked: {reason}")


class ComputeBackendError(PegasusError):
    """Base failure for centralized numerical backend planning."""


class CUDARequiredError(ComputeBackendError):
    """Raised when policy requires CUDA and no CUDA device is available."""


class MemoryPreflightError(ComputeBackendError):
    """Raised when a numerical task exceeds configured memory policy."""


class StorageBackendError(PegasusError):
    """Raised when a configured storage backend or schema operation is unavailable."""
