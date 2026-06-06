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
