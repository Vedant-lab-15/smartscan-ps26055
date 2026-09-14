"""Custom exceptions for the EW Smart Scan system."""


class MemoryGuardError(RuntimeError):
    """Raised when estimated pulse count would exceed the configured memory guard."""


class CredentialError(RuntimeError):
    """Raised when HF_TOKEN environment variable is not set."""


class CrossFileEmitterError(ValueError):
    """Raised when an emitter label is looked up across different .h5 files."""
