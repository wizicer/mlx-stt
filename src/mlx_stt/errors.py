"""User-facing exceptions for mlx-stt."""


class MlxSttError(Exception):
    """Base class for errors that should be shown without a traceback."""


class AudioPreparationError(MlxSttError):
    """Input audio cannot be decoded or converted."""


class ModelResolutionError(MlxSttError):
    """A required model cannot be found or downloaded."""
