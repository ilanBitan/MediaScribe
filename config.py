import os
from dataclasses import dataclass


def positive_integer(name: str, default: int) -> int:
    """Read a positive integer from the environment."""
    raw_value = os.getenv(name, str(default))

    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer.") from exc

    # Fail at startup rather than accepting an unsafe configuration.
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero.")

    return value


@dataclass(frozen=True)
class Settings:
    """Keep validated configuration immutable during application execution."""

    model_id: str
    max_upload_bytes: int
    max_audio_seconds: int


def load_settings() -> Settings:
    model_id = os.getenv(
        "TRANSCRIPTION_MODEL",
        "Systran/faster-whisper-small",
    ).strip()

    if not model_id:
        raise ValueError("TRANSCRIPTION_MODEL cannot be empty.")

    return Settings(
        model_id=model_id,
        max_upload_bytes=positive_integer(
            "MAX_UPLOAD_BYTES",
            25 * 1024 * 1024,
        ),
        max_audio_seconds=positive_integer(
            "MAX_AUDIO_SECONDS",
            300,
        ),
    )