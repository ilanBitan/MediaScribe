from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from faster_whisper import WhisperModel

from config import load_settings
from media import SAMPLE_RATE, decode_media


@dataclass(frozen=True)
class TranscriptionResult:
    """Represent output independently of HTTP and presentation."""

    text: str
    language: str
    language_probability: float
    audio_duration_seconds: float
    processing_time_seconds: float


class TranscriptionService:
    """Reuse one model instance across transcription calls."""

    def __init__(
        self,
        model_id: str | None = None,
        max_audio_seconds: int | None = None,
    ) -> None:
        settings = load_settings()

        self._max_audio_seconds = (
            settings.max_audio_seconds
            if max_audio_seconds is None
            else max_audio_seconds
        )

        if self._max_audio_seconds <= 0:
            raise ValueError("max_audio_seconds must be greater than zero.")

        # Keep model loading outside individual transcription calls.
        self._model = WhisperModel(
            model_id if model_id is not None else settings.model_id,
            device="cpu",
            compute_type="int8",
        )

    def transcribe(
        self,
        audio_path: Path,
        language: str | None = None,
    ) -> TranscriptionResult:
        """Decode bounded audio and run transcription locally."""

        if not audio_path.is_file():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        started = perf_counter()

        # Decode once and enforce duration before running expensive inference.
        audio = decode_media(audio_path, self._max_audio_seconds)

        # Passing an array avoids decoding the file a second time.
        segments, info = self._model.transcribe(
            audio,
            language=language,
            beam_size=5,
        )

        # Consume the lazy generator before stopping the processing timer.
        parts = []

        for segment in segments:
            text = segment.text.strip()
            if text:
                parts.append(text)

        return TranscriptionResult(
            text=" ".join(parts),
            language=info.language,
            language_probability=info.language_probability,
            audio_duration_seconds=len(audio) / SAMPLE_RATE,
            processing_time_seconds=perf_counter() - started,
        )