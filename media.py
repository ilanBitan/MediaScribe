from pathlib import Path

import av
import numpy as np
from av.error import FFmpegError

from errors import AudioTooLongError, InvalidMediaError, NoAudioError


SAMPLE_RATE = 16_000


def decode_media(
    path: Path,
    max_duration_seconds: int,
) -> np.ndarray:
    """Decode the first audio stream into bounded mono float32 audio."""

    chunks: list[np.ndarray] = []
    total_samples = 0
    max_samples = max_duration_seconds * SAMPLE_RATE

    def collect(frame: av.AudioFrame) -> None:
        nonlocal total_samples

        # Count actual decoded samples instead of trusting duration metadata.
        total_samples += frame.samples

        if total_samples > max_samples:
            raise AudioTooLongError(
                f"Audio exceeds the {max_duration_seconds}-second limit."
            )

        # Copy samples so their lifetime does not depend on the PyAV frame.
        chunks.append(frame.to_ndarray().reshape(-1).copy())

    try:
        # Decode media contents; filename extensions are not trusted.
        with av.open(str(path)) as container:
            if not container.streams.audio:
                raise NoAudioError("The file does not contain an audio stream.")

            # Explicit policy: process the first audio stream only.
            resampler = av.AudioResampler(
                format="fltp",
                layout="mono",
                rate=SAMPLE_RATE,
            )
            decoded_any_frame = False

            for frame in container.decode(audio=0):
                decoded_any_frame = True

                # Input timestamps may be discontinuous.
                # The model needs sequential samples, not original timestamps.
                frame.pts = None

                for output_frame in resampler.resample(frame):
                    collect(output_frame)

            # Flush buffered samples only after the resampler received input.
            if decoded_any_frame:
                for output_frame in resampler.resample(None):
                    collect(output_frame)

    except FFmpegError as exc:
        # Convert decoder failures only; inference errors are handled elsewhere.
        raise InvalidMediaError(
            "The file could not be decoded as supported audio."
        ) from exc

    if not chunks or total_samples == 0:
        raise InvalidMediaError("The audio stream contains no decodable samples.")

    return np.concatenate(chunks).astype(np.float32, copy=False)