import wave

import numpy as np
import pytest

from errors import AudioTooLongError, InvalidMediaError
from media import SAMPLE_RATE, decode_media


def write_wav(path, duration_seconds):
    """Generate a valid stereo WAV without external sample files."""

    source_rate = 8_000
    sample_count = int(source_rate * duration_seconds)

    # Stereo silence is sufficient to test decoding and resampling.
    samples = np.zeros((sample_count, 2), dtype="<i2")

    with wave.open(str(path), "wb") as output:
        output.setnchannels(2)
        output.setsampwidth(2)
        output.setframerate(source_rate)
        output.writeframes(samples.tobytes())


def test_decode_resamples_to_mono_float32(tmp_path):
    path = tmp_path / "sample.wav"
    write_wav(path, duration_seconds=0.5)

    audio = decode_media(path, max_duration_seconds=1)

    # Verify the format passed to the transcription model.
    assert audio.ndim == 1
    assert audio.dtype == np.float32
    assert len(audio) == SAMPLE_RATE // 2
    assert np.isfinite(audio).all()


def test_duration_limit_accepts_exact_boundary(tmp_path):
    path = tmp_path / "boundary.wav"
    write_wav(path, duration_seconds=1)

    audio = decode_media(path, max_duration_seconds=1)

    assert len(audio) == SAMPLE_RATE


def test_duration_limit_rejects_long_audio(tmp_path):
    path = tmp_path / "too-long.wav"
    write_wav(path, duration_seconds=2)

    # Exercise real decoding rather than simulating the duration exception.
    with pytest.raises(AudioTooLongError):
        decode_media(path, max_duration_seconds=1)


def test_invalid_media(tmp_path):
    path = tmp_path / "invalid.txt"
    path.write_text("This is not audio.", encoding="utf-8")

    with pytest.raises(InvalidMediaError):
        decode_media(path, max_duration_seconds=1)