class MediaError(Exception):
    """Base class for expected media input failures."""


class InvalidMediaError(MediaError):
    """The file cannot be decoded into usable audio."""


class NoAudioError(MediaError):
    """The media container does not contain an audio stream."""


class AudioTooLongError(MediaError):
    """Decoded audio exceeds the configured duration limit."""