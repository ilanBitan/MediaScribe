from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from threading import Event

import pytest
from fastapi.testclient import TestClient

import main as api
from errors import AudioTooLongError, InvalidMediaError, NoAudioError
from transcription import TranscriptionResult


class FakeTranscriptionService:
    """Control inference behavior without loading a real model."""

    def __init__(self):
        self.failure = None
        self.paths = []
        self.started = Event()
        self.release = Event()
        self.block = False

    def transcribe(self, audio_path: Path) -> TranscriptionResult:
        # Confirm that the API created a readable temporary file.
        assert audio_path.is_file()
        self.paths.append(audio_path)
        self.started.set()

        # Events allow a deterministic concurrency test without sleep().
        if self.block and not self.release.wait(timeout=10):
            raise RuntimeError("Timed out waiting for the test to release inference.")

        if self.failure is not None:
            raise self.failure

        return TranscriptionResult(
            text="Test transcript",
            language="en",
            language_probability=0.95,
            audio_duration_seconds=1.0,
            processing_time_seconds=0.1,
        )


@pytest.fixture
def context(monkeypatch):
    """Start the application with a fake service and isolated settings."""

    service = FakeTranscriptionService()

    # Patch the constructor before lifespan runs, avoiding model loading.
    monkeypatch.setattr(
        api,
        "TranscriptionService",
        lambda **kwargs: service,
    )

    # Keep tests independent of deployment environment configuration.
    monkeypatch.setattr(
        api,
        "settings",
        replace(api.settings, max_upload_bytes=32),
    )

    # The context manager runs application startup and shutdown.
    # Disable exception propagation to inspect actual HTTP 500 responses.
    with TestClient(api.app, raise_server_exceptions=False) as client:
        yield client, service


def upload(client, content=b"sample"):
    """Submit multipart data through the API rather than calling the route."""

    return client.post(
        "/transcriptions",
        files={"file": ("sample.wav", content, "audio/wav")},
    )


def assert_error(response, status, code):
    """Check the error contract and request correlation header."""

    assert response.status_code == status
    body = response.json()
    assert body["code"] == code
    assert isinstance(body["detail"], str)
    assert body["request_id"]
    assert body["request_id"] == response.headers["X-Request-ID"]


def test_health_and_readiness(context):
    client, _ = context

    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/ready").json() == {"status": "ready"}


def test_success_and_temporary_file_cleanup(context):
    client, service = context

    response = upload(client)

    assert response.status_code == 200
    assert response.json()["text"] == "Test transcript"
    assert response.headers["X-Request-ID"]

    # The service saw a file during processing; it must be gone afterward.
    assert service.paths
    assert all(not path.exists() for path in service.paths)


def test_missing_file(context):
    client, _ = context

    response = client.post("/transcriptions")

    assert_error(response, 422, "invalid_request")


def test_empty_file(context):
    client, service = context

    response = upload(client, b"")

    assert_error(response, 400, "empty_file")
    assert service.paths == []


def test_oversized_file(context):
    client, service = context

    # The fixture sets the upload limit to 32 bytes.
    response = upload(client, b"x" * 33)

    assert_error(response, 413, "file_too_large")
    assert service.paths == []


@pytest.mark.parametrize(
    ("failure", "status", "code"),
    [
        (InvalidMediaError("Invalid audio."), 422, "invalid_media"),
        (NoAudioError("No audio stream."), 422, "no_audio"),
        (AudioTooLongError("Audio is too long."), 422, "audio_too_long"),
        (RuntimeError("Private diagnostic detail"), 500, "internal_error"),
    ],
)
def test_failure_cleanup_and_recovery(context, failure, status, code):
    client, service = context
    service.failure = failure

    response = upload(client)

    assert_error(response, status, code)
    assert "Private diagnostic detail" not in response.text
    assert all(not path.exists() for path in service.paths)

    # A second request proves the capacity lock was released after failure.
    service.failure = None
    assert upload(client).status_code == 200


def test_concurrent_request_is_rejected(context):
    client, service = context
    service.block = True

    with ThreadPoolExecutor(max_workers=1) as executor:
        first_request = executor.submit(upload, client)

        try:
            # Wait until the first request is actually inside inference.
            assert service.started.wait(timeout=5)

            second_response = upload(client)

            assert_error(second_response, 503, "service_busy")
            assert second_response.headers["Retry-After"] == "5"

            # A busy model must not make health checks unavailable.
            assert client.get("/health").status_code == 200

        finally:
            # Always unblock the worker, even if an assertion fails.
            service.release.set()

        assert first_request.result(timeout=10).status_code == 200

    # Capacity must also be available after the successful request.
    service.block = False
    assert upload(client).status_code == 200

def test_request_body_limit_before_inference(context):
    client, service = context

    # The fixture configures a 32-byte file limit.
    # Exceed the complete body limit, including its 1 MiB allowance.
    response = upload(client, b"x" * (1024 * 1024 + 33))

    assert_error(response, 413, "request_too_large")
    assert service.paths == []

    # Early rejection must not affect processing of subsequent valid requests.
    assert upload(client).status_code == 200


def test_openapi_documents_custom_error_contract(context):
    client, _ = context

    schema = client.get("/openapi.json").json()
    responses = schema["paths"]["/transcriptions"]["post"]["responses"]

    for status in ("400", "413", "422", "500", "503"):
        documented_schema = responses[status]["content"]["application/json"]["schema"]

        assert documented_schema["$ref"] == "#/components/schemas/ErrorResponse"