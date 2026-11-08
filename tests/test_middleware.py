import asyncio

import pytest
from starlette.responses import JSONResponse

from middleware import UploadBodyLimitMiddleware


def run_request(chunks, limit, headers=None):
    """Exercise ASGI messages without multipart parsing or a real model."""

    async def scenario():
        downstream_calls = []
        sent_messages = []
        received_chunks = iter(chunks)

        async def receive():
            try:
                chunk, more_body = next(received_chunks)
            except StopIteration:
                return {"type": "http.disconnect"}

            return {
                "type": "http.request",
                "body": chunk,
                "more_body": more_body,
            }

        async def send(message):
            sent_messages.append(message)

        async def downstream(scope, receive, send):
            # Reconstruct the accepted body exactly as a parser would receive it.
            body = bytearray()

            while True:
                message = await receive()
                body.extend(message.get("body", b""))

                if not message.get("more_body", False):
                    break

            downstream_calls.append(bytes(body))
            response = JSONResponse({"accepted": True})
            await response(scope, receive, send)

        scope = {
            "type": "http",
            "method": "POST",
            "path": "/transcriptions",
            "headers": headers or [],
            "state": {"request_id": "test-request-id"},
        }

        middleware = UploadBodyLimitMiddleware(
            downstream,
            get_max_file_bytes=lambda: limit,
        )

        await middleware(scope, receive, send)
        return downstream_calls, sent_messages

    return asyncio.run(scenario())


def response_status(messages):
    return next(
        message["status"]
        for message in messages
        if message["type"] == "http.response.start"
    )


def test_accepted_body_is_replayed_without_changes(monkeypatch):
    # Use a small allowance to test boundaries without allocating large data.
    monkeypatch.setattr("middleware.MULTIPART_ALLOWANCE_BYTES", 8)

    calls, messages = run_request(
        chunks=[(b"abc", True), (b"defghijk", False)],
        limit=3,
    )

    # Exactly 11 bytes is accepted: 3 bytes plus an 8-byte allowance.
    assert response_status(messages) == 200
    assert calls == [b"abcdefghijk"]


@pytest.mark.parametrize(
    "headers",
    [
        [],
        [(b"content-length", b"1")],
    ],
)
def test_oversized_body_never_reaches_parser(monkeypatch, headers):
    monkeypatch.setattr("middleware.MULTIPART_ALLOWANCE_BYTES", 8)

    calls, messages = run_request(
        chunks=[(b"abcdef", True), (b"ghijkl", False)],
        limit=3,
        headers=headers,
    )

    # Actual size is 12 bytes, exceeding the 11-byte request limit.
    assert response_status(messages) == 413
    assert calls == []

    start = next(
        message
        for message in messages
        if message["type"] == "http.response.start"
    )

    assert (
        b"x-request-id",
        b"test-request-id",
    ) in start["headers"]