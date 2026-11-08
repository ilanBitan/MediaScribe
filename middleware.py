from collections.abc import Callable
from tempfile import SpooledTemporaryFile
from uuid import uuid4

from starlette.concurrency import run_in_threadpool
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send


MULTIPART_ALLOWANCE_BYTES = 1024 * 1024
REPLAY_CHUNK_BYTES = 64 * 1024


class UploadBodyLimitMiddleware:
    """Bound upload request bodies before multipart parsing starts."""

    def __init__(
        self,
        app: ASGIApp,
        get_max_file_bytes: Callable[[], int],
    ) -> None:
        self.app = app
        self.get_max_file_bytes = get_max_file_bytes

    async def __call__(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        # Leave unrelated requests and application lifecycle events untouched.
        is_upload = (
            scope["type"] == "http"
            and scope["method"] == "POST"
            and scope["path"].rstrip("/") == "/transcriptions"
        )

        if not is_upload:
            await self.app(scope, receive, send)
            return

        # The complete multipart body includes boundaries and field headers.
        # The route separately enforces the exact uploaded file size.
        max_body_bytes = (
            self.get_max_file_bytes() + MULTIPART_ALLOWANCE_BYTES
        )

        total_bytes = 0

        # Keep small bodies in memory and spill larger bodies to disk.
        # The context manager also cleans up rejected and disconnected uploads.
        with SpooledTemporaryFile(
            max_size=MULTIPART_ALLOWANCE_BYTES,
            mode="w+b",
        ) as buffer:
            while True:
                message = await receive()

                # A disconnected client no longer needs a response.
                if message["type"] == "http.disconnect":
                    return

                chunk = message.get("body", b"")
                total_bytes += len(chunk)

                # Count actual bytes, even if Content-Length is missing or false.
                if total_bytes > max_body_bytes:
                    state = scope.setdefault("state", {})
                    request_id = state.setdefault("request_id", uuid4().hex)

                    response = JSONResponse(
                        status_code=413,
                        content={
                            "code": "request_too_large",
                            "detail": (
                                "The request exceeds the configured body size limit."
                            ),
                            "request_id": request_id,
                        },
                        headers={"X-Request-ID": request_id},
                    )

                    await response(scope, receive, send)
                    return

                # Disk-backed writes must not block the event loop.
                await run_in_threadpool(buffer.write, chunk)

                if not message.get("more_body", False):
                    break

            await run_in_threadpool(buffer.seek, 0)
            remaining_bytes = total_bytes
            replay_finished = False

            async def replay_receive() -> Message:
                nonlocal remaining_bytes, replay_finished

                # After replaying the body, forward disconnect notifications.
                if replay_finished:
                    return await receive()

                data = await run_in_threadpool(
                    buffer.read,
                    REPLAY_CHUNK_BYTES,
                )
                remaining_bytes -= len(data)
                replay_finished = remaining_bytes == 0

                return {
                    "type": "http.request",
                    "body": data,
                    "more_body": not replay_finished,
                }

            # Multipart parsing starts only after the complete body was accepted.
            await self.app(scope, replay_receive, send)