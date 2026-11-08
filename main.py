import logging
from contextlib import asynccontextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock
from time import perf_counter
from typing import Annotated, AsyncIterator
from uuid import uuid4
from middleware import UploadBodyLimitMiddleware
from schemas import TRANSCRIPTION_ERROR_RESPONSES

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse

from config import load_settings
from errors import AudioTooLongError, InvalidMediaError, NoAudioError
from transcription import TranscriptionResult, TranscriptionService


settings = load_settings()
INDEX_PATH = Path(__file__).resolve().parent / "static" / "index.html"
CHUNK_SIZE_BYTES = 1024 * 1024

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class APIError(Exception):
    """Carry an intentional API failure to the centralized handler."""

    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.headers = headers


def error_response(
    request: Request,
    status_code: int,
    code: str,
    message: str,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    """Build a consistent response without exposing internal exceptions."""

    request_id = request.state.request_id

    return JSONResponse(
        status_code=status_code,
        content={
            "code": code,
            "detail": message,
            "request_id": request_id,
        },
        headers={
            **(headers or {}),
            "X-Request-ID": request_id,
        },
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initialize one shared model before accepting requests."""

    app.state.ready = False
    logger.info("Loading model: %s", settings.model_id)

    app.state.transcription_service = TranscriptionService(
        model_id=settings.model_id,
        max_audio_seconds=settings.max_audio_seconds,
    )
    app.state.transcription_lock = Lock()
    app.state.ready = True

    logger.info("Transcription service is ready")

    try:
        yield
    finally:
        app.state.ready = False
        del app.state.transcription_service
        logger.info("Transcription service stopped")


app = FastAPI(
    title="OriginAI Transcription Service",
    version="0.2.0",
    lifespan=lifespan,
)

# Register body limiting before request tracking is registered.
# Starlette wraps later registrations around earlier middleware,
# so request tracking assigns the ID before body limiting runs.
app.add_middleware(
    UploadBodyLimitMiddleware,
    get_max_file_bytes=lambda: settings.max_upload_bytes,
)

@app.middleware("http")
async def track_request(request: Request, call_next):
    """Assign a server-generated ID and measure total request handling time."""

    # Generate our own ID rather than trusting arbitrary client header values.
    request.state.request_id = uuid4().hex
    started = perf_counter()

    try:
        response = await call_next(request)
    except Exception:
        logger.exception(
            "Unhandled request failure: request_id=%s",
            request.state.request_id,
        )
        response = error_response(
            request,
            500,
            "internal_error",
            "An internal server error occurred.",
        )

    response.headers["X-Request-ID"] = request.state.request_id

    # Log method and path, but not request bodies or transcript contents.
    logger.info(
        "request_id=%s method=%s path=%s status=%d elapsed=%.3fs",
        request.state.request_id,
        request.method,
        request.url.path,
        response.status_code,
        perf_counter() - started,
    )

    return response


@app.exception_handler(APIError)
async def handle_api_error(request: Request, exc: APIError):
    """Translate expected application failures into JSON."""

    return error_response(
        request,
        exc.status_code,
        exc.code,
        exc.message,
        exc.headers,
    )


@app.exception_handler(RequestValidationError)
async def handle_validation_error(
    request: Request,
    exc: RequestValidationError,
):
    """Avoid exposing raw validation input in public responses."""

    return error_response(
        request,
        422,
        "invalid_request",
        "A media file must be provided in the 'file' field.",
    )


@app.get("/", response_class=FileResponse, include_in_schema=False)
def home() -> FileResponse:
    """Serve the existing interface without a separate frontend server."""

    return FileResponse(INDEX_PATH, media_type="text/html")


@app.get("/health")
def health() -> dict[str, str]:
    """Report application liveness without invoking the model."""

    return {"status": "ok"}


@app.get("/ready")
def readiness(request: Request):
    """Report whether the shared model is available for processing."""

    # Being busy does not mean the application is unready.
    if not getattr(request.app.state, "ready", False):
        raise APIError(503, "not_ready", "The model is not ready.")

    return {"status": "ready"}


@app.post(
    "/transcriptions",
    response_model=TranscriptionResult,
    responses=TRANSCRIPTION_ERROR_RESPONSES,
)
def create_transcription(
    request: Request,
    file: Annotated[UploadFile, File(description="Media file to transcribe")],
) -> TranscriptionResult:
    """Validate an upload, transcribe it, and clean up temporary resources."""

    processing_lock = request.app.state.transcription_lock

    # This capacity guard applies to one process; run with one Uvicorn worker.
    if not processing_lock.acquire(blocking=False):
        raise APIError(
            503,
            "service_busy",
            "The transcription service is busy. Please try again shortly.",
            headers={"Retry-After": "5"},
        )

    try:
        with TemporaryDirectory(prefix="originai-") as temporary_directory:
            # Never construct a server path from the uploaded filename.
            audio_path = Path(temporary_directory) / "media"
            total_bytes = 0

            with audio_path.open("wb") as destination:
                while chunk := file.file.read(CHUNK_SIZE_BYTES):
                    total_bytes += len(chunk)

                    if total_bytes > settings.max_upload_bytes:
                        raise APIError(
                            413,
                            "file_too_large",
                            "The file exceeds the configured upload size limit.",
                        )

                    destination.write(chunk)

            if total_bytes == 0:
                raise APIError(
                    400,
                    "empty_file",
                    "The uploaded file is empty.",
                )

            service = request.app.state.transcription_service

            try:
                result = service.transcribe(audio_path)

            except AudioTooLongError as exc:
                raise APIError(
                    422,
                    "audio_too_long",
                    str(exc),
                ) from exc

            except NoAudioError as exc:
                raise APIError(
                    422,
                    "no_audio",
                    str(exc),
                ) from exc

            except InvalidMediaError as exc:
                raise APIError(
                    422,
                    "invalid_media",
                    str(exc),
                ) from exc

            logger.info(
                "request_id=%s bytes=%d audio_seconds=%.2f processing_seconds=%.2f",
                request.state.request_id,
                total_bytes,
                result.audio_duration_seconds,
                result.processing_time_seconds,
            )

            return result

    finally:
        # The temporary directory is cleaned first, then capacity is released.
        # This runs after success and after all processing failures.
        processing_lock.release()