from pydantic import BaseModel, Field


class ErrorResponse(BaseModel):
    """Document the error contract shared by API failure responses."""

    code: str = Field(
        description="Stable error identifier for client-side handling.",
        examples=["invalid_media"],
    )
    detail: str = Field(
        description="Human-readable explanation of the failure.",
        examples=["The file could not be decoded as supported audio."],
    )
    request_id: str = Field(
        description="Request identifier also returned in X-Request-ID.",
        examples=["a3f5279f30564843bbd9454de964cd29"],
    )


# Explicitly document the custom error schema instead of FastAPI's
# default validation-error schema.
TRANSCRIPTION_ERROR_RESPONSES = {
    400: {
        "model": ErrorResponse,
        "description": "The uploaded file is empty.",
    },
    413: {
        "model": ErrorResponse,
        "description": "The request body or uploaded file is too large.",
    },
    422: {
        "model": ErrorResponse,
        "description": (
            "Missing file, invalid media, missing audio, or excessive audio duration."
        ),
    },
    500: {
        "model": ErrorResponse,
        "description": "An unexpected internal failure occurred.",
    },
    503: {
        "model": ErrorResponse,
        "description": "Processing capacity is currently occupied.",
        "headers": {
            "Retry-After": {
                "description": "Suggested delay in seconds before retrying.",
                "schema": {"type": "integer"},
            },
        },
    },
}