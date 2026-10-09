"""Actionable API errors: stable code, human message, suggested action."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class ErrorDetail(BaseModel):
    code: str
    message: str
    action: str | None = None
    details: dict[str, Any] | None = None


class ErrorBody(BaseModel):
    error: ErrorDetail


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str, action: str | None = None,
                 details: dict[str, Any] | None = None, headers: dict[str, str] | None = None):
        super().__init__(message)
        self.status = status
        self.body = ErrorBody(error=ErrorDetail(code=code, message=message, action=action, details=details))
        self.headers = headers or {}


def not_found(kind: str) -> ApiError:
    # 404 for both missing and foreign objects so object IDs cannot be probed.
    return ApiError(404, "not_found", f"{kind} not found", "Check the link or refresh the list.")


# Media/pipeline error codes mapped to user-facing explanations.
MEDIA_ERROR_TEXT: dict[str, tuple[str, str]] = {
    "corrupt_input": ("The file could not be read as audio, video or an image.",
                      "Record again or choose a different file; partially downloaded files often fail."),
    "file_too_large": ("The file is larger than the allowed size.", "Trim the recording or record a shorter clip."),
    "duration_too_long": ("The recording is longer than the allowed duration.", "Record a shorter clip (see limits in Settings)."),
    "unsupported_input": ("The file has no usable audio or video stream.", "Use a common format such as MP4, WebM, WAV or JPEG."),
    "probe_timeout": ("The file took too long to inspect.", "Try again; if it keeps failing, re-encode the file."),
    "not_found": ("The uploaded file is missing on the server.", "Upload the file again."),
    "analysis_timeout": ("Processing took longer than the allowed time.", "Try a shorter clip."),
    "worker_lost": ("Processing was interrupted repeatedly.", "Retry later; if it persists, contact the operator."),
    "internal_error": ("An unexpected processing error occurred.", "Retry; the operator can find details in the server logs."),
    "model_unavailable": ("A required model is not installed on the server.", "The operator must run `petperception fetch-models`."),
}
