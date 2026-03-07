from typing import Optional
from pydantic import BaseModel
from fastapi.responses import JSONResponse


class OpenAIErrorDetail(BaseModel):
    message: str
    type: str = "invalid_request_error"
    param: Optional[str] = None
    code: Optional[str] = None


class OpenAIError(BaseModel):
    error: OpenAIErrorDetail


def create_openai_error_response(
    message: str,
    error_type: str = "invalid_request_error",
    param: Optional[str] = None,
    code: Optional[str] = None,
    status_code: int = 400
) -> JSONResponse:
    """
    Create an OpenAI-compatible error response.

    Args:
        message: Error message
        error_type: Type of error (e.g., invalid_request_error, server_error)
        param: Parameter that caused the error (optional)
        code: Error code (optional)
        status_code: HTTP status code

    Returns:
        JSONResponse with OpenAI error format
    """
    error_detail = OpenAIErrorDetail(
        message=message,
        type=error_type,
        param=param,
        code=code
    )
    error_response = OpenAIError(error=error_detail)

    return JSONResponse(
        status_code=status_code,
        content=error_response.model_dump()
    )


def create_file_error(message: str, status_code: int = 400) -> JSONResponse:
    """Create error for file-related issues."""
    return create_openai_error_response(
        message=message,
        error_type="invalid_request_error",
        param="file",
        status_code=status_code
    )


def create_model_error(message: str, status_code: int = 400) -> JSONResponse:
    """Create error for model-related issues."""
    return create_openai_error_response(
        message=message,
        error_type="invalid_request_error",
        param="model",
        status_code=status_code
    )


def create_server_error(message: str = "Internal server error") -> JSONResponse:
    """Create error for server-side issues."""
    return create_openai_error_response(
        message=message,
        error_type="server_error",
        status_code=500
    )