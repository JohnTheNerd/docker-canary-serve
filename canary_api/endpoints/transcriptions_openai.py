"""
OpenAI-compatible transcriptions endpoint.

Implements: POST /v1/audio/transcriptions
Supports both standard and streaming (stream=True) responses.
"""
import logging
import json
import wave
from typing import Optional, AsyncGenerator
from tempfile import NamedTemporaryFile
import os

from fastapi import APIRouter, UploadFile, Form, HTTPException
from fastapi.responses import Response, JSONResponse, StreamingResponse

from canary_api.endpoints.audio_common import process_audio_request
from canary_api.utils.openai_errors import create_file_error, create_server_error
from canary_api.utils.convert_audio_to_wav import convert_audio_to_wav, SUPPORTED_FORMATS
from canary_api.utils.split_audio_into_chunks import split_audio_into_chunks
from canary_api.services.canary_service import CanaryService
from canary_api.settings import settings

logger = logging.getLogger(__name__)

router = APIRouter()

# Supported response formats
SUPPORTED_RESPONSE_FORMATS = ['json', 'text', 'srt', 'vtt', 'verbose_json']

# Model name exposed to API (OpenAI uses "whisper-1")
EXPOSED_MODEL_NAME = "whisper-1"

# Supported languages
SUPPORTED_LANGUAGES = ['en', 'de', 'fr', 'es']


def format_sse_event(data: dict) -> str:
    """Format data as Server-Sent Event."""
    return f"data: {json.dumps(data)}\n\n"


async def stream_transcription(
    audio_bytes: bytes,
    filename: str,
    language: Optional[str],
    beam_size: int,
    max_file_size_bytes: int,
) -> AsyncGenerator[str, None]:
    """
    Stream transcription results as SSE events.

    Follows OpenAI streaming format with transcript.text.delta and transcript.text.done events.
    """
    # Check file size
    if len(audio_bytes) > max_file_size_bytes:
        raise HTTPException(
            status_code=400,
            detail=create_file_error(
                f"File size ({len(audio_bytes) // (1024 * 1024)}MB) exceeds maximum"
            ).body
        )

    # Check file format
    audio_format = filename.split('.')[-1].lower()
    if audio_format not in SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=create_file_error(
                f"Unsupported file format: {audio_format}"
            ).body
        )

    # Validate language if provided
    if language and language not in SUPPORTED_LANGUAGES:
        raise HTTPException(
            status_code=400,
            detail=create_file_error(
                f"Unsupported language '{language}'"
            ).body
        )

    # Convert to WAV
    try:
        wav_bytes = convert_audio_to_wav(audio_bytes, filename)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=create_file_error(f"Failed to process audio: {str(e)}").body
        )

    # Save to temp file
    with NamedTemporaryFile(delete=False, suffix=".wav") as temp:
        temp.write(wav_bytes)
        audio_path = temp.name

    try:
        # Determine language for transcription
        source_lang = language if language else 'en'
        target_lang = source_lang

        transcriber = CanaryService()

        # Check duration
        with wave.open(audio_path, 'rb') as wav:
            frames = wav.getnframes()
            rate = wav.getframerate()
            duration = frames / float(rate)

        # Split into chunks if needed
        if duration > settings.max_chunk_duration_sec:
            chunk_paths = split_audio_into_chunks(audio_path, settings.max_chunk_duration_sec)
        else:
            chunk_paths = [audio_path]

        # Process each chunk and stream results
        accumulated_text = ""
        for i, chunk_path in enumerate(chunk_paths):
            try:
                results = transcriber.transcribe(
                    audio_input=[chunk_path],
                    batch_size=1,
                    pnc="yes",
                    timestamps=None,
                    source_lang=source_lang,
                    target_lang=target_lang,
                    beam_size=beam_size
                )

                text = results[0].text.strip()

                if text:
                    # Stream word by word as deltas
                    words = text.split()
                    for word in words:
                        accumulated_text += word + " "
                        yield format_sse_event({
                            "type": "transcript.text.delta",
                            "delta": word + " "
                        })

            finally:
                # Clean up chunk if it's not the original file
                # split_audio_into_chunks creates new temp files for ALL chunks,
                # so we only skip deletion for the last chunk when no splitting occurred
                if len(chunk_paths) > 1 or i < len(chunk_paths) - 1:
                    if os.path.exists(chunk_path):
                        os.remove(chunk_path)

        # Emit done event with full text
        if accumulated_text:
            yield format_sse_event({
                "type": "transcript.text.done",
                "text": accumulated_text.strip()
            })
        else:
            yield format_sse_event({
                "type": "transcript.text.done",
                "text": ""
            })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Streaming transcription failed: {e}")
        yield format_sse_event({
            "type": "error",
            "message": str(e)
        })
    finally:
        # Clean up original audio file if it hasn't been deleted already
        if os.path.exists(audio_path):
            os.remove(audio_path)


@router.post("/transcriptions")
async def transcriptions_endpoint(
    file: UploadFile = Form(...),
    model: str = Form(...),
    response_format: Optional[str] = Form(None),
    language: Optional[str] = Form(None),
    temperature: float = Form(0.0),
    prompt: Optional[str] = Form(None),
    stream: bool = Form(False),
    beam_size: Optional[int] = Form(None),
):
    """
    Transcribe audio to text.

    OpenAI-compatible endpoint for audio transcription.
    Supports streaming with stream=True parameter.

    Args:
        file: Audio file to transcribe (required)
        model: Model to use (required, but ignored - always uses configured model)
        response_format: Output format (json, text, srt, vtt, verbose_json). Ignored when stream=True
        language: Source language code (en, de, fr, es)
        temperature: Accepted for OpenAI compatibility but ignored
        prompt: Prompt to guide transcription (not currently supported)
        stream: If True, return Server-Sent Events stream
        beam_size: Beam size for decoding (default: 1)

    Returns:
        Transcription result in requested format, or SSE stream if stream=True
    """
    try:
        # Validate model parameter (accept but log if not whisper-1)
        if model != EXPOSED_MODEL_NAME:
            logger.warning(
                f"Model '{model}' requested, but only '{EXPOSED_MODEL_NAME}' is available. "
                f"Using '{EXPOSED_MODEL_NAME}'."
            )

        # Validate file
        if not file.filename:
            raise HTTPException(
                status_code=400,
                detail=create_file_error("Missing file").body
            )

        # Read file content
        audio_bytes = await file.read()

        if not audio_bytes:
            raise HTTPException(
                status_code=400,
                detail=create_file_error("Empty file").body
            )

        # Handle streaming response
        if stream:
            # Default beam_size to 1 if not provided
            if beam_size is None:
                beam_size = 1
            return StreamingResponse(
                stream_transcription(
                    audio_bytes=audio_bytes,
                    filename=file.filename,
                    language=language,
                    beam_size=beam_size,
                    max_file_size_bytes=settings.api_max_file_size_mb * 1024 * 1024,
                ),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",  # Disable nginx buffering
                }
            )

        # Default to json if response_format not specified
        if response_format is None:
            response_format = "json"

        # Validate response format
        if response_format not in SUPPORTED_RESPONSE_FORMATS:
            raise HTTPException(
                status_code=400,
                detail=create_file_error(
                    f"Invalid response_format '{response_format}'. "
                    f"Supported: {', '.join(SUPPORTED_RESPONSE_FORMATS)}"
                ).body
            )

        # Warn if prompt is provided but not supported
        if prompt:
            logger.warning("Prompt parameter is not currently supported and will be ignored")

        # Default beam_size to 1 if not provided
        if beam_size is None:
            beam_size = 1

        # Process the request
        result = await process_audio_request(
            audio_bytes=audio_bytes,
            filename=file.filename,
            language=language,
            response_format=response_format,
            temperature=temperature,
            beam_size=beam_size,
            source_lang=language,
            target_lang=language if language else 'en',
            max_file_size_bytes=settings.api_max_file_size_mb * 1024 * 1024,
        )

        # Return response in appropriate format
        if response_format == 'text':
            return Response(content=result['text'], media_type="text/plain")
        elif response_format in ('srt', 'vtt'):
            media_type = 'text/vtt' if response_format == 'vtt' else 'text/subrip'
            return Response(content=result['text'], media_type=media_type)
        else:
            return JSONResponse(content=result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=create_server_error(str(e)).body
        )