"""
OpenAI-compatible translations endpoint.

Implements: POST /v1/audio/translations
"""
import logging
from typing import Optional

from fastapi import APIRouter, UploadFile, Form, HTTPException
from fastapi.responses import Response, JSONResponse

from canary_api.endpoints.audio_common import process_audio_request, SUPPORTED_LANGUAGES
from canary_api.utils.openai_errors import create_file_error, create_server_error
from canary_api.settings import settings

logger = logging.getLogger(__name__)

router = APIRouter()

# Supported response formats
SUPPORTED_RESPONSE_FORMATS = ['json', 'text', 'srt', 'vtt', 'verbose_json']

# Model name exposed to API (OpenAI uses "whisper-1")
EXPOSED_MODEL_NAME = "whisper-1"


@router.post("/translations")
async def translations_endpoint(
    file: UploadFile = Form(...),
    model: str = Form(...),
    response_format: str = Form("json"),
    temperature: float = Form(0.0),
    prompt: Optional[str] = Form(None),
    target_lang: Optional[str] = Form(None),
    beam_size: Optional[int] = Form(None),
):
    """
    Translate audio to a target language.

    OpenAI-compatible endpoint for audio translation.
    Transcribes audio from any supported language and translates to the target language.

    Args:
        file: Audio file to translate (required)
        model: Model to use (required, but ignored - always uses configured model)
        response_format: Output format (json, text, srt, vtt, verbose_json)
        temperature: Accepted for OpenAI compatibility but ignored
        prompt: Prompt to guide translation (not currently supported)
        target_lang: Target language for translation (default: 'en').
                    Supported: bg, hr, cs, da, nl, en, et, fi, fr, de, el, hu,
                    it, lv, lt, mt, pl, pt, ro, sk, sl, es, sv, ru, uk
        beam_size: Beam size for decoding (default: 1)

    Returns:
        Translation result in requested format
    """
    try:
        # Validate model parameter (accept but log if not whisper-1)
        if model != EXPOSED_MODEL_NAME:
            logger.warning(
                f"Model '{model}' requested, but only '{EXPOSED_MODEL_NAME}' is available. "
                f"Using '{EXPOSED_MODEL_NAME}'."
            )

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

        # Validate target language if provided
        if target_lang is None:
            target_lang = 'en'  # Default to English for backward compatibility
        elif target_lang not in SUPPORTED_LANGUAGES:
            raise HTTPException(
                status_code=400,
                detail=create_file_error(
                    f"Unsupported target_lang '{target_lang}'. "
                    f"Supported: {', '.join(SUPPORTED_LANGUAGES)}"
                ).body
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

        # Default beam_size to 1 if not provided
        if beam_size is None:
            beam_size = 1

        # Process the request for translation
        result = await process_audio_request(
            audio_bytes=audio_bytes,
            filename=file.filename,
            response_format=response_format,
            temperature=temperature,
            beam_size=beam_size,
            source_lang=None,  # Auto-detect source language
            target_lang=target_lang,
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
        logger.error(f"Translation failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=create_server_error(str(e)).body
        )