"""
Shared audio processing logic for OpenAI-compatible endpoints.
"""
import logging
import os
import wave
from typing import Optional
from tempfile import NamedTemporaryFile

from fastapi import HTTPException

from canary_api.services.canary_service import CanaryService
from canary_api.utils.split_audio_into_chunks import split_audio_into_chunks
from canary_api.utils.convert_audio_to_wav import convert_audio_to_wav, SUPPORTED_FORMATS
from canary_api.utils.generate_srt_from_words import generate_srt_from_words
from canary_api.utils.clean_transcription import clean_transcription
from canary_api.settings import settings
from canary_api.utils.openai_errors import create_file_error, create_server_error

logger = logging.getLogger(__name__)

SUPPORTED_LANGUAGES = [
    'bg', 'hr', 'cs', 'da', 'nl', 'en', 'et', 'fi', 'fr', 'de',
    'el', 'hu', 'it', 'lv', 'lt', 'mt', 'pl', 'pt', 'ro', 'sk',
    'sl', 'es', 'sv', 'ru', 'uk'
]


def save_temp_audio(data: bytes) -> str:
    """Save audio bytes to a temporary WAV file."""
    with NamedTemporaryFile(delete=False, suffix=".wav") as temp:
        temp.write(data)
        return temp.name


async def process_audio_request(
    audio_bytes: bytes,
    filename: str,
    language: Optional[str] = None,
    pnc: str = 'yes',
    timestamps: str = 'no',
    temperature: float = 0.0,
    beam_size: int = 1,
    batch_size: int = 1,
    response_format: str = 'json',
    source_lang: Optional[str] = None,
    target_lang: str = 'en',
    max_file_size_bytes: int = 100 * 1024 * 1024,  # 100MB default
) -> dict:
    """
    Process audio transcription/translation request.

    Args:
        audio_bytes: Raw audio data
        filename: Original filename
        language: Source language (optional, 'en' default)
        pnc: Punctuation and capitalization ('yes' or 'no')
        timestamps: Include timestamps ('yes' or 'no')
        temperature: Accepted for OpenAI compatibility but ignored
        beam_size: Beam size for decoding (default: 1)
        batch_size: Batch size for processing
        response_format: Output format (json, text, srt, vtt, verbose_json)
        source_lang: Source language for translation
        target_lang: Target language for translation
        max_file_size_bytes: Maximum allowed file size

    Returns:
        Transcription result in requested format
    """
    # Check file size
    if len(audio_bytes) > max_file_size_bytes:
        max_mb = max_file_size_bytes // (1024 * 1024)
        raise HTTPException(
            status_code=400,
            detail=create_file_error(
                f"File size ({len(audio_bytes) // (1024 * 1024)}MB) exceeds maximum ({max_mb}MB)"
            ).body
        )

    # Check file format
    audio_format = filename.split('.')[-1].lower()
    if audio_format not in SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=create_file_error(
                f"Unsupported file format: {audio_format}. Supported: {', '.join(SUPPORTED_FORMATS)}"
            ).body
        )

    # Convert to WAV if needed
    try:
        wav_bytes = convert_audio_to_wav(audio_bytes, filename)
    except Exception as e:
        logger.error(f"Failed to convert audio: {e}")
        raise HTTPException(
            status_code=400,
            detail=create_file_error(f"Failed to process audio file: {str(e)}").body
        )

    # Set language defaults
    if source_lang is None:
        source_lang = language if language else 'en'

    # Validate language
    if source_lang not in SUPPORTED_LANGUAGES:
        raise HTTPException(
            status_code=400,
            detail=create_file_error(
                f"Unsupported language '{source_lang}'. Supported: {', '.join(SUPPORTED_LANGUAGES)}"
            ).body
        )

    # Handle timestamps based on response_format
    if response_format == 'text':
        timestamps_flag = None
    else:
        if response_format in ['srt', 'vtt']:
            timestamps = 'yes'
        if timestamps == 'yes':
            timestamps_flag = True
        elif timestamps == 'no' or timestamps is None:
            timestamps_flag = None
        else:
            logger.warning(f"Unknown timestamps value '{timestamps}', defaulting to None")
            timestamps_flag = None

    # Save audio to temp file
    audio_path = save_temp_audio(wav_bytes)

    try:
        transcriber = CanaryService()

        # Check if timestamps are requested and if the model supports it
        if timestamps_flag and not transcriber.is_flash_model:
            raise HTTPException(
                status_code=400,
                detail=create_file_error(
                    "Timestamps are only supported with flash models (e.g., canary-1b-flash)"
                ).body
            )

        # Check duration
        with wave.open(audio_path, 'rb') as wav:
            frames = wav.getnframes()
            rate = wav.getframerate()
            duration = frames / float(rate)

        texts = []
        timestamps_all = {"word": [], "segment": []}
        all_results = []

        if duration > settings.max_chunk_duration_sec:
            logger.info(
                f"Audio longer than {settings.max_chunk_duration_sec} sec ({duration:.2f} sec), using chunked inference."
            )
            chunk_paths = split_audio_into_chunks(audio_path, settings.max_chunk_duration_sec)

            offset = 0.0

            for chunk_path in chunk_paths:
                results = transcriber.transcribe(
                    audio_input=[chunk_path],
                    batch_size=batch_size,
                    pnc=pnc,
                    timestamps=timestamps_flag,
                    source_lang=source_lang,
                    target_lang=target_lang,
                    beam_size=beam_size
                )

                with wave.open(chunk_path, 'rb') as wav_chunk:
                    frames = wav_chunk.getnframes()
                    rate = wav_chunk.getframerate()
                    chunk_duration = frames / float(rate)

                texts.append(results[0].text)

                if timestamps_flag and hasattr(results[0], 'timestamp') and results[0].timestamp:
                    if 'word' in results[0].timestamp:
                        for word in results[0].timestamp['word']:
                            word['start'] += offset
                            word['end'] += offset
                            timestamps_all['word'].append(word)

                    if 'segment' in results[0].timestamp:
                        for segment in results[0].timestamp['segment']:
                            segment['start'] += offset
                            segment['end'] += offset
                            timestamps_all['segment'].append(segment)

                offset += chunk_duration
                os.remove(chunk_path)

        else:
            results = transcriber.transcribe(
                audio_input=[audio_path],
                batch_size=batch_size,
                pnc=pnc,
                timestamps=timestamps_flag,
                source_lang=source_lang,
                target_lang=target_lang,
                beam_size=beam_size
            )
            all_results.extend(results)
            texts.append(results[0].text)

            if timestamps_flag and hasattr(results[0], 'timestamp') and results[0].timestamp:
                timestamps_all['word'].extend(results[0].timestamp.get('word', []))
                timestamps_all['segment'].extend(results[0].timestamp.get('segment', []))

        full_text = " ".join(texts)

        # Format response
        if response_format == 'text':
            return {"text": clean_transcription(full_text)}
        elif response_format == 'json':
            return {"text": full_text}
        elif response_format == 'verbose_json':
            verbose_results = []
            for result in all_results:
                verbose_results.append(result.__dict__)
            return verbose_results
        elif response_format in ('srt', 'vtt'):
            if not timestamps_flag or not timestamps_all['word']:
                raise HTTPException(
                    status_code=400,
                    detail=create_file_error(
                        "Timestamps are required for SRT/VTT output. Set timestamps=yes."
                    ).body
                )

            srt_data = generate_srt_from_words(timestamps_all['word'])

            if response_format == 'srt':
                return {"text": srt_data}
            else:  # vtt
                vtt_data = "WEBVTT\n\n" + srt_data.replace(",", ".")
                return {"text": vtt_data}
        else:
            return {"text": full_text}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=create_server_error(str(e)).body
        )
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)