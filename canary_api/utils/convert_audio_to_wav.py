import io
from pathlib import Path
from pydub import AudioSegment

SUPPORTED_FORMATS = {'mp3', 'mp4', 'mpeg', 'mpga', 'm4a', 'wav', 'webm'}
TARGET_SAMPLE_RATE = 16000


def get_audio_format(filename: str) -> str:
    """Extract format from filename extension."""
    return Path(filename).suffix.lower().lstrip('.')


def convert_audio_to_wav(audio_bytes: bytes, filename: str, sample_rate: int = TARGET_SAMPLE_RATE) -> bytes:
    """
    Convert audio from various formats to mono WAV.

    Args:
        audio_bytes: Raw audio data
        filename: Original filename (used to detect format)
        sample_rate: Target sample rate (default 16000)

    Returns:
        WAV audio bytes (mono)
    """
    audio_format = get_audio_format(filename)

    if audio_format not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported audio format: {audio_format}. Supported: {', '.join(SUPPORTED_FORMATS)}")

    # Load audio using pydub
    audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format=audio_format)

    # Convert to mono if not already
    if audio.channels > 1:
        audio = audio.set_channels(1)

    # Resample if needed
    if audio.frame_rate != sample_rate:
        audio = audio.set_frame_rate(sample_rate)

    # Export as WAV
    output_buffer = io.BytesIO()
    audio.export(output_buffer, format="wav")

    return output_buffer.getvalue()