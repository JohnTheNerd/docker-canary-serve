from pydantic import Field
from pydantic_settings import BaseSettings


class CanarySettings(BaseSettings):
    # Core
    models_path: str = Field("./models", description="Path to the models directory")

    # Model settings
    model_name: str = Field("nvidia/canary-180m-flash", description="Name of the pretrained Canary model")
    beam_size: int = Field(1, description="Beam size for decoding strategy")
    batch_size: int = Field(1, description="Default batch size for transcription")
    pnc: str = Field("yes", description="Punctuation and capitalization: 'yes' or 'no'")
    timestamps: str = Field("no", description="Timestamps in output: 'yes' or 'no'")
    model_precision: str = Field("fp32", description="Model precision: 'fp32', 'fp16', or 'bf16'")

    # Long audio settings
    max_chunk_duration_sec: int = Field(10, description="Maximum chunk duration in seconds")

    # API settings
    api_host: str = Field("0.0.0.0", description="Host to bind the API to")
    api_port: int = Field(8000, description="Port to bind the API to")
    api_max_file_size_mb: int = Field(100, description="Maximum file size in MB for API requests")
    api_model: str = Field("whisper-1", description="Model name exposed to OpenAI-compatible API")

    # Audio settings
    audio_sample_rate: int = Field(16000, description="Target sample rate for audio conversion")

    class Config:
        env_prefix = "CANARY_"
        env_file = ".env"


settings = CanarySettings()
