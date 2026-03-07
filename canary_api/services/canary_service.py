import gc
import logging
from pathlib import Path
from typing import Optional
import torch
from nemo.collections.asr.models import EncDecMultiTaskModel

from canary_api.utils.download_model import download_model
from canary_api.settings import settings

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(name)s] [%(levelname)s] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Suppress NeMo warnings about training/validation data setup
# These are emitted when loading a pre-trained model that has train_ds/validation_ds configs
# but we're only using it for inference
logging.getLogger("nemo.core.classes.modelPT").setLevel(logging.ERROR)


class CanaryService:
    """
    A class to handle transcription and translation using the NVIDIA Canary models.
    Uses strict singleton pattern: only ONE model instance exists globally.
    """

    _instance: Optional["CanaryService"] = None

    def __new__(
        cls,
        model_name: str = settings.model_name
    ) -> "CanaryService":
        """
        Strict singleton pattern: only ONE model instance exists globally.
        If an instance already exists, it is returned regardless of model_name.
        """
        if cls._instance is not None:
            if model_name != settings.model_name:
                logger.warning(
                    f"Model already loaded ({settings.model_name}). "
                    f"Ignoring requested model: {model_name}"
                )
            logger.debug(f"Reusing existing Canary model instance")
            return cls._instance

        # Create the single instance
        instance = super().__new__(cls)
        cls._instance = instance
        return instance

    def __init__(
        self,
        model_name: str = settings.model_name
    ):
        """
        Initializes the Canary model. Downloads it if not already present locally.
        Only runs once globally due to strict singleton pattern.
        """
        # Skip initialization if already initialized (strict singleton)
        if hasattr(self, '_initialized') and self._initialized:
            return

        logger.info(f"Initializing Canary model: {model_name}")

        # Construct full local path
        model_dir = Path(settings.models_path) / model_name
        model_file = model_dir / f"{model_dir.name}.nemo"

        # Download if not exists
        if not model_file.exists():
            logger.info(f"Downloading model: {model_name}")
            Path(download_model(model_name=model_name, local_dir=settings.models_path))

        # Determine device and load model directly to GPU if available
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            logger.info(f"Loading model directly to GPU: {device}")
        else:
            device = torch.device("cpu")
            logger.warning("CUDA not available, loading model to CPU")

        # Load model from local path directly to the target device
        self.model = EncDecMultiTaskModel.restore_from(
            str(model_file),
            map_location=device
        )

        # Apply model precision for reduced VRAM usage
        if torch.cuda.is_available():
            precision = settings.model_precision.lower()
            if precision == "fp16":
                logger.info("Converting model to FP16 for reduced VRAM usage")
                self.model = self.model.half()
            elif precision == "bf16":
                logger.info("Converting model to BF16 for reduced VRAM usage")
                self.model = self.model.to(torch.bfloat16)
            elif precision != "fp32":
                logger.warning(f"Unknown precision '{precision}', defaulting to FP32")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug("Cleared CUDA cache after model loading")
            # Log memory stats
            logger.info(f"GPU memory after cleanup: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")

        self.model.eval()  # Ensure evaluation mode
        self.is_flash_model = "flash" in model_name.lower()
        self._initialized = True

    def transcribe(
        self,
        audio_input: list,
        batch_size: int = settings.batch_size,
        pnc: str = settings.pnc,
        timestamps: bool | None = False,
        source_lang: str = 'en',
        target_lang: str = 'en',
        beam_size: int = settings.beam_size,
    ):
        """
        Transcribes or translates the given audio input.
        """
        if not isinstance(audio_input, list):
            raise ValueError("audio_input must be a list of audio file paths.")

        # Fix timestamps value
        if isinstance(timestamps, str):
            if timestamps.lower() == 'yes':
                timestamps = True
            else:
                timestamps = None

        # Apply beam_size dynamically if different from default
        if beam_size != settings.beam_size:
            decode_cfg = self.model.cfg.decoding
            decode_cfg.beam.beam_size = beam_size
            self.model.change_decoding_strategy(decode_cfg)

        logger.debug({
            "source_lang": source_lang,
            "target_lang": target_lang,
            "batch_size":  batch_size,
            "pnc":         pnc,
            "timestamps":  timestamps,
            "beam_size":   beam_size
        })

        with torch.no_grad():
            results = self.model.transcribe(
                audio_input,
                source_lang=source_lang,
                target_lang=target_lang,
                batch_size=batch_size,
                pnc=pnc,
                timestamps=timestamps
            )

        # Clear CUDA cache to free fragmented memory after inference
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return results


if __name__ == "__main__":
    # Initialize the transcriber
    transcriber = CanaryService()

    # Transcribe a list of audio files
    results = transcriber.transcribe(
        audio_input=['audio1.wav', 'audio2.wav'],
        batch_size=2,
        pnc='yes',
        timestamps=True,
        source_lang='en',
        target_lang='en'
    )

    # Print the transcriptions
    for result in results:
        print(result.text)
