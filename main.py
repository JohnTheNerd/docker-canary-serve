from fastapi import FastAPI
from canary_api.endpoints.transcriptions_endpoint import router as asr_router
from canary_api.endpoints.transcriptions_openai import router as transcriptions_router
from canary_api.endpoints.translations_openai import router as translations_router
from canary_api.settings import settings
from canary_api.services.canary_service import CanaryService

app = FastAPI(
    title="Nvidia Canary ASR API",
    version="1.0.0",
    description="OpenAI-compatible API for Nvidia Canary models"
)


@app.on_event("startup")
async def preload_model():
    """Preload the Canary model on application startup."""
    CanaryService()


# Legacy endpoint
app.include_router(asr_router)

# OpenAI-compatible endpoints
app.include_router(transcriptions_router, prefix="/v1/audio", tags=["transcriptions"])
app.include_router(translations_router, prefix="/v1/audio", tags=["translations"])

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=settings.api_host, port=settings.api_port)
