from __future__ import annotations

import os
from functools import lru_cache
from typing import List

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()


class Settings(BaseModel):
    """Runtime configuration with sensible defaults for local development."""

    openai_api_key: str | None = Field(
        default=os.getenv("OPENAI_API_KEY"),
        description="API key for OpenAI Whisper transcription.",
    )
    openai_whisper_model: str = Field(
        default=os.getenv("OPENAI_WHISPER_MODEL", "gpt-4o-mini-transcribe"),
        description="Model id to use for OpenAI audio transcription.",
    )
    hf_api_key: str | None = Field(
        default=os.getenv("HF_API_KEY"),
        description="Hugging Face Inference API token.",
    )
    sentiment_model: str = Field(
        default=os.getenv(
            "HF_SENTIMENT_MODEL", "distilbert-base-uncased-finetuned-sst-2-english"
        ),
        description="Hugging Face sentiment model repo.",
    )
    frontend_origin: str = Field(
        default=os.getenv("FRONTEND_ORIGIN", "http://localhost:3000"),
        description="Expected frontend origin for CORS.",
    )
    cors_allow_origins: List[str] = Field(
        default_factory=lambda: [
            origin.strip()
            for origin in os.getenv(
                "CORS_ALLOW_ORIGINS", "http://localhost:3000"
            ).split(",")
            if origin.strip()
        ],
        description="Comma separated list of allowed origins.",
    )
    demo_mode: bool = Field(
        default=os.getenv("DEMO_MODE", "true").lower() == "true",
        description="When true, enables lightweight fallbacks when API keys are missing.",
    )
    use_openai_sentiment: bool = Field(
        default=os.getenv("USE_OPENAI_SENTIMENT", "false").lower() == "true",
        description="If true, use OpenAI sentiment classification instead of Hugging Face.",
    )
    openai_sentiment_model: str = Field(
        default=os.getenv("OPENAI_SENTIMENT_MODEL", "gpt-4o-mini"),
        description="Model used for OpenAI sentiment classification.",
    )
    gemini_api_key: str | None = Field(
        default=os.getenv("GEMINI_API_KEY"),
        description="API key for Google Gemini sentiment classification.",
    )
    gemini_sentiment_model: str = Field(
        default=os.getenv("GEMINI_SENTIMENT_MODEL", "gemini-2.0-flash"),
        description="Gemini model ID used for sentiment calls.",
    )
    transcription_provider: str = Field(
        default=os.getenv("TRANSCRIPTION_PROVIDER", "local"),
        description="Preferred transcription backend: 'openai' or 'local'.",
    )
    local_asr_model_name: str = Field(
        default=os.getenv("LOCAL_ASR_MODEL_NAME", "vosk-model-small-en-us-0.15"),
        description="Directory name of the local ASR model.",
    )
    local_asr_model_url: str = Field(
        default=os.getenv(
            "LOCAL_ASR_MODEL_URL",
            "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip",
        ),
        description="Download URL for the Vosk ASR model zip.",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()



