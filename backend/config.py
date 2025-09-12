# backend/config.py
from functools import lru_cache
from pathlib import Path
from typing import Optional
import os
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]

# Load in ascending precedence; later overrides earlier
load_dotenv(ROOT / ".env")
load_dotenv(ROOT / ".env.local", override=True)
load_dotenv(ROOT / "backend" / ".env", override=True)
load_dotenv(ROOT / "backend" / ".env.local", override=True)

class Settings:
    # Server
    PORT: int = int(os.getenv("PORT", "8000"))
    DATA_DIR: str = os.getenv("DATA_DIR", str(ROOT / "data"))
    MAX_UPLOAD_MB: int = int(os.getenv("MAX_UPLOAD_MB", "200"))

    # CORS
    ALLOWED_ORIGINS: list[str] = [
        s for s in os.getenv("ALLOWED_ORIGINS", "http://localhost:5173").split(",") if s
    ]

    # ASR
    ASR_MODEL_NAME: str = os.getenv("ASR_MODEL_NAME", "tiny")
    ASR_DEVICE: str = os.getenv("ASR_DEVICE", "cpu")
    ASR_COMPUTE_TYPE: str = os.getenv("ASR_COMPUTE_TYPE", "int8")

    # LLM
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "openai")
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    ANTHROPIC_API_KEY: Optional[str] = os.getenv("ANTHROPIC_API_KEY")
    ANTHROPIC_MODEL: str = os.getenv("ANTHROPIC_MODEL", "claude-sonnet")

    # TTS
    TTS_PROVIDER: str = os.getenv("TTS_PROVIDER", "elevenlabs")
    ELEVENLABS_API_KEY: Optional[str] = os.getenv("ELEVENLABS_API_KEY")
    ELEVENLABS_VOICE_ID: str = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")

@lru_cache
def get_settings() -> Settings:
    return Settings()
