# backend/routers/health.py
from fastapi import APIRouter
from pathlib import Path
import os

router = APIRouter()

@router.get("/health/env")
def env_preview():
    repo_root = Path(__file__).resolve().parents[2]  # .../Mr.TAI
    data_dir = Path(os.getenv("DATA_DIR", repo_root / "data"))
    upload_dir = data_dir / "uploads"
    demo_dir = data_dir / "demo"

    return {
        "status": "ok",
        # server
        "PORT": int(os.getenv("PORT", "8000")),
        "DATA_DIR": str(data_dir.resolve()),
        "UPLOAD_DIR": str(upload_dir.resolve()),
        "DEMO_DIR": str(demo_dir.resolve()),
        "ALLOWED_ORIGINS": [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "http://localhost:5173").split(",") if o.strip()],
        # ASR
        "ASR": {
            "model": os.getenv("ASR_MODEL_NAME", "tiny"),
            "device": os.getenv("ASR_DEVICE", "cpu"),
            "compute_type": os.getenv("ASR_COMPUTE_TYPE", "int8"),
        },
        # LLM/TTS (no secrets)
        "LLM": {
            "provider": os.getenv("LLM_PROVIDER", "openai"),
            "openai_model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            "anthropic_model": os.getenv("ANTHROPIC_MODEL", "claude-sonnet"),
        },
        "TTS": {
            "provider": os.getenv("TTS_PROVIDER", "elevenlabs"),
            "elevenlabs_voice_id": os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM"),
        },
    }
