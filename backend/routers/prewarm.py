# backend/routers/prewarm.py
from fastapi import APIRouter
from dotenv import load_dotenv
from faster_whisper import WhisperModel
import os

router = APIRouter()

@router.post("/prewarm")
def prewarm():
    """
    Warm up heavy dependencies so the first real request is fast.
    For now: loads the ASR model (faster-whisper).
    LLM/TTS warming will be added in Week 4 integration.
    """
    load_dotenv()  # ensure .env/.env.local are loaded

    model = os.getenv("ASR_MODEL_NAME", "tiny")
    device = os.getenv("ASR_DEVICE", "cpu")              # "cpu" or "auto"
    compute = os.getenv("ASR_COMPUTE_TYPE", "int8")      # e.g. "int8", "float16"

    try:
        # Model load itself is the warm-up
        WhisperModel(model, device=device, compute_type=compute)
        return {"ok": True, "asr": {"model": model, "device": device, "compute_type": compute}}
    except Exception as e:
        return {"ok": False, "asr": {"error": str(e)}}
