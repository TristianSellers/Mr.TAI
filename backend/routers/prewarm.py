# backend/routers/prewarm.py
from fastapi import APIRouter
from backend.main import get_asr_model, ASR_MODEL_NAME, ASR_DEVICE, ASR_COMPUTE_TYPE

router = APIRouter()

@router.get("/prewarm")
@router.post("/prewarm")
def prewarm():
    """
    Warm up heavy dependencies so the first real request is fast.
    For now: calls get_asr_model() to populate its cache.
    """
    try:
        model = get_asr_model()  # cached singleton
        # trigger a trivial call to ensure weights are loaded
        _ = model.transcribe("tests/silence.wav") if False else None
        return {
            "ok": True,
            "asr": {
                "model": ASR_MODEL_NAME,
                "device": ASR_DEVICE,
                "compute_type": ASR_COMPUTE_TYPE,
            },
        }
    except Exception as e:
        return {"ok": False, "asr": {"error": str(e)}}
