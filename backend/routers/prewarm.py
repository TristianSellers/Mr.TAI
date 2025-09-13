# backend/routers/prewarm.py
from fastapi import APIRouter
import os, time

# warm ASR via the shared cache in main.py
from backend.main import get_asr_model
# provider factories (instantiate only; avoid paid calls)
from backend.services.llm_providers import get_llm
from backend.services.tts_providers import get_tts

router = APIRouter()

def _env(name: str, default: str | None = None) -> str | None:
    val = os.getenv(name, default)
    return val.strip() if isinstance(val, str) else val

@router.get("/prewarm")
@router.post("/prewarm")
def prewarm():
    """
    Initialize heavy dependencies so the first real request is faster.
    - ASR: populates the lru_cache() instance via get_asr_model()
    - LLM/TTS: instantiate providers (no network calls to avoid cost)
    """
    out = {"ok": True, "timings_ms": {}, "providers": {}}

    # ASR
    t0 = time.time()
    try:
        asr = get_asr_model()  # cached singleton
        out["providers"]["asr"] = {
            "model": _env("ASR_MODEL_NAME", "tiny"),
            "device": _env("ASR_DEVICE", "cpu"),
            "compute_type": _env("ASR_COMPUTE_TYPE", "int8"),
            "loaded": True,
            "cls": type(asr).__name__,
        }
    except Exception as e:
        out["ok"] = False
        out.setdefault("errors", []).append(f"ASR: {e}")
        out["providers"]["asr"] = {"loaded": False, "error": str(e)}
    out["timings_ms"]["asr"] = int((time.time() - t0) * 1000)

    # LLM
    t0 = time.time()
    try:
        llm = get_llm()  # do not call .generate() to avoid paid tokens
        out["providers"]["llm"] = {
            "provider": _env("LLM_PROVIDER", "stub"),
            "model_openai": _env("OPENAI_MODEL"),
            "model_anthropic": _env("ANTHROPIC_MODEL"),
            "loaded": True,
            "cls": type(llm).__name__,
        }
    except Exception as e:
        out["ok"] = False
        out.setdefault("errors", []).append(f"LLM: {e}")
        out["providers"]["llm"] = {"loaded": False, "error": str(e)}
    out["timings_ms"]["llm"] = int((time.time() - t0) * 1000)

    # TTS
    t0 = time.time()
    try:
        tts = get_tts()  # do not synthesize to avoid paid audio
        out["providers"]["tts"] = {
            "provider": _env("TTS_PROVIDER", "elevenlabs"),
            "voice_elevenlabs": _env("ELEVENLABS_VOICE_ID"),
            "model_elevenlabs": _env("ELEVENLABS_MODEL"),
            "model_openai": _env("OPENAI_TTS_MODEL"),
            "voice_openai": _env("OPENAI_TTS_VOICE"),
            "loaded": True,
            "cls": type(tts).__name__,
        }
    except Exception as e:
        out["ok"] = False
        out.setdefault("errors", []).append(f"TTS: {e}")
        out["providers"]["tts"] = {"loaded": False, "error": str(e)}
    out["timings_ms"]["tts"] = int((time.time() - t0) * 1000)

    return out
