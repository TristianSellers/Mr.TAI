# backend/routers/infer.py
from __future__ import annotations

import os
import json
import asyncio
import tempfile
from uuid import uuid4
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import JSONResponse

# Import your gameplay pipeline
from mr_tai_gameplay.src.pipeline_infer import run_inference_multi

# Optional: torch only for device selection (handle absence gracefully)
try:
    import torch
except Exception:
    torch = None

router = APIRouter()


# (Stub) your TTS hook — wire this up later
def synth_tts(text: str, voice_id: str | None, emotion: str | None) -> str | None:
    # TODO: integrate your TTS here and return a relative path like "static/audio/xyz.mp3"
    # Return None to skip audio for now.
    return None


def _select_device() -> str:
    """
    Auto-select an available device, with env override:
    - MODEL_DEVICE can be "cpu" | "cuda" | "mps"
    """
    forced = (os.getenv("MODEL_DEVICE") or "").lower().strip()
    if forced in {"cpu", "cuda", "mps"}:
        return forced
    if torch is not None:
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    return "cpu"


@router.post("/infer")
async def infer_clip(
    file: UploadFile = File(...),
    tone: str = Form("hype"),
    bias: str = Form("neutral"),
    gender: str = Form("male"),
    tts: int = Form(0),                # 0 = no audio, 1 = synthesize audio
    voiceId: str | None = Form(None),  # optional voice id
    emotion: str | None = Form(None),  # optional emotion
):
    device = _select_device()
    ckpt = os.getenv("MODEL_CKPT", "out_ckpt/r3d18_ep10.pt")
    if not Path(ckpt).exists():
        # Fall back gracefully if missing — model will use torchvision weights
        ckpt = None

    try:
        # 1) Save upload safely to a temp file (streamed; no large RAM usage)
        with tempfile.TemporaryDirectory() as td:
            safe_name = Path(file.filename or "upload.mp4").name
            tmp_path = Path(td) / f"{uuid4().hex}_{safe_name}"

            # Stream the file to disk in 1 MB chunks
            with tmp_path.open("wb") as out_file:
                while chunk := await file.read(1024 * 1024):
                    out_file.write(chunk)

            # 2) Run heavy inference off the event loop (in a worker thread)
            out_json = Path(td) / "out.json"
            await asyncio.to_thread(
                run_inference_multi,
                str(tmp_path),
                str(out_json),
                None,            # clip_id
                None,            # scoreboard_json
                None,            # banner_json
                device,
                ckpt,
                32,              # frames_per_window
            )

            payload = json.loads(out_json.read_text())

        # 3) Stitch commentary text from timed lines (also return lines verbatim)
        lines = payload.get("tts", {}).get("lines", [])
        stitched = "\n".join(
            [ln.get("text", "").strip() for ln in lines if (ln.get("text") or "").strip()]
        )

        # 4) (Optional) TTS — wire this up later
        audio_rel = synth_tts(stitched, voiceId, emotion) if tts else None

        return JSONResponse({
            "text": stitched,
            "audio_url": audio_rel,     # e.g., "static/audio/abc.mp3" or None
            "video_url": None,          # set if you dub later
            "meta": {
                "prompt_tone": tone,
                "prompt_bias": bias,
                "gender": gender,
                "device": device,
            },
            # Debug + UI data
            "gameplay_only": payload.get("gameplay_only"),
            "tts": payload.get("tts"),  # {"clip_id", "lines":[{t_say, text}, ...]}
        })

    except (ValueError, FileNotFoundError) as e:
        # Bad input or unreadable video (return 400, not a crashed worker)
        return JSONResponse({"error": str(e)}, status_code=400)
    except Exception as e:
        # Unexpected errors — keep the server alive
        return JSONResponse({"error": f"internal error: {e}"}, status_code=500)
