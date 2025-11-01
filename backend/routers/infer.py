# src/api/routes/infer.py
from __future__ import annotations
from mr_tai_gameplay.src.pipeline_infer import run_inference_multi
from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pathlib import Path
import tempfile, json

router = APIRouter()

# (Stub) your TTS hook — wire this up later
def synth_tts(text: str, voice_id: str | None, emotion: str | None) -> str | None:
    # TODO: integrate your TTS here and return a relative path like "static/audio/xyz.mp3"
    # Return None to skip audio for now.
    return None

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
    # 1) Save upload to temp and run multi-segment inference
    with tempfile.TemporaryDirectory() as td:
        filename = file.filename or "upload.mp4"
        tmp_path = Path(td) / filename
        tmp_path.write_bytes(await file.read())

        out_json = Path(td) / "out.json"
        run_inference_multi(
            str(tmp_path),
            str(out_json),
            device="mps",
            ckpt="out_ckpt/r3d18_ep10.pt",
        )
        payload = json.loads(out_json.read_text())

    # 2) Stitch commentary text from timed lines (also return lines verbatim)
    lines = payload.get("tts", {}).get("lines", [])
    stitched = "\n".join([ln.get("text", "").strip() for ln in lines if (ln.get("text") or "").strip()])

    # 3) (Optional) TTS — wire this up later
    audio_rel = synth_tts(stitched, voiceId, emotion) if tts else None

    return JSONResponse({
        "text": stitched,
        "audio_url": audio_rel,     # e.g., "static/audio/abc.mp3" or None
        "video_url": None,          # set if you dub later
        "meta": {
            "prompt_tone": tone,
            "prompt_bias": bias,
            "gender": gender,
        },
        # Debug + UI data
        "gameplay_only": payload.get("gameplay_only"),
        "tts": payload.get("tts"),  # includes {"clip_id", "lines":[{t_say, text}, ...]}
    })
