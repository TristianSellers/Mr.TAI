# backend/routers/pipeline_api.py
from __future__ import annotations
from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from pydantic import BaseModel
from typing import Optional, Iterable, Dict, Any
from pathlib import Path, PurePath
import shutil, os, uuid, subprocess, shlex

from backend.services.ocr_from_video import extract_scoreboard_from_video
from backend.services.context_from_ocr import ocr_to_commentary_context

# Reuse the same services used by analyze_commentate
from backend.services.llm_providers import get_llm
from backend.services.tts_providers import get_tts
from backend.services.tone_profiles import build_llm_prompt, normalize_tone
from backend.services.mux import mux_audio_video
from backend.main import DATA_DIR, to_static_url

router = APIRouter(prefix="/pipeline", tags=["pipeline"])

TMP_DIR = Path(os.getenv("DATA_DIR", "data")) / "tmp" / "uploads"
TMP_DIR.mkdir(parents=True, exist_ok=True)

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".mpg", ".mpeg"}
MIME_TO_EXT = {
    "video/mp4": ".mp4",
    "video/quicktime": ".mov",
    "video/x-matroska": ".mkv",
    "video/x-msvideo": ".avi",
    "video/webm": ".webm",
    "video/mpeg": ".mpg",
}

def _safe_ext(upload: UploadFile, allow: Iterable[str]) -> str:
    ext = MIME_TO_EXT.get((upload.content_type or "").lower(), "")
    if not ext:
        name_only = Path(PurePath(upload.filename or "")).name
        ext = Path(name_only).suffix.lower()
    if not ext or ext not in allow:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext or 'unknown'}")
    return ext

def _safe_save_upload(upload: UploadFile, allow_exts: Iterable[str]) -> Path:
    ext = _safe_ext(upload, allow_exts)
    uid = uuid.uuid4().hex
    dest = (TMP_DIR / f"{uid}{ext}").resolve()
    tmp_root = TMP_DIR.resolve()
    if not str(dest).startswith(str(tmp_root) + os.sep):
        raise HTTPException(status_code=400, detail="Invalid upload destination")

    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    fd = os.open(str(dest), flags, 0o600)
    try:
        with os.fdopen(fd, "wb") as f:
            shutil.copyfileobj(upload.file, f, length=1024 * 1024)
    finally:
        try:
            upload.file.close()
        except Exception:
            pass
    return dest

# ---------- Models ----------
class CommentaryPrepResponse(BaseModel):
    context: Dict[str, Any]
    ocr: Dict[str, Any]
    used_stub: bool

class CommentaryRunResponse(BaseModel):
    id: str
    text: str
    audio_url: Optional[str] = None
    video_url: Optional[str] = None
    meta: Dict[str, Any]

# ---------- Helpers ----------
def _to_wav_from_mp3(mp3_path: Path) -> Path:
    """Convert MP3 -> 16k mono WAV for muxing where needed."""
    wav_path = mp3_path.with_suffix(".wav")
    cmd = f'ffmpeg -y -i {shlex.quote(str(mp3_path))} -ac 1 -ar 16000 {shlex.quote(str(wav_path))}'
    subprocess.run(shlex.split(cmd), check=True)
    return wav_path

# ---------- Endpoints ----------
@router.post("/commentary-from-video", response_model=CommentaryPrepResponse)
def commentary_from_video(
    video: UploadFile = File(...),
    viz: bool = Form(False),
    dx: int = Form(0),
    dy: int = Form(0),
    t: float = Form(0.10),
):
    dest = _safe_save_upload(video, VIDEO_EXTS)
    keep_uploads = os.getenv("KEEP_UPLOADS", "0") == "1"
    try:
        ocr = extract_scoreboard_from_video(str(dest), viz=viz, dx=dx, dy=dy, t=t)
        ctx = ocr_to_commentary_context(ocr)
        return CommentaryPrepResponse(context=ctx, ocr=ocr, used_stub=bool(ocr.get("used_stub", False)))
    finally:
        if not keep_uploads:
            try:
                dest.unlink(missing_ok=True)
            except Exception:
                pass

@router.post("/run-commentary-from-video", response_model=CommentaryRunResponse)
def run_commentary_from_video(
    video: UploadFile = File(...),
    viz: bool = Form(False),
    dx: int = Form(0),
    dy: int = Form(0),
    t: float = Form(0.10),
    # allow optional override of tone/bias; falls back to defaults in context_from_ocr
    tone: Optional[str] = Form(None),
    bias: Optional[str] = Form(None),
    audio_only: bool = Form(False),
):
    """
    One-shot pipeline:
      upload video -> OCR -> build context -> LLM -> TTS -> (optional) mux
    Returns the same shape as AnalyzeOut for consistency with your UI.
    """
    dest = _safe_save_upload(video, VIDEO_EXTS)
    keep_uploads = os.getenv("KEEP_UPLOADS", "0") == "1"
    run_id = f"run_{uuid.uuid4().hex[:10]}"

    try:
        # 1) OCR -> context
        ocr = extract_scoreboard_from_video(str(dest), viz=viz, dx=dx, dy=dy, t=t)
        ctx = ocr_to_commentary_context(ocr)
        if tone:
            ctx["tone"] = tone
        if bias:
            ctx["bias"] = bias

        # 2) LLM
        llm = get_llm()
        prompt = build_llm_prompt(ctx)
        text = llm.generate(prompt, meta={"tone": normalize_tone(str(ctx.get("tone", "play-by-play"))),
                                          "bias": str(ctx.get("bias", "neutral")).lower()})

        # 3) TTS (mp3)
        tts = get_tts()
        audio_mp3_path = tts.synth_to_file(text, DATA_DIR / "uploads" / "tts",
                                           tone=str(ctx.get("tone", "play-by-play")),
                                           bias=str(ctx.get("bias", "neutral")))  # type: ignore[call-arg]
        audio_mp3 = Path(audio_mp3_path)

        # 4) Mux (if not audio-only)
        video_out: Optional[Path] = None
        if not audio_only and audio_mp3.exists():
            audio_wav = _to_wav_from_mp3(audio_mp3)
            vid_out_path = mux_audio_video(str(dest), str(audio_wav))
            video_out = Path(vid_out_path) if isinstance(vid_out_path, str) else vid_out_path

        # 5) Build response (align with AnalyzeOut)
        return CommentaryRunResponse(
            id=run_id,
            text=text,
            audio_url=(to_static_url(audio_mp3) if (audio_mp3 and audio_mp3.exists()) else None),
            video_url=(to_static_url(video_out) if video_out else None),
            meta={
                "usedManualContext": False,
                "provider": {"llm": type(llm).__name__, "tts": type(tts).__name__},
                "prompt_tone": normalize_tone(str(ctx.get("tone", "play-by-play"))),
                "prompt_bias": str(ctx.get("bias", "neutral")).lower(),
                "audio_only": audio_only,
                "ocr_used_stub": bool(ocr.get("used_stub", False)),
                "ocr_raw": ocr,  # helpful for debugging
            },
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {e}")
    finally:
        if not keep_uploads:
            try:
                dest.unlink(missing_ok=True)
            except Exception:
                pass
