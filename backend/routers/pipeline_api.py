# backend/routers/pipeline_api.py
from __future__ import annotations
from fastapi import APIRouter, UploadFile, File, HTTPException, Form, Query
from pydantic import BaseModel
from typing import Optional, Iterable, Dict, Any, List
from pathlib import Path, PurePath
import shutil, os, uuid, subprocess, shlex

from backend.services.ocr_paddle import extract_scoreboard_from_video_paddle
from backend.services.context_from_ocr import ocr_to_commentary_context
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

# ---------- Voice catalog & mapping ----------
VOICE_CATALOG: List[Dict[str, Any]] = [
    {"id": "tc_673eb45cdc1073aef51e6b90", "name": "Dean",    "gender": "male",   "emotions": ["tonedown", "normal", "toneup"]},
    {"id": "tc_6412c42d733f60ab8ad369a9", "name": "Caitlyn", "gender": "female", "emotions": ["normal"]},
    {"id": "tc_6837b58f80ceeb17115bb771", "name": "Walter",  "gender": "male",   "emotions": ["normal"]},
    {"id": "tc_684a5a7ba2ce934624b59c6e", "name": "Nia",     "gender": "female", "emotions": ["normal", "tonedown"]},
    {"id": "tc_623145940c2c1ff30c30f3a9", "name": "Matthew", "gender": "male",   "emotions": ["normal", "shout"]},
    {"id": "tc_630494521f5003bebbfdafef", "name": "Rachel",  "gender": "female", "emotions": ["normal", "toneup", "happy"]},
]

VOICE_BY_TONE_GENDER = {
    "neutral": {"male": "tc_673eb45cdc1073aef51e6b90", "female": "tc_6412c42d733f60ab8ad369a9"},
    "radio":   {"male": "tc_6837b58f80ceeb17115bb771", "female": "tc_684a5a7ba2ce934624b59c6e"},
    "hype":    {"male": "tc_623145940c2c1ff30c30f3a9", "female": "tc_630494521f5003bebbfdafef"},
}

PREFERRED_BY_TONE = {
    "neutral": ["normal"],
    "radio":   ["tonedown", "normal", "tonemid"],
    "hype":    ["shout", "toneup", "happy", "normal"],
}

def _pick_emotion(tone: str, supported: List[str]) -> str:
    for e in PREFERRED_BY_TONE.get(tone, ["normal"]):
        if e in supported:
            return e
    return "normal"

def _to_wav_from_mp3(mp3_path: Path) -> Path:
    wav_path = mp3_path.with_suffix(".wav")
    cmd = f'ffmpeg -y -i {shlex.quote(str(mp3_path))} -ac 1 -ar 16000 {shlex.quote(str(wav_path))}'
    subprocess.run(shlex.split(cmd), check=True)
    return wav_path

# ---------- Voice option & preview (for UI) ----------
@router.get("/voice-options")
def voice_options(
    tone: Optional[str] = Query(None),
    gender: Optional[str] = Query(None),
):
    tone_val = normalize_tone(str(tone or "neutral"))
    items = [v for v in VOICE_CATALOG if (not gender or v["gender"] == gender)]
    out = [{"suggested_emotion": _pick_emotion(tone_val, v["emotions"]), **v} for v in items]
    return {"voices": out, "tone": tone_val, "gender": (gender or "").lower() or None}

@router.post("/voice-preview")
def voice_preview(
    tone: Optional[str] = Form("neutral"),
    bias: Optional[str] = Form("neutral"),
    gender: Optional[str] = Form("male"),
    text: Optional[str] = Form("Mic check. One-two."),
    voice_id: Optional[str] = Form(None),
    emotion_preset: Optional[str] = Form(None),
):
    tone_val = normalize_tone(str(tone or "neutral"))
    gender_val = (gender or "male").strip().lower()
    if gender_val not in ("male", "female"):
        gender_val = "male"

    if not voice_id:
        voice_id = VOICE_BY_TONE_GENDER.get(tone_val, {}).get(gender_val)
    if not voice_id:
        raise HTTPException(status_code=400, detail="No voice_id available for given tone/gender")

    if not emotion_preset:
        v = next((v for v in VOICE_CATALOG if v["id"] == voice_id), None)
        supported = list(v["emotions"]) if v else ["normal"]
        emotion_preset = _pick_emotion(tone_val, supported)

    tts = get_tts()
    out_path = tts.synth_to_file(
        text or "Mic check. One-two.",
        DATA_DIR / "uploads" / "tts",
        tone=tone_val,
        bias=str(bias or "neutral"),
        voice_id=voice_id,
        emotion_preset=emotion_preset,
    )  # type: ignore[arg-type]

    return {
        "audio_url": to_static_url(Path(out_path)),
        "meta": {"tone": tone_val, "gender": gender_val, "voice_id": voice_id, "emotion": emotion_preset},
    }

# ---------- Prep ----------
@router.post("/commentary-from-video", response_model=CommentaryPrepResponse)
def commentary_from_video(
    video: UploadFile = File(...),
    viz: bool = Form(False),
    dx: int = Form(0),   # legacy compat
    dy: int = Form(0),
    t: float = Form(0.10),
    scoreboard_type: Optional[str] = Form(None),  # "MNP" | "TNP" | "SNP" | None
    debug_prompt: Optional[int] = Query(None),
):
    dest = _safe_save_upload(video, VIDEO_EXTS)
    keep_uploads = os.getenv("KEEP_UPLOADS", "0") == "1"
    try:
        ocr = extract_scoreboard_from_video_paddle(
            str(dest), t=t, attempts=3, viz=viz, profile_key=scoreboard_type
        )
        ctx = ocr_to_commentary_context(ocr)
        return CommentaryPrepResponse(context=ctx, ocr=ocr, used_stub=bool(ocr.get("used_stub", False)))
    finally:
        if not keep_uploads:
            try:
                dest.unlink(missing_ok=True)
            except Exception:
                pass

# ---------- Run ----------
@router.post("/run-commentary-from-video", response_model=CommentaryRunResponse)
def run_commentary_from_video(
    video: UploadFile = File(...),
    viz: bool = Form(False),
    dx: int = Form(0),
    dy: int = Form(0),
    t: float = Form(0.10),
    tone: Optional[str] = Form(None),
    bias: Optional[str] = Form(None),
    gender: Optional[str] = Form(None),
    audio_only: bool = Form(False),
    voice_id: Optional[str] = Form(None),
    emotion_preset: Optional[str] = Form(None),
    scoreboard_type: Optional[str] = Form(None),  # "MNP" | "TNP" | "SNP" | None
    debug_prompt: Optional[int] = Query(None),
):
    dest = _safe_save_upload(video, VIDEO_EXTS)
    keep_uploads = os.getenv("KEEP_UPLOADS", "0") == "1"
    run_id = f"run_{uuid.uuid4().hex[:10]}"

    try:
        # 1) OCR via PaddleOCR (override or auto-detect)
        ocr = extract_scoreboard_from_video_paddle(
            str(dest), t=t, attempts=3, viz=viz, profile_key=scoreboard_type
        )
        ctx = ocr_to_commentary_context(ocr)
        if tone:
            ctx["tone"] = tone
        if bias:
            ctx["bias"] = bias

        tone_val = normalize_tone(str(ctx.get("tone", "neutral")))
        gender_val = (gender or "male").strip().lower()
        if gender_val not in ("male", "female"):
            gender_val = "male"

        # 2) LLM — build prompt & (optionally) save it
        llm = get_llm()
        prompt = build_llm_prompt(ctx)

        if os.getenv("DEBUG_PROMPT", "0") == "1" or (debug_prompt and int(debug_prompt) == 1):
            print("\n=== LLM PROMPT (run_id:", run_id, ") ===\n")
            print(prompt)
            print("\n=== END PROMPT ===\n")
            out_dir = DATA_DIR / "tmp" / "prompts"
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / f"prompt_{run_id}.txt").write_text(prompt, encoding="utf-8")

        text = llm.generate(
            prompt,
            meta={"tone": tone_val, "bias": str(ctx.get("bias", "neutral")).lower(), "gender": gender_val},
        )

        # 3) TTS selection
        if not voice_id:
            voice_id = VOICE_BY_TONE_GENDER.get(tone_val, {}).get(gender_val)
        if not voice_id:
            raise HTTPException(status_code=400, detail=f"No mapped voice for tone={tone_val} gender={gender_val}")

        if not emotion_preset:
            v = next((v for v in VOICE_CATALOG if v["id"] == voice_id), None)
            supported = list(v["emotions"]) if v else ["normal"]
            emotion_preset = _pick_emotion(tone_val, supported)

        tts = get_tts()
        audio_mp3_path = tts.synth_to_file(
            text,
            DATA_DIR / "uploads" / "tts",
            tone=tone_val,
            bias=str(ctx.get("bias", "neutral")),
            voice_id=voice_id,
            emotion_preset=emotion_preset,
        )  # type: ignore[arg-type]
        audio_mp3 = Path(audio_mp3_path)

        # 4) Mux (if not audio-only)
        video_out: Optional[Path] = None
        if not audio_only and audio_mp3.exists():
            audio_wav = _to_wav_from_mp3(audio_mp3)
            vid_out_path = mux_audio_video(str(dest), str(audio_wav))
            video_out = Path(vid_out_path) if isinstance(vid_out_path, str) else vid_out_path

        # Surface OCR extras (viz dir / preset) in meta if present
        meta_extra: Dict[str, Any] = {}
        for k in ("_viz_dir", "_preset"):
            if isinstance(ocr.get(k), str):
                meta_extra[k] = ocr[k]

        return CommentaryRunResponse(
            id=run_id,
            text=text,
            audio_url=(to_static_url(audio_mp3) if (audio_mp3 and audio_mp3.exists()) else None),
            video_url=(to_static_url(video_out) if video_out else None),
            meta={
                "usedManualContext": False,
                "provider": {"llm": type(llm).__name__, "tts": type(tts).__name__, "ocr": "paddleocr"},
                "prompt_tone": tone_val,
                "prompt_bias": str(ctx.get("bias", "neutral")).lower(),
                "gender": gender_val,
                "selected_voice_id": voice_id,
                "selected_emotion": emotion_preset,
                "audio_only": audio_only,
                "ocr_used_stub": bool(ocr.get("used_stub", False)),
                "ocr_raw": ocr,
                "prompt_file": f"/data/tmp/prompts/prompt_{run_id}.txt"
                    if os.getenv("DEBUG_PROMPT", "0") == "1" or (debug_prompt and int(debug_prompt) == 1)
                    else None,
                **meta_extra,
            },
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {e}")
    finally:
        if not keep_uploads:
            try:
                dest.unlink(missing_ok=True)
            except Exception:
                pass
