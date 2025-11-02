# backend/routers/pipeline_api.py
from __future__ import annotations
from fastapi import APIRouter, UploadFile, File, HTTPException, Form, Query
from pydantic import BaseModel
from typing import Optional, Iterable, Dict, Any, List
from pathlib import Path, PurePath
import shutil, os, uuid, subprocess, shlex, json

from backend.services.ocr_paddle import extract_scoreboard_from_video_paddle
from backend.services.context_from_ocr import ocr_to_commentary_context
from backend.services.llm_providers import get_llm
from backend.services.tts_providers import get_tts
from backend.services.tone_profiles import build_llm_prompt as build_llm_prompt_scoreboard
from backend.services.tone_profiles import normalize_tone
from backend.services.mux import mux_audio_video
from backend.main import DATA_DIR, to_static_url

# Gameplay recognizer & device select
from mr_tai_gameplay.src.pipeline_infer import run_inference_multi

try:
    import torch
except Exception:
    torch = None  # torch optional for device auto-select

try:
    import cv2
except Exception:
    cv2 = None

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

def _select_device() -> str:
    # Env override: MODEL_DEVICE=cpu|cuda|mps
    forced = (os.getenv("MODEL_DEVICE") or "").lower().strip()
    if forced in {"cpu", "cuda", "mps"}:
        return forced
    if torch is not None:
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    return "cpu"

def _video_duration_seconds(path: Path) -> float:
    try:
        if cv2 is None:
            return 0.0
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return 0.0
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        nframes = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        cap.release()
        return float(nframes / fps) if (fps > 0 and nframes > 0) else 0.0
    except Exception:
        return 0.0

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
        # Merge new OCR fields into context if provided
        for _k in ("down", "distance", "distance_text", "yardline"):
            if ocr.get(_k) is not None:
                ctx[_k] = ocr[_k]
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
        # Merge new OCR fields into context if provided
        for _k in ("down", "distance", "distance_text", "yardline"):
            if ocr.get(_k) is not None:
                ctx[_k] = ocr[_k]
        if tone:
            ctx["tone"] = tone
        if bias:
            ctx["bias"] = bias

        tone_val = normalize_tone(str(ctx.get("tone", "neutral")))
        gender_val = (gender or "male").strip().lower()
        if gender_val not in ("male", "female"):
            gender_val = "male"

        # 2) Gameplay recognizer → event timeline (multi-segment)
        device = _select_device()
        ckpt = os.getenv("MODEL_CKPT", "out_ckpt/r3d18_ep10.pt")
        if ckpt and not Path(ckpt).exists():
            ckpt = None  # fall back gracefully

        gp_json = (TMP_DIR / f"{run_id}_gp.json").resolve()
        run_inference_multi(
            str(dest),
            str(gp_json),
            device=device,
            ckpt=ckpt,
            segmentation="peaks",
            frames_per_window=32,
        )
        gp_payload = json.loads(gp_json.read_text()) if gp_json.exists() else {}

        # --- RAW timeline for the LLM to repair (do not sanitize here) ---
        events: List[Dict[str, Any]] = gp_payload.get("gameplay_only", {}).get("events", [])
        raw_timeline_lines: List[str] = []
        for ev in sorted(events, key=lambda e: float(e.get("t_start", 0.0))):
            t0 = float(ev.get("t_start", 0.0))
            t1 = float(ev.get("t_end", 0.0))
            lbl = str(ev.get("primary_label", "no_event")).replace("_", " ")
            # Keep raw, even if end<=start or end==0; the LLM will repair
            raw_timeline_lines.append(f"- [{t0:0.2f}s → {t1:0.2f}s] {lbl}")
        raw_timeline_block = "\n".join(raw_timeline_lines) if raw_timeline_lines else "- (no detected events)"

        # Determine total duration (upper bound for t_say)
        duration = _video_duration_seconds(dest)
        if duration <= 0.0 and events:
            duration = max(float(ev.get("t_end", 0.0)) for ev in events)

        # 3) LLM — scoreboard prompt + timeline REPAIR + pacing + strict JSON lines
        llm = get_llm()
        base_prompt = build_llm_prompt_scoreboard(ctx)
        augmented = f"""{base_prompt}

---
GAMEPLAY TIMELINE (raw; may include noise or invalid windows — you must repair):
{raw_timeline_block}

DATA QUALITY & REPAIR RULES
- Treat the timeline above as noisy input.
- Ignore any window where end <= start, end == 0, or duration < 0.5s.
- Merge adjacent/overlapping windows with the same action label if within 1.0s.
- Prefer representative labels in this order when merging:
  touchdown > interception > fumble > sack > complete pass > incomplete pass > pass attempt > run > QB scramble > big hit > special teams results.
- Never hallucinate team names, yardage, penalties, or scores that are not present in CONTEXT.
- Constrain all times to [0, {duration:0.2f}].

PACING RULES
- Produce one commentary line for each repaired window at about (window_end + 0.30s).
- If a gap between repaired windows is ≥ 2.0s, insert one brief filler line mid-gap (single short clause).
- Keep lines punchy (broadcast play-by-play), max 1–2 exclamations total.

OUTPUT FORMAT (strict):
Return a JSON object with a single key "lines" whose value is an array of objects:
  {{"lines":[{{"t_say": <seconds float>, "text": "<one sentence>"}}, ...]}}
Constraints:
- Use only the repaired windows you decided on.
- t_say must be non-decreasing and within [0, {duration:0.2f}].
- One sentence per line.
- No extra keys. No code block fences.
- If you cannot produce valid JSON, return: {{"lines":[{{"t_say":0.30,"text":"No commentary."}}]}}
"""

        model_out = llm.generate(
            augmented,
            meta={"tone": tone_val, "bias": str(ctx.get("bias", "neutral")).lower(), "gender": gender_val},
        )

        # Parse structured JSON -> stitched text for current TTS
        parsed_lines: Optional[List[Dict[str, Any]]] = None
        stitched_text = model_out
        try:
            obj = json.loads(model_out)
            if isinstance(obj, dict) and isinstance(obj.get("lines"), list):
                parsed_lines = [
                    {
                        "t_say": float(max(0.0, min(duration, float(l.get("t_say", 0.0))))),
                        "text": str(l.get("text", "")).strip(),
                    }
                    for l in obj["lines"]
                    if isinstance(l, dict) and str(l.get("text", "")).strip()
                ]
                parsed_lines.sort(key=lambda x: x["t_say"])
                stitched_text = " ".join(l["text"] for l in parsed_lines)
        except Exception:
            parsed_lines = None

        # 4) TTS selection
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
            stitched_text,
            DATA_DIR / "uploads" / "tts",
            tone=tone_val,
            bias=str(ctx.get("bias", "neutral")),
            voice_id=voice_id,
            emotion_preset=emotion_preset,
        )  # type: ignore[arg-type]
        audio_mp3 = Path(audio_mp3_path)

        # 5) Mux (if not audio-only)
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
            text=stitched_text,
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
                "events_used": events,     # helpful for UI/debug
                "llm_lines": parsed_lines, # final timed script (if parsed)
                "device": device,
                "ckpt": ckpt,
            } | meta_extra,
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
