# backend/routers/analyze_commentate.py
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from pathlib import Path
import uuid, time, subprocess, shlex

from backend.services.llm_providers import get_llm
from backend.services.tts_providers import get_tts
from backend.services.runlog import append_run_log
from backend.services.mux import mux_audio_video
from backend.services.tone_profiles import build_llm_prompt, normalize_tone

# pull shared objects from main
from backend.main import DATA_DIR, UPLOAD_DIR, to_static_url

router = APIRouter()

INPUT_DIR = UPLOAD_DIR / "input"
TTS_DIR   = UPLOAD_DIR / "tts"
DUB_DIR   = UPLOAD_DIR / "dubbed"
for d in (INPUT_DIR, TTS_DIR, DUB_DIR):
    d.mkdir(parents=True, exist_ok=True)

class AnalyzeOut(BaseModel):
    id: str
    text: str
    audio_url: str | None = None
    video_url: str | None = None
    meta: dict

def _safe_ext(name: str, default: str = ".mp4") -> str:
    s = (name or "").lower()
    for ext in (".mp4", ".mov", ".mkv", ".webm", ".mpg", ".mpeg"):
        if s.endswith(ext):
            return ext
    return default

def _wav_to_mp3(wav_path: Path) -> Path:
    mp3_path = wav_path.with_suffix(".mp3")
    cmd = f'ffmpeg -y -i {shlex.quote(str(wav_path))} -b:a 128k {shlex.quote(str(mp3_path))}'
    subprocess.run(shlex.split(cmd), check=True)
    return mp3_path

@router.post("/analyze_commentate", response_model=AnalyzeOut)
async def analyze_commentate(
    file: UploadFile | None = File(default=None),       # optional for audio-only runs
    home_team: str | None = Form(default=None),
    away_team: str | None = Form(default=None),
    score: str | None = Form(default=None),
    quarter: str | None = Form(default=None),
    clock: str | None = Form(default=None),
    tone: str = Form(default="play-by-play"),
    bias: str = Form(default="neutral"),                # NEW: neutral | home | away
    voice: str = Form(default="default"),
    audio_only: bool = Form(default=False),
):
    t0 = time.time()
    run_id = f"run_{uuid.uuid4().hex[:10]}"
    errors: list[str] = []

    in_path: Path | None = None
    if file is not None:
        # 1) Save input (only when a clip is provided)
        try:
            ext = _safe_ext(file.filename or "clip.mp4")
            in_path = INPUT_DIR / f"{run_id}{ext}"
            with in_path.open("wb") as f:
                f.write(await file.read())
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to save input: {e}")
    else:
        if not audio_only:
            # No file and not audio-only → invalid request
            raise HTTPException(status_code=400, detail="No clip provided and audio_only=false")

    context = {
        "home_team": home_team,
        "away_team": away_team,
        "score": score,
        "quarter": quarter,
        "clock": clock,
        "tone": tone,
        "bias": bias,  # NEW: threaded through to the prompt builder
    }

    # 2) Generate text via LLM (tone/bias-aware)
    try:
        llm = get_llm()
        prompt = build_llm_prompt(context)  # expects tone/bias in context
        text = llm.generate(prompt, meta={"tone": normalize_tone(tone), "bias": (bias or "neutral").lower()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM error: {e}")

    audio_wav: Path | None = None
    audio_mp3: Path | None = None
    video_out: Path | None = None
    tts = None

    # 3) TTS (provider-based mp3)
    try:
        tts = get_tts()
        # Pass tone/bias as optional kwargs; providers that don't use them will ignore safely
        audio_mp3_path = tts.synth_to_file(text, TTS_DIR, tone=tone, bias=bias)  # type: ignore[call-arg]
        audio_mp3 = Path(audio_mp3_path)
    except Exception as e:
        errors.append(f"TTS error: {e}")
        audio_mp3 = None

    # 4) Mux — only if we have a video input and not audio-only
    if (not audio_only) and in_path and in_path.exists() and audio_mp3 and audio_mp3.exists():
        try:
            # Convert MP3 → WAV (mono 16k) if mux expects wav
            audio_wav = audio_mp3.with_suffix(".wav")
            cmd = f'ffmpeg -y -i {shlex.quote(str(audio_mp3))} -ac 1 -ar 16000 {shlex.quote(str(audio_wav))}'
            subprocess.run(shlex.split(cmd), check=True)

            video_out_path = mux_audio_video(str(in_path), str(audio_wav))
            video_out = Path(video_out_path) if isinstance(video_out_path, str) else video_out_path
        except Exception as e:
            errors.append(f"Mux error: {e}")

    # 5) Log
    elapsed = round(time.time() - t0, 3)
    append_run_log({
        "id": run_id,
        "input_clip": str(in_path) if in_path else None,
        "context": context,
        "text": text,
        "audio_wav": str(audio_wav) if audio_wav else None,
        "audio_mp3": str(audio_mp3) if audio_mp3 else None,
        "video_path": str(video_out) if video_out else None,
        "audio_only": audio_only,
        "errors": errors or None,
        "elapsed_s": elapsed,
    })

    tts_name = type(tts).__name__ if tts else "Unavailable"

    return AnalyzeOut(
        id=run_id,
        text=text,
        audio_url=(to_static_url(audio_mp3) if (audio_mp3 and audio_mp3.exists()) else None),
        video_url=(to_static_url(video_out) if video_out else None),
        meta={
            "usedManualContext": any([home_team, away_team, score, quarter, clock]),
            "duration_s": elapsed,
            "provider": {"llm": type(llm).__name__, "tts": tts_name},
            "prompt_tone": normalize_tone(tone),
            "prompt_bias": (bias or "neutral").lower(),  # NEW: surfaced to UI
            "audio_only": audio_only,
            "errors": errors or None,
        },
    )
