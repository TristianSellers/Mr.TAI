# backend/routers/analyze_commentate.py
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from pathlib import Path
import uuid, time, subprocess, shlex

from backend.services.llm_providers import get_llm
from backend.services.tts_providers import get_tts
from backend.services.runlog import append_run_log
from backend.services.mux import mux_audio_video
from backend.services.tone_profiles import build_llm_prompt, normalize_tone  # <— NEW

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
    file: UploadFile = File(...),
    home_team: str | None = Form(default=None),
    away_team: str | None = Form(default=None),
    score: str | None = Form(default=None),
    quarter: str | None = Form(default=None),
    clock: str | None = Form(default=None),
    tone: str = Form(default="play-by-play"),
    voice: str = Form(default="default"),
):
    t0 = time.time()
    run_id = f"run_{uuid.uuid4().hex[:10]}"
    errors: list[str] = []

    # 1) Save input
    try:
        ext = _safe_ext(file.filename or "clip.mp4")
        in_path = INPUT_DIR / f"{run_id}{ext}"
        with in_path.open("wb") as f:
            f.write(await file.read())
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to save input: {e}")

    context = {
        "home_team": home_team,
        "away_team": away_team,
        "score": score,
        "quarter": quarter,
        "clock": clock,
        "tone": tone,
    }

    # 2) Generate text via LLM (tone-aware)
    try:
        llm = get_llm()
        prompt = build_llm_prompt(context)  # tone handled inside
        text = llm.generate(prompt, meta={"tone": normalize_tone(tone)})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM error: {e}")

    audio_wav: Path | None = None
    audio_mp3: Path | None = None
    video_out: Path | None = None
    tts = None

    # 3) TTS (provider-based mp3)
    try:
        tts = get_tts()
        audio_mp3_path = tts.synth_to_file(text, TTS_DIR, tone=tone)  # <— pass tone  # returns MP3 path
        audio_mp3 = Path(audio_mp3_path)
    except Exception as e:
        errors.append(f"TTS error: {e}")
        audio_mp3 = None

    # 4) Mux (convert mp3 -> wav if needed, then mux)
    if audio_mp3 and audio_mp3.exists():
        try:
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
        "input_clip": str(in_path),
        "context": context,
        "text": text,
        "audio_wav": str(audio_wav) if audio_wav else None,
        "audio_mp3": str(audio_mp3) if audio_mp3 else None,
        "video_path": str(video_out) if video_out else None,
        "errors": errors or None,
        "elapsed_s": elapsed,
    })

    tts_name = type(tts).__name__ if tts else "Unavailable"

    return AnalyzeOut(
        id=run_id,
        text=text,
        audio_url=(to_static_url(audio_mp3) if (audio_mp3 and audio_mp3.exists()) else
                   (to_static_url(audio_wav) if (audio_wav and audio_wav.exists()) else None)),
        video_url=(to_static_url(video_out) if video_out else None),
        meta={
            "usedManualContext": any([home_team, away_team, score, quarter, clock]),
            "duration_s": elapsed,
            "provider": {"llm": type(llm).__name__, "tts": tts_name},
            "prompt_tone": normalize_tone(tone),
            "errors": errors or None,
        },
    )
