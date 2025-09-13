from backend.core.models import Job, JobState
from backend.core.registry import jobs
from backend.providers.tts_local import synth_to_wav
from backend.steps.script import segments_to_commentary

from fastapi import FastAPI, UploadFile, File, HTTPException, status, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

from faster_whisper import WhisperModel
from functools import lru_cache
from pydantic import BaseModel
from pathlib import Path
from typing import Dict, Literal, Optional
from datetime import datetime
from dotenv import load_dotenv

from .middleware_logging import register_request_logging
from .error_handlers import register_error_handlers

from collections import Counter
import uuid
import shutil
import os
import ffmpeg
import mimetypes
import re
import subprocess
import shlex

# =========================
# ---- Config / Env ----
# =========================
ROOT = Path(__file__).resolve().parents[1]
# load in increasing precedence
load_dotenv(ROOT / ".env")
load_dotenv(ROOT / ".env.local", override=True)
load_dotenv(ROOT / "backend" / ".env", override=True)
load_dotenv(ROOT / "backend" / ".env.local", override=True)

REPO_ROOT = Path(__file__).resolve().parents[1]  # .../Mr.TAI

ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "http://localhost:5173").split(",") if o.strip()]
PORT = int(os.getenv("PORT", "8000"))

DATA_DIR = Path(os.getenv("DATA_DIR", REPO_ROOT / "data"))
UPLOAD_DIR = DATA_DIR / "uploads"
DEMO_DIR = DATA_DIR / "demo"

ASR_MODEL_NAME = os.getenv("ASR_MODEL_NAME", "tiny")
ASR_DEVICE = os.getenv("ASR_DEVICE", "cpu")              # "cpu" or "auto"
ASR_COMPUTE_TYPE = os.getenv("ASR_COMPUTE_TYPE", "int8") # e.g. "int8", "int8_float16", "float16", "float32"

MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "200"))

# Ensure dirs
DATA_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
DEMO_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# ---- App Init ----
# =========================
app = FastAPI(title="Mr. TAI Backend", version="0.1.0")
register_request_logging(app)
register_error_handlers(app)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve anything inside DATA_DIR at /static/*
app.mount("/static", StaticFiles(directory=str(DATA_DIR)), name="static")


@app.get("/")
def root():
    return {"message": "Hello, I am Mr. TAI!\nMr. TAI stands for (M)ulti-media (R)eal (T)ime AI!"}


def _safe_name(name: str) -> str:
    base = os.path.basename(name)
    return "".join(c for c in base if c.isalnum() or c in ("-", "_", ".", " ")).strip() or "file"


class ProcessRequest(BaseModel):
    filename: str  # should match "stored_name" returned by /upload


def _probe_duration_seconds(path: str | os.PathLike[str]) -> float:
    path_str = str(path)
    try:
        out = subprocess.check_output([
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", path_str
        ])
        return float(out.decode().strip())
    except Exception:
        return 0.0


def _is_audio(filename: str) -> bool:
    mime, _ = mimetypes.guess_type(filename)
    return (mime or "").startswith("audio/")


def _is_video(filename: str) -> bool:
    mime, _ = mimetypes.guess_type(filename)
    return (mime or "").startswith("video/")


def to_static_url(p: str | os.PathLike[str]) -> str:
    """Convert any absolute/relative path under DATA_DIR to /static URL."""
    p_abs = Path(p).resolve()
    data_abs = DATA_DIR.resolve()
    rel = p_abs.relative_to(data_abs)  # raises if not under DATA_DIR
    return f"/static/{rel.as_posix()}"


def _ffprobe_can_decode(path: str) -> bool:
    try:
        cmd = f'ffprobe -v error -show_streams -of json {shlex.quote(path)}'
        out = subprocess.check_output(cmd, shell=True)
        return b'"streams":' in out and b'codec_type' in out
    except Exception:
        return False


def _extract_audio_to_wav(in_path: str, out_dir: Path) -> Path:
    out_path = out_dir / f"audio_{uuid.uuid4().hex}.wav"
    (
        ffmpeg
        .input(in_path)
        .output(str(out_path), ac=1, ar=16000, format="wav", loglevel="error")
        .overwrite_output()
        .run()
    )
    return out_path


# very small stopword list for quick keywords
_STOP = {
    "the","a","an","and","or","but","if","in","on","at","to","of","for","with",
    "is","it","this","that","be","as","are","was","were","by","from","you","i",
    "we","they","he","she","him","her","them","our","your","their"
}


def _top_keywords(text: str, k: int = 5):
    words = re.findall(r"[a-zA-Z']+", text.lower())
    words = [w for w in words if w not in _STOP and len(w) > 2]
    return [w for w, _ in Counter(words).most_common(k)]


@lru_cache(maxsize=1)
def get_asr_model() -> WhisperModel:
    return WhisperModel(ASR_MODEL_NAME, device=ASR_DEVICE, compute_type=ASR_COMPUTE_TYPE)


# =========================
# ---- Routes ----
# =========================

@app.post("/upload", status_code=status.HTTP_201_CREATED)
async def upload_file(file: UploadFile = File(...)):
    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    safe = _safe_name(file.filename)
    unique = f"{uuid.uuid4().hex}_{safe}"
    out_path = UPLOAD_DIR / unique

    with out_path.open("wb") as buf:
        shutil.copyfileobj(file.file, buf)

    return {"message": f"Uploaded '{safe}'", "saved_to": str(out_path), "stored_name": unique}


def _simulate_processing(job_id: str):
    import time
    job = jobs[job_id]
    try:
        job.state = JobState.processing
        time.sleep(2)
        job.state = JobState.done
    except Exception as ex:
        job.state = JobState.error
        job.error = str(ex)


@app.post("/process")
def process_file(req: ProcessRequest, background: BackgroundTasks):
    src_path = UPLOAD_DIR / req.filename
    if not src_path.exists():
        raise HTTPException(status_code=404, detail="Uploaded file not found. Use 'stored_name' from /upload.")

    job = Job()
    job.ensure_layout()

    dest = job.dir() / "input" / Path(req.filename).name
    try:
        shutil.copy2(src_path, dest)
        job.input_name = dest.name
    except Exception as e:
        job.state = JobState.error
        job.error = f"Failed to stage input: {e}"
        jobs[job.id] = job
        return JSONResponse(job.to_api(), status_code=500)

    jobs[job.id] = job
    background.add_task(_simulate_processing, job.id)

    return {"message": f"Processing started for {req.filename}", "job_id": job.id, "status": job.state.value}


@app.get("/status/{job_id}")
def get_status(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job_id not found")

    artifacts = []
    for sub in ["events", "audio", "outputs"]:
        p = job.dir() / sub
        if p.exists():
            for x in p.glob("*"):
                if x.is_file():
                    artifacts.append(str(x.relative_to(job.dir())))

    payload = job.to_api()
    payload["artifacts"] = artifacts
    return payload


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "version": "0.1.0",
        "port": PORT,
        "dataDir": str(DATA_DIR.resolve()),
        "uploadDir": str(UPLOAD_DIR.resolve()),
        "allowedOrigins": ALLOWED_ORIGINS,
        "asr": {
            "model": ASR_MODEL_NAME,
            "device": ASR_DEVICE,
            "compute_type": ASR_COMPUTE_TYPE,
        },
    }


async def _write_upload_stream(dest_path: Path, up: UploadFile, max_mb: int = MAX_UPLOAD_MB):
    chunk_size = 1024 * 1024  # 1MB
    max_bytes = max_mb * 1024 * 1024
    written = 0
    with dest_path.open("wb") as buf:
        while True:
            chunk = await up.read(chunk_size)
            if not chunk:
                break
            written += len(chunk)
            if written > max_bytes:
                buf.close()
                try:
                    dest_path.unlink(missing_ok=True)
                except Exception:
                    pass
                raise HTTPException(status_code=413, detail=f"File too large (> {max_mb} MB)")
            buf.write(chunk)


@app.post("/process-upload")
async def process_upload(
    file: UploadFile = File(...),
    task: Literal["transcribe", "translate"] = "transcribe",
    language: Optional[str] = None,
    vad_filter: bool = True,
    word_timestamps: bool = False,
):
    if not file or not file.filename:
        return JSONResponse({"error": "No file provided"}, status_code=400)

    safe_base = _safe_name(file.filename)
    unique_name = f"{uuid.uuid4().hex}_{safe_base}"

    if not (_is_audio(safe_base) or _is_video(safe_base)):
        return JSONResponse({"error": "Unsupported file type. Please upload audio or video."}, status_code=400)

    in_path = UPLOAD_DIR / unique_name

    try:
        await _write_upload_stream(in_path, file)
    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse({"error": f"Failed to save upload: {e}"}, status_code=500)

    media_type = "video" if _is_video(safe_base) else "audio"

    try:
        if media_type == "video":
            audio_path = _extract_audio_to_wav(str(in_path), UPLOAD_DIR)
        else:
            audio_path = str(in_path)
    except Exception as e:
        try:
            in_path.unlink(missing_ok=True)
        except Exception:
            pass
        return JSONResponse({"error": f"Failed to extract audio: {e}"}, status_code=500)

    try:
        asr_model = get_asr_model()
        seg_iter, info = asr_model.transcribe(
            str(audio_path),
            task=task,
            language=language,
            vad_filter=vad_filter,
            beam_size=5,
            word_timestamps=word_timestamps,
        )

        seg_list = []
        full_text_parts = []
        for i, s in enumerate(seg_iter):
            item = {
                "id": i,
                "start": float(s.start),
                "end": float(s.end),
                "text": s.text.strip(),
            }
            if word_timestamps and getattr(s, "words", None):
                item["words"] = [
                    {"start": float(w.start), "end": float(w.end), "word": w.word}
                    for w in (s.words or [])
                ]
            seg_list.append(item)
            full_text_parts.append(item["text"])

        full_text = " ".join(full_text_parts).strip()

    except Exception as e:
        try:
            in_path.unlink(missing_ok=True)
            if isinstance(audio_path, Path):
                audio_path.unlink(missing_ok=True)
        except Exception:
            pass
        return JSONResponse({"error": f"Transcription failed: {e}"}, status_code=500)

    duration = _probe_duration_seconds(audio_path)
    keywords = _top_keywords(full_text, k=5)

    try:
        in_path.unlink(missing_ok=True)
        if isinstance(audio_path, Path):
            audio_path.unlink(missing_ok=True)
    except Exception:
        pass

    return {
        "message": "Processed successfully",
        "stored_name": unique_name,
        "input_file": safe_base,
        "media_type": media_type,
        "task": task,
        "forced_language": language,
        "detected_language": getattr(info, "language", None),
        "duration_sec": duration,
        "transcript": full_text[:2000],
        "segments": seg_list,
        "keywords": keywords,
        "settings": {
            "vad_filter": vad_filter,
            "beam_size": 5,
            "word_timestamps": word_timestamps,
            "model": ASR_MODEL_NAME,
            "device": ASR_DEVICE,
            "compute_type": ASR_COMPUTE_TYPE,
        },
    }


@app.post("/demo/say")
def demo_say(text: str = "Hello from Mr. TAI!"):
    demo_out = DEMO_DIR / "hello.wav"
    synth_to_wav(text, demo_out)
    return {
        "message": "ok",
        "audio": str(demo_out),
        "audio_url": to_static_url(demo_out),
    }


class CommentaryRequest(BaseModel):
    filename: str  # stored_name from /upload
    max_lines: int = 3


@app.post("/demo/commentary")
async def demo_commentary(req: CommentaryRequest):
    src = UPLOAD_DIR / req.filename
    if not src.exists():
        raise HTTPException(status_code=404, detail="Uploaded file not found")

    name = req.filename
    if not (_is_audio(name) or _is_video(name)):
        raise HTTPException(status_code=400, detail="Unsupported file type. Please upload audio or video.")

    if not _ffprobe_can_decode(str(src)):
        raise HTTPException(status_code=400, detail="File is not a valid/decodable media file.")

    # ensure WAV if video
    media_type = "video" if _is_video(req.filename) else "audio"
    if media_type == "video":
        audio_path = _extract_audio_to_wav(str(src), DATA_DIR)
        audio_path_str = str(audio_path)
    else:
        audio_path_str = str(src)

    # single ASR call
    asr_model = get_asr_model()
    seg_iter, _info = asr_model.transcribe(audio_path_str, task="transcribe", vad_filter=True, beam_size=5)

    segments = []
    for i, s in enumerate(seg_iter):
        segments.append({
            "id": i,
            "start": float(s.start),
            "end": float(s.end),
            "text": (s.text or "").strip()
        })

    text = segments_to_commentary(segments, max_lines=req.max_lines)

    out_path = DEMO_DIR / f"commentary_{uuid.uuid4().hex}.wav"
    synth_to_wav(text, out_path)

    # cleanup temp wav if created under DATA_DIR
    try:
        if media_type == "video":
            Path(audio_path_str).unlink(missing_ok=True)
    except Exception:
        pass

    return {
        "message": "ok",
        "commentary_text": text,
        "audio": str(out_path),
        "audio_url": to_static_url(out_path),
        "segments_used": min(req.max_lines, len(segments)),
    }


from backend.routers.analyze_commentate import router as analyze_router
app.include_router(analyze_router)

from backend.routers.demo import router as demo_router
app.include_router(demo_router)

from backend.routers.health import router as health_router
app.include_router(health_router)

from backend.routers.prewarm import router as prewarm_router
app.include_router(prewarm_router)

from backend.routers.prewarm import router as prewarm_router
app.include_router(prewarm_router)
