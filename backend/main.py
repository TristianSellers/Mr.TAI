from backend.core.models import Job, JobState  
from backend.core.registry import jobs         
from backend.providers.tts_local import synth_to_wav  
from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi import BackgroundTasks
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

# ---- Config / Env ----
load_dotenv()

allowed_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173").split(",")
port = int(os.getenv("PORT", 8000))

# Introduce DATA_DIR (base) and make UPLOAD_DIR live under it
REPO_ROOT = Path(__file__).resolve().parents[1]   # .../Mr.TAI
DATA_DIR = Path(os.getenv("DATA_DIR", REPO_ROOT / "data"))
UPLOAD_DIR = DATA_DIR / "uploads"
(DATA_DIR).mkdir(parents=True, exist_ok=True)
(UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
app = FastAPI(title="Mr. TAI Backend", version="0.1.0")

register_request_logging(app)
register_error_handlers(app)

# Use env-driven CORS instead of hard-coded list
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in allowed_origins if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

@app.get("/")
def root():
    # small typo fix: (M} -> (M)
    return {"message": "Hello, I am Mr. TAI!\nMr. TAI stands for (M)ulti-media (R)eal (T)ime AI!"}

def _safe_name(name: str) -> str:
    # keep just the base name, strip directories
    base = os.path.basename(name)
    # optional: strip weird chars
    return "".join(c for c in base if c.isalnum() or c in ("-", "_", ".", " ")).strip() or "file"

@app.post("/upload", status_code=status.HTTP_201_CREATED)
async def upload_file(file: UploadFile = File(...)):
    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    safe = _safe_name(file.filename)
    # make it unique
    unique = f"{uuid.uuid4().hex}_{safe}"
    out_path = UPLOAD_DIR / unique

    with out_path.open("wb") as buf:
        shutil.copyfileobj(file.file, buf)

    return {"message": f"Uploaded '{safe}'", "saved_to": str(out_path), "stored_name": unique}

class ProcessRequest(BaseModel):
    filename: str  # should match "stored_name" returned by /upload

def _simulate_processing(job_id: str):
    import time
    job = jobs[job_id]  # look up the Job instance
    try:
        job.state = JobState.processing
        time.sleep(2)  # pretend work
        job.state = JobState.done
    except Exception as ex:
        job.state = JobState.error
        job.error = str(ex)

@app.post("/process")
def process_file(req: ProcessRequest, background: BackgroundTasks):
    # verify uploaded file exists in UPLOAD_DIR
    src_path = UPLOAD_DIR / req.filename
    if not src_path.exists():
        raise HTTPException(status_code=404, detail="Uploaded file not found. Use 'stored_name' from /upload.")

    # Create a new Job and layout
    job = Job()
    job.ensure_layout()

    # Copy original file into the job's input folder
    dest = job.dir() / "input" / Path(req.filename).name
    try:
        shutil.copy2(src_path, dest)
        job.input_name = dest.name
    except Exception as e:
        job.state = JobState.error
        job.error = f"Failed to stage input: {e}"
        jobs[job.id] = job
        return JSONResponse(job.to_api(), status_code=500)

    # Register job (queued). We'll simulate processing like before.
    jobs[job.id] = job

    def _simulate_processing(job_id: str):
        import time
        j = jobs[job_id]
        try:
            j.state = JobState.processing
            time.sleep(2)  # pretend work
            j.state = JobState.done
        except Exception as ex:
            j.state = JobState.error
            j.error = str(ex)

    background.add_task(_simulate_processing, job.id)

    return {"message": f"Processing started for {req.filename}", "job_id": job.id, "status": job.state.value}

@app.get("/status/{job_id}")
def get_status(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job_id not found")

    # Provide a light artifact list (just files that exist) to help the UI later
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
        "port": port,
        "dataDir": str(DATA_DIR.resolve()),
        "uploadDir": str(UPLOAD_DIR.resolve()),
        "allowedOrigins": [o.strip() for o in allowed_origins if o.strip()],
        "asr": {
            "model": os.getenv("ASR_MODEL_NAME", "tiny"),
            "device": os.getenv("ASR_DEVICE", "cpu"),
            "compute_type": os.getenv("ASR_COMPUTE_TYPE", "int8"),
        },
    }

def _env(name: str, default: str) -> str:
    return os.getenv(name, default)

@lru_cache(maxsize=1)
def get_asr_model() -> WhisperModel:
    model_name = _env("ASR_MODEL_NAME", "tiny")
    device = _env("ASR_DEVICE", "cpu")              # options: "cpu" or "auto"
    compute_type = _env("ASR_COMPUTE_TYPE", "int8") # e.g. "int8", "int8_float16", "float16", "float32"
    return WhisperModel(model_name, device=device, compute_type=compute_type)


# very small stopword list for quick keywords
_STOP = {
    "the","a","an","and","or","but","if","in","on","at","to","of","for","with",
    "is","it","this","that","be","as","are","was","were","by","from","you","i",
    "we","they","he","she","him","her","them","our","your","their"
}

def _probe_duration_seconds(path: str | os.PathLike[str]) -> float:
    path_str = str(path)  # normalize Path/PathLike/str to str
    try:
        out = subprocess.check_output([
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", path_str
        ])
        return float(out.decode().strip())
    except Exception:
        return 0.0

def _is_video(filename: str) -> bool:
    mime, _ = mimetypes.guess_type(filename)
    return (mime or "").startswith("video/")

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

def _top_keywords(text: str, k: int = 5):
    words = re.findall(r"[a-zA-Z']+", text.lower())
    words = [w for w in words if w not in _STOP and len(w) > 2]
    return [w for w, _ in Counter(words).most_common(k)]

MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", 200))

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
    language: Optional[str] = None,          # e.g. "en", "es", "fr" (force)
    vad_filter: bool = True,                  # trims silence; helpful for long files
    word_timestamps: bool = False,            # include per-word timings
):
    if not file or not file.filename:
        return JSONResponse({"error": "No file provided"}, status_code=400)

    # 1) Save original safely with unique name
    safe_base = _safe_name(file.filename)
    unique_name = f"{uuid.uuid4().hex}_{safe_base}"
    in_path = UPLOAD_DIR / unique_name

    try:
        await _write_upload_stream(in_path, file)
    except HTTPException:
        # size exceeded or write error already handled
        raise
    except Exception as e:
        return JSONResponse({"error": f"Failed to save upload: {e}"}, status_code=500)

    media_type = "video" if _is_video(safe_base) else "audio"

    # 2) Ensure we have a WAV for ASR (mono 16k) if video, else keep audio as provided
    try:
        audio_path: Path | str
        if media_type == "video":
            audio_path = _extract_audio_to_wav(str(in_path), UPLOAD_DIR)  # returns Path
        else:
            audio_path = str(in_path)
    except Exception as e:
        # clean original on failure
        try:
            in_path.unlink(missing_ok=True)
        except Exception:
            pass
        return JSONResponse({"error": f"Failed to extract audio: {e}"}, status_code=500)

    # 3) Transcribe (timestamps + optional word timings + translate)
    try:
        asr_model = get_asr_model()
        seg_iter, info = asr_model.transcribe(
            str(audio_path),
            task=task,                 # "transcribe" or "translate"
            language=language,         # None = auto detect
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
        # cleanup both files
        try:
            in_path.unlink(missing_ok=True)
            if isinstance(audio_path, Path):
                audio_path.unlink(missing_ok=True)
        except Exception:
            pass
        return JSONResponse({"error": f"Transcription failed: {e}"}, status_code=500)

    # 4) Summaries
    duration = _probe_duration_seconds(audio_path)
    keywords = _top_keywords(full_text, k=5)

    # 5) Cleanup (best-effort)
    try:
        in_path.unlink(missing_ok=True)
        if isinstance(audio_path, Path):
            audio_path.unlink(missing_ok=True)
    except Exception:
        pass

    # 6) Response (richer AI payload)
    return {
        "message": "Processed successfully",
        "stored_name": unique_name,
        "input_file": safe_base,
        "media_type": media_type,
        "task": task,
        "forced_language": language,
        "detected_language": getattr(info, "language", None),
        "duration_sec": duration,
        "transcript": full_text[:2000],    # clip long payloads
        "segments": seg_list,              # <-- timestamped segments (+words if requested)
        "keywords": keywords,
        "settings": {                      # helpful for debugging
            "vad_filter": vad_filter,
            "beam_size": 5,
            "word_timestamps": word_timestamps,
            "model": os.getenv("ASR_MODEL_NAME", "tiny"),
            "device": os.getenv("ASR_DEVICE", "cpu"),
            "compute_type": os.getenv("ASR_COMPUTE_TYPE", "int8"),
        },
    }

@app.post("/demo/say")
def demo_say(text: str = "Hello from Mr. TAI!"):
    demo_out = DATA_DIR / "demo" / "hello.wav"
    synth_to_wav(text, demo_out)
    return {
        "message": "ok",
        "audio": str(demo_out),
    }
