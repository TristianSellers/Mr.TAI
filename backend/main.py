from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi import BackgroundTasks
from pydantic import BaseModel
from pathlib import Path
from typing import Dict, Literal
from datetime import datetime
import uuid
import shutil
import os

app = FastAPI(title="Mr. TAI Backend", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
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

JobStatus = Literal["queued", "processing", "done", "error"]
JOBS: Dict[str, dict] = {}

class ProcessRequest(BaseModel):
    filename: str  # should match "stored_name" returned by /upload

def _simulate_processing(job_id: str):
    # NOTE: this runs in-process; fine for dev demos
    import time
    JOBS[job_id]["status"] = "processing"
    time.sleep(2)  # pretend work
    JOBS[job_id]["status"] = "done"
    JOBS[job_id]["completed_at"] = datetime.utcnow().isoformat() + "Z"

@app.post("/process")
def process_file(req: ProcessRequest, background: BackgroundTasks):
    # optional: verify file exists
    if not (UPLOAD_DIR / req.filename).exists():
        raise HTTPException(status_code=404, detail="Uploaded file not found. Use 'stored_name' from /upload.")

    job_id = str(uuid.uuid4())
    JOBS[job_id] = {
        "job_id": job_id,
        "filename": req.filename,
        "status": "queued",
        "created_at": datetime.utcnow().isoformat() + "Z",
    }
    background.add_task(_simulate_processing, job_id)
    return {"message": f"Processing started for {req.filename}", "job_id": job_id, "status": "queued"}

@app.get("/status/{job_id}")
def get_status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job_id not found")
    return job

@app.get("/health")
def health_check():
    return {"status": "ok", "version": "0.1.0"}
