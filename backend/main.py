from fastapi import FastAPI

app = FastAPI(title="Mr. TAI Backend", version="0.1.0")

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        # keep these if you might use CRA/other ports later:
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"message": "Hello, I am Mr. TAI!\nMr. TAI stands for (M}ulti-media (R)eal (T)ime AI!"}

from fastapi import UploadFile, File
from pathlib import Path
import shutil

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    # basic guard
    if not file or not file.filename:
        return {"error": "No file provided"}

    out_path = UPLOAD_DIR / file.filename
    with out_path.open("wb") as buf:
        shutil.copyfileobj(file.file, buf)

    return {"message": f"Uploaded '{file.filename}'", "saved_to": str(out_path)}

from pydantic import BaseModel
import uuid

# ---- simple in-memory job store (dev only) ----
from typing import Dict, Literal
from datetime import datetime

JobStatus = Literal["queued", "processing", "done", "error"]
JOBS: Dict[str, dict] = {}


class ProcessRequest(BaseModel):
    filename: str

@app.post("/process")
def process_file(req: ProcessRequest):
    # In the future: kick off AI pipeline here
    job_id = str(uuid.uuid4())  # generate fake job id
    # record a fake job for later lookup
    JOBS[job_id] = {
        "job_id": job_id,
        "filename": req.filename,
        "status": "queued",
        "created_at": datetime.utcnow().isoformat() + "Z",
    }
    return {
        "message": f"Processing started for {req.filename}",
        "job_id": job_id,
        "status": "queued"
    }

@app.get("/status/{job_id}")
def get_status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        return {"error": "job_id not found", "job_id": job_id}
    return job


@app.get("/health")
def health_check():
    return {"status": "ok", "version": "0.1.0"}
