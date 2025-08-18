from fastapi import FastAPI

app = FastAPI(title="Mr. TAI Backend", version="0.1.0")

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
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

@app.get("/health")
def health_check():
    return {"status": "ok", "version": "0.1.0"}
