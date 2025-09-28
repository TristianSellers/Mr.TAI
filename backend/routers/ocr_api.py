from __future__ import annotations
from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from pydantic import BaseModel
from typing import Optional, Iterable
from pathlib import Path, PurePath
import shutil
import os
import uuid

from backend.services.ocr import extract_scoreboard_from_image
from backend.services.ocr_from_video import extract_scoreboard_from_video

router = APIRouter(prefix="/ocr", tags=["ocr"])

TMP_DIR = Path(os.getenv("DATA_DIR", "data")) / "tmp" / "uploads"
TMP_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi"}
MIME_TO_EXT = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/webp": ".webp",
    "image/bmp": ".bmp",
    "image/tiff": ".tiff",
    "video/mp4": ".mp4",
    "video/quicktime": ".mov",
    "video/x-matroska": ".mkv",
    "video/x-msvideo": ".avi",
}

def _safe_ext(upload: UploadFile, allow: Iterable[str]) -> str:
    ext = MIME_TO_EXT.get((upload.content_type or "").lower(), "")
    if not ext:
        # sanitize filename
        name_only = Path(PurePath(upload.filename or "")).name
        ext = Path(name_only).suffix.lower()
    if not ext or ext not in allow:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext or 'unknown'}")
    return ext

def _safe_save_upload(upload: UploadFile, allow_exts: Iterable[str]) -> Path:
    """
    Save upload safely inside TMP_DIR:
    - UUID filename with validated extension
    - exclusive create with 0600 perms
    - confined to TMP_DIR
    """
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

# ---------- Schemas ----------
class OCRImageResponse(BaseModel):
    home_team: Optional[str]
    away_team: Optional[str]
    score: Optional[str]
    quarter: Optional[str]
    clock: Optional[str]
    ocr_text: Optional[str]
    used_stub: bool
    width: Optional[int] = None
    height: Optional[int] = None
    debug_dir: Optional[str] = None
    boxes_png: Optional[str] = None

class OCRVideoResponse(BaseModel):
    home_team: Optional[str]
    away_team: Optional[str]
    score: Optional[str]
    quarter: Optional[str]
    clock: Optional[str]
    ocr_text: Optional[str]
    used_stub: bool
    sampled_from_s: float

# ---------- Endpoints ----------
@router.post("/image", response_model=OCRImageResponse)
async def ocr_image(
    image: UploadFile = File(...),
    debug: bool = Form(False),
    viz: bool = Form(False),
    dx: int = Form(0),
    dy: int = Form(0),
):
    dest = _safe_save_upload(image, IMAGE_EXTS)
    keep_uploads = os.getenv("KEEP_UPLOADS", "0") == "1"
    try:
        # width/height before unlink
        width = height = None
        try:
            from PIL import Image
            with Image.open(dest) as im:
                width, height = im.size
        except Exception:
            pass

        result = extract_scoreboard_from_image(
            str(dest),
            debug_crops=debug,
            viz_boxes_flag=viz,
            dx=dx,
            dy=dy,
        )
        out = result.to_dict()

        return OCRImageResponse(
            home_team=out.get("home_team"),
            away_team=out.get("away_team"),
            score=out.get("score"),
            quarter=out.get("quarter"),
            clock=out.get("clock"),
            ocr_text=out.get("ocr_text"),
            used_stub=bool(out.get("used_stub", False)),
            width=width,
            height=height,
            debug_dir="data/tmp/ocr_debug" if debug else None,
            boxes_png="data/tmp/ocr_debug/boxes.png" if viz else None,
        )
    finally:
        if not keep_uploads:
            try:
                dest.unlink(missing_ok=True)
            except Exception:
                pass

@router.post("/video", response_model=OCRVideoResponse)
async def ocr_video(
    video: UploadFile = File(...),
    viz: bool = Form(False),
    dx: int = Form(0),
    dy: int = Form(0),
    t: float = Form(0.10),
):
    dest = _safe_save_upload(video, VIDEO_EXTS)
    keep_uploads = os.getenv("KEEP_UPLOADS", "0") == "1"
    try:
        data = extract_scoreboard_from_video(
            str(dest),
            viz=viz,
            dx=dx,
            dy=dy,
            t=t,
        )
        return OCRVideoResponse(
            home_team=data.get("home_team"),
            away_team=data.get("away_team"),
            score=data.get("score"),
            quarter=data.get("quarter"),
            clock=data.get("clock"),
            ocr_text=data.get("ocr_text"),
            used_stub=bool(data.get("used_stub", False)),
            sampled_from_s=float(t),
        )
    finally:
        if not keep_uploads:
            try:
                dest.unlink(missing_ok=True)
            except Exception:
                pass
