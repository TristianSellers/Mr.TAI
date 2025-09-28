# backend/routers/ocr_api.py
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
    # 1) Trust content-type → ext if known, else fall back to sanitized filename’s ext
    ext = MIME_TO_EXT.get((upload.content_type or "").lower(), "")
    if not ext:
        # Drop any path components and weirdness
        name_only = Path(PurePath(upload.filename or "")).name
        ext = Path(name_only).suffix.lower()
    if not ext:
        raise HTTPException(status_code=400, detail="Missing or unsupported file extension.")
    if ext not in allow:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")
    return ext

def _safe_save_upload(upload: UploadFile, allow_exts: Iterable[str]) -> Path:
    """
    Save upload safely inside TMP_DIR:
    - generates a UUID filename with validated/derived extension
    - exclusive create with 0600 perms
    - verifies resolved path is within TMP_DIR
    """
    ext = _safe_ext(upload, allow_exts)
    uid = uuid.uuid4().hex
    dest = (TMP_DIR / f"{uid}{ext}").resolve()

    # Ensure within TMP_DIR after resolution
    tmp_root = TMP_DIR.resolve()
    if not str(dest).startswith(str(tmp_root) + os.sep):
        raise HTTPException(status_code=400, detail="Invalid upload destination.")

    # Exclusive create (no race) with 0o600 perms
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    # On Windows, os.O_BINARY may be needed, but on *nix this is fine.
    fd = os.open(str(dest), flags, 0o600)
    try:
        with os.fdopen(fd, "wb") as f:
            # Stream copy
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
    try:
        dest = _safe_save_upload(image, IMAGE_EXTS)

        result = extract_scoreboard_from_image(
            str(dest),
            debug_crops=debug,
            viz_boxes_flag=viz,
            dx=dx,
            dy=dy,
        )
        out = result.to_dict()

        width = height = None
        try:
            from PIL import Image  # type: ignore
            with Image.open(dest) as im:
                width, height = im.size
        except Exception:
            pass

        return OCRImageResponse(
            home_team = out.get("home_team"),
            away_team = out.get("away_team"),
            score     = out.get("score"),
            quarter   = out.get("quarter"),
            clock     = out.get("clock"),
            ocr_text  = out.get("ocr_text"),
            used_stub = bool(out.get("used_stub", False)),
            width=width, height=height,
            debug_dir = "data/tmp/ocr_debug" if debug else None,
            boxes_png = "data/tmp/ocr_debug/boxes.png" if viz else None,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"OCR failed: {e}")

@router.post("/video", response_model=OCRVideoResponse)
async def ocr_video(
    video: UploadFile = File(...),
    viz: bool = Form(False),
    dx: int = Form(0),
    dy: int = Form(0),
    t: float = Form(0.10),
):
    try:
        dest = _safe_save_upload(video, VIDEO_EXTS)

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
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"OCR(video) failed: {e}")
