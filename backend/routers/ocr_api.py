# backend/routers/ocr_api.py
from __future__ import annotations
from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from pydantic import BaseModel
from typing import Optional
from pathlib import Path
import shutil
import os
import uuid

from backend.services.ocr import extract_scoreboard_from_image
from backend.services.ocr_from_video import extract_scoreboard_from_video

router = APIRouter(prefix="/ocr", tags=["ocr"])

TMP_DIR = Path(os.getenv("DATA_DIR", "data")) / "tmp" / "uploads"
TMP_DIR.mkdir(parents=True, exist_ok=True)

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
        # Save upload to tmp
        uid = uuid.uuid4().hex
        dest = TMP_DIR / f"{uid}_{image.filename}"
        with dest.open("wb") as f:
            shutil.copyfileobj(image.file, f)

        # Run your existing OCR
        result = extract_scoreboard_from_image(
            str(dest),
            debug_crops=debug,
            viz_boxes_flag=viz,
            dx=dx,
            dy=dy,
        )
        out = result.to_dict()  # mixed types inside (str|None + bool)

        # Image size metadata (optional)
        width = height = None
        try:
            from PIL import Image  # type: ignore
            with Image.open(dest) as im:
                width, height = im.size
        except Exception:
            pass

        # Pylance-friendly explicit mapping (and force bool for used_stub)
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
        uid = uuid.uuid4().hex
        dest = TMP_DIR / f"{uid}_{video.filename}"
        with dest.open("wb") as f:
            shutil.copyfileobj(video.file, f)

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
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"OCR(video) failed: {e}")
