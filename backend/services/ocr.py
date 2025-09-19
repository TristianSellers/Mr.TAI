# backend/services/ocr.py
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Optional, Dict, TYPE_CHECKING, Any
import os, re, logging
from pathlib import Path

# Type-only import so Pylance can see the type without requiring Pillow at runtime
if TYPE_CHECKING:
    # import the class type only for type checkers
    from PIL.Image import Image as _PILImageType
    PILImageLike = _PILImageType
else:
    # at runtime (or if Pillow missing), treat as Any
    PILImageLike = Any  # type: ignore[assignment]

log = logging.getLogger(__name__)

@dataclass
class OCRScoreboard:
    home_team: Optional[str] = None
    away_team: Optional[str] = None
    score:     Optional[str] = None   # e.g., "21-24"
    quarter:   Optional[str] = None   # e.g., "Q4", "OT"
    clock:     Optional[str] = None   # e.g., "0:42"
    # Debug helpers
    ocr_text:  Optional[str] = None
    used_stub: bool = False

    def to_dict(self) -> Dict[str, Optional[str]]:
        return asdict(self)

def _parse_scoreboard_text(text: str) -> OCRScoreboard:
    """
    Heuristics to pull (teams, score, quarter, clock) from OCR text.
    """
    t = (text or "").upper()

    # quarter: Q1..Q4 or OT
    q = re.search(r"\b(Q[1-4]|OT)\b", t)
    quarter = q.group(1) if q else None

    # clock: 12:34 or 0:42
    clk = re.search(r"\b([0-5]?\d:[0-5]\d)\b", t)
    clock = clk.group(1) if clk else None

    # score: ##-## (allow single digit; also accept colon)
    sc = re.search(r"\b(\d{1,2})\s*[-:]\s*(\d{1,2})\b", t)
    score = f"{sc.group(1)}-{sc.group(2)}" if sc else None

    # crude team extraction: tokens near score
    home_team = away_team = None
    if sc:
        i = sc.start()
        left = t[:i].strip()[-25:]
        right = t[sc.end():].strip()[:25]
        left_tokens = re.findall(r"[A-Z]{2,12}", left)
        right_tokens = re.findall(r"[A-Z]{2,12}", right)
        home_team = left_tokens[-1] if left_tokens else None
        away_team = right_tokens[0] if right_tokens else None

    return OCRScoreboard(
        home_team=home_team,
        away_team=away_team,
        score=score,
        quarter=quarter,
        clock=clock,
        ocr_text=text,
        used_stub=False,
    )

def _preprocess(img: PILImageLike) -> PILImageLike:
    """
    Lightweight preprocessing for OCR: grayscale -> autocontrast -> sharpen.
    Imports are local to avoid Pylance 'possibly unbound' when Pillow is absent.
    """
    try:
        from PIL import ImageOps, ImageFilter  # type: ignore
        g = ImageOps.grayscale(img)
        g = ImageOps.autocontrast(g)
        g = g.filter(ImageFilter.SHARPEN)
        return g
    except Exception:
        # If Pillow pieces not available, return original image
        return img


def extract_scoreboard_from_image(image_path: str | os.PathLike[str]) -> OCRScoreboard:
    """
    Prototype OCR: returns OCRScoreboard. If OCR isn’t configured, returns stub.
    Env:
      - TESSERACT_CMD (optional): path to `tesseract` binary if not on PATH
    """
    # Try to import dependencies lazily so editors don't warn
    try:
        from PIL import Image  # type: ignore
    except Exception:
        log.warning("Pillow (PIL) not installed; returning OCR stub.")
        return OCRScoreboard(used_stub=True)

    try:
        import pytesseract  # type: ignore
    except Exception:
        log.warning("pytesseract not installed; returning OCR stub.")
        return OCRScoreboard(used_stub=True)

    # Configure tesseract binary path if provided
    tess_cmd = os.getenv("TESSERACT_CMD")
    if tess_cmd:
        pytesseract.pytesseract.tesseract_cmd = tess_cmd

    p = Path(image_path)
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {p}")

    try:
        with Image.open(p) as im:  # PIL.Image.Image
            im = _preprocess(im)
            # whitelist helpful chars, English by default
            config = "--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789:-"
            text = pytesseract.image_to_string(im, lang="eng", config=config) or ""
    except Exception as e:
        log.exception("OCR failed: %s", e)
        return OCRScoreboard(used_stub=True)

    return _parse_scoreboard_text(text)
