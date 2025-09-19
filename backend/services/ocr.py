# backend/services/ocr.py
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Optional, Dict, TYPE_CHECKING, Any
import os, re, logging
from pathlib import Path

# Type-only alias so editors get hints without requiring Pillow at runtime
if TYPE_CHECKING:
    from PIL.Image import Image as _PILImageType
    PILImageLike = _PILImageType
else:
    PILImageLike = Any  # type: ignore[assignment]

log = logging.getLogger(__name__)

# --------------------------- Model ---------------------------

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

# --------------------------- Parsing ---------------------------

def _parse_scoreboard_text(text: str) -> OCRScoreboard:
    """
    Heuristics to pull (teams, score, quarter, clock) from OCR text.
    Also handles cases where the right score is glued to the clock (e.g., '0:4224')
    and where 'Q4' is misread as ':4'.
    """
    t = (text or "").upper()

    # quarter: Q1..Q4 or OT
    q = re.search(r"\b(Q[1-4]|OT)\b", t)
    quarter = q.group(1) if q else None

    # --- NEW: tolerate ':4' / ';3' misreads for 'Q4' / 'Q3'
    if not quarter:
        m_colon_q = re.search(r"(^|[^A-Z0-9])[:;]\s*([1-4])\b", t)
        if m_colon_q:
            quarter = f"Q{m_colon_q.group(2)}"

    # clock: 12:34 or 0:42
    clk = re.search(r"\b([0-5]?\d:[0-5]\d)\b", t)
    clock = clk.group(1) if clk else None

    # score: ##-## (allow single digit; also accept colon)
    sc = re.search(r"\b(\d{1,2})\s*[-:]\s*(\d{1,2})\b", t)
    score = f"{sc.group(1)}-{sc.group(2)}" if sc else None

    # If no explicit score, infer L...MM:SS...R
    if not score:
        m = re.search(r"\b(\d{1,2})\D{0,10}([0-5]?\d:[0-5]\d)\D{0,10}(\d{1,2})\b", t)
        if m:
            score = f"{m.group(1)}-{m.group(3)}"
            if not clock:
                clock = m.group(2)
        elif clock:
            # handle glued right score like '0:4224'
            i = t.find(clock)
            if i != -1:
                tail = t[i + len(clock): i + len(clock) + 6]
                r = re.search(r"\s*[:\-]?\s*(\d{1,2})\b", tail)
                if r:
                    head = t[max(0, i - 6): i]
                    l = re.search(r"(\d{1,2})\b", head)
                    if l:
                        score = f"{l.group(1)}-{r.group(1)}"

    # crude team extraction: tokens near score
    home_team = away_team = None
    anchor = sc.start() if sc else (t.find(score.replace('-', '')) if score else -1)
    if score and anchor >= 0:
        left = t[:anchor].strip()[-25:]
        right = t[anchor + len(score):].strip()[:25]
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

# --------------------------- OCR helpers ---------------------------

def _ocr_text(img: Any, config: str) -> str:
    """Run Tesseract with given config; returns trimmed text (or empty on failure)."""
    try:
        import pytesseract  # type: ignore
        txt = pytesseract.image_to_string(img, lang="eng", config=config) or ""
        return txt.strip()
    except Exception:
        return ""

def _cv_red_mask(img_pil: PILImageLike) -> Optional[PILImageLike]:
    """
    Emphasize red seven-segment digits using OpenCV (HSV threshold + dilate + invert).
    Returns a high-contrast L image or None if cv2/numpy are unavailable.
    """
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
        from PIL import Image  # type: ignore

        # PIL -> BGR
        np_img = np.array(img_pil.convert("RGB"))[:, :, ::-1]
        hsv = cv2.cvtColor(np_img, cv2.COLOR_BGR2HSV)

        # two red bands in HSV; use uint8 arrays to appease type checkers & cv2
        lower1 = np.array([0, 80, 60], dtype=np.uint8)
        upper1 = np.array([10, 255, 255], dtype=np.uint8)
        lower2 = np.array([170, 80, 60], dtype=np.uint8)
        upper2 = np.array([180, 255, 255], dtype=np.uint8)

        mask1 = cv2.inRange(hsv, lower1, upper1)
        mask2 = cv2.inRange(hsv, lower2, upper2)
        mask = cv2.bitwise_or(mask1, mask2)

        # thicken strokes
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)

        # invert (dark digits on light bg often OCR better)
        inv = cv2.bitwise_not(mask)

        # back to PIL L
        return Image.fromarray(inv)
    except Exception:
        return None

def _preprocess(img: PILImageLike) -> PILImageLike:
    """
    Prefer OpenCV red-mask; else fallback to PIL autocontrast + threshold + sharpen.
    """
    cv_img = _cv_red_mask(img)
    if cv_img is not None:
        return cv_img

    # Fallback: PIL-only pipeline
    try:
        from PIL import ImageOps, ImageFilter  # type: ignore
        # Use red channel if available
        try:
            r = img.split()[0]  # R from RGB
        except Exception:
            r = img
        r = ImageOps.autocontrast(r)

        # gentler threshold so thin segments survive
        th = 110
        lut = ([0] * (th + 1)) + ([255] * (255 - th))
        r = r.point(lut, mode="1").convert("L")

        r = r.filter(ImageFilter.SHARPEN)
        return r
    except Exception:
        return img

def _ocr_digits_line(img: Any, *, allow_colon: bool = False, allow_minus: bool = False) -> str:
    wl = "0123456789"
    if allow_colon:
        wl += ":"
    if allow_minus:
        wl += "-"
    cfg = f"--oem 1 --psm 7 -c tessedit_char_whitelist={wl}"
    return _ocr_text(img, cfg)

def _preprocess_soft(img: PILImageLike) -> PILImageLike:
    """Keep labels like 'Q4': grayscale + autocontrast (no hard threshold)."""
    try:
        from PIL import ImageOps  # type: ignore
        g = ImageOps.grayscale(img)
        g = ImageOps.autocontrast(g)
        return g
    except Exception:
        return img

def _ocr_quarter_line(img: Any) -> str:
    # Allow Q, O, T + digits; single-line
    cfg = "--oem 1 --psm 7 -c tessedit_char_whitelist=QOT01234"
    return _ocr_text(img, cfg)

# --------------------------- Public API ---------------------------

def extract_scoreboard_from_image(image_path: str | os.PathLike[str]) -> OCRScoreboard:
    """
    Prototype OCR: returns OCRScoreboard. If OCR isn’t configured, returns stub.
    Env:
      - TESSERACT_CMD (optional): path to `tesseract` binary if not on PATH
    """
    # Lazy import to keep editors happy if packages are missing
    try:
        from PIL import Image  # type: ignore
        import pytesseract  # type: ignore
    except Exception:
        log.warning("OCR deps missing; returning stub.")
        return OCRScoreboard(used_stub=True)

    # allow overriding tesseract path
    tess_cmd = os.getenv("TESSERACT_CMD")
    if tess_cmd:
        pytesseract.pytesseract.tesseract_cmd = tess_cmd

    p = Path(image_path)
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {p}")

    try:
        with Image.open(p) as im:
            base = _preprocess(im)
            W, H = base.size

            # Soft image keeps white labels like "Q4"
            soft = _preprocess_soft(im)

            # Region under/near the clock (tweak fractions as needed)
            def crop_frac_soft(x, y, w, h):
                return soft.crop((int(x*W), int(y*H), int((x+w)*W), int((y+h)*H)))

            quarter_region = crop_frac_soft(0.40, 0.62, 0.20, 0.20)  # below center clock
            quarter_txt = _ocr_quarter_line(quarter_region)


            # 1) general pass (labels + any digits we can grab)
            general_cfg = "--oem 1 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789:-"
            general_txt = _ocr_text(base, general_cfg)

            # 2) region passes tuned generically:
            #    left = home score, center = clock, right = away score
            def crop_frac(x: float, y: float, w: float, h: float):
                return base.crop((int(x*W), int(y*H), int((x+w)*W), int((y+h)*H)))

            left_region   = crop_frac(0.05, 0.30, 0.25, 0.40)
            center_region = crop_frac(0.31, 0.25, 0.38, 0.32)
            right_region  = crop_frac(0.70, 0.30, 0.25, 0.40)

            clock_txt  = _ocr_digits_line(center_region, allow_colon=True)
            lscore_txt = _ocr_digits_line(left_region, allow_minus=True)
            rscore_txt = _ocr_digits_line(right_region, allow_minus=True)

            merged = "\n".join([
                general_txt or "",
                clock_txt or "",
                lscore_txt or "",
                rscore_txt or "",
                quarter_txt or "",
            ]).strip()

    except Exception as e:
        log.exception("OCR failed: %s", e)
        return OCRScoreboard(used_stub=True)

    # Parse with heuristic
    result = _parse_scoreboard_text(merged)

    # If score missing but we got two numbers from left/right, stitch them
    if not result.score:
        lh = re.search(r"\b(\d{1,2})\b", lscore_txt or "")
        rh = re.search(r"\b(\d{1,2})\b", rscore_txt or "")
        if lh and rh:
            result.score = f"{lh.group(1)}-{rh.group(1)}"

    # If quarter missing, try from general text
    if not result.quarter:
        q = re.search(r"\b(Q[1-4]|OT)\b", (general_txt or "").upper())
        if q:
            result.quarter = q.group(1)

    # If clock missing, take from focused pass
    if not result.clock and clock_txt:
        c = re.search(r"\b([0-5]?\d:[0-5]\d)\b", clock_txt)
        if c:
            result.clock = c.group(1)

    result.ocr_text = merged
    return result
