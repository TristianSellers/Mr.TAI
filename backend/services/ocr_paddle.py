# backend/services/ocr_paddle.py
from __future__ import annotations

import json
import logging
import re
import shlex
import subprocess
import time
import uuid
from rapidfuzz import fuzz
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageOps, ImageDraw
import pytesseract  # <-- added back for Tesseract fallback

try:
    import cv2 as _cv2  # optional
except Exception:
    _cv2 = None
cv2 = _cv2

# EasyOCR (deep-learning OCR)
try:
    import easyocr as _easyocr  # type: ignore[import]
except Exception:
    _easyocr = None

_easyocr_reader = None  # lazy-initialized EasyOCR reader

log = logging.getLogger(__name__)
if not log.handlers:
    logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# Geometry / layout constants (multi-skin: MNP / TNP / SNP / DEFAULT)
# ---------------------------------------------------------------------------

REF_FRAME_W, REF_FRAME_H = 2047, 1155

SCOREBAR_BOX_DEFAULT: Tuple[int, int, int, int] = (0, 1080, 2047, 1155)

SCOREBAR_BOX_PRESETS: Dict[str, Tuple[int, int, int, int]] = {
    "MNP": (0, 1080, 2000, 1150),
    "TNP": (630, 1020, 1420, 1145),
    "SNP": (590, 1040, 1510, 1140),
    "DEFAULT": SCOREBAR_BOX_DEFAULT,
}

CANON_SIZE: Dict[str, Tuple[int, int]] = {
    "MNP": (2000, 72),
    "TNP": (800, 128),
    "SNP": (920, 100),
    "DEFAULT": (1445, 72),
}

ROI_PRESETS_CANON: Dict[str, Dict[str, Tuple[int, int, int, int]]] = {
    "MNP": {
        "away_team":  (550,  15,  626,  60),
        "away_score": (650,   0,  775,  70),
        "home_team":  (975,  15, 1050,  60),
        "home_score": (825,   0,  950,  70),
        "quarter":    (1350, 15, 1410,  60),
        "clock":      (1410, 15, 1490,  60),
        # NEW: split down / distance
        "down":       (1675, 15, 1740,  60),
        "distance":   (1760, 15, 1810,  60),
        "yardline":   (1900, 15, 2000,  60),
        "gameclock":  (1500, 15, 1600,  60),
    },
    "TNP": {
        "away_team":  (210,  25,  275,  75),
        "away_score": (275,   5,  365,  75),
        "home_team":  (525,  25,  600,  75),
        "home_score": (440,   5,  525,  75),
        "quarter":    (300,  75,  350, 120),
        "clock":      (350,  75,  450, 120),
        # NEW: split down / distance
        "down":       (175,  75,  233, 120),
        "distance":   (247,  75,  300, 120),
        "yardline":   (525,  75,  600, 120),
        "gameclock":  (450,  75,  500, 120),
    },
    "SNP": {
        "away_team":  (130,  25,  205,  85),
        "away_score": (210,  25,  300,  85),
        "home_team":  (665,  25,  735,  85),
        "home_score": (570,  25,  660,  85),
        "quarter":    (340,  55,  375,  90),
        "clock":      (375,  55,  450,  90),
        # NEW: split down / distance
        "down":       (375,  20,  433,  50),
        "distance":   (447,  20,  490,  50),
        "yardline":   (485,  55,  525,  90),
        "gameclock":  (450,  55,  485,  90),
    },
    "DEFAULT": {
        "away_team":  (375,   10,  465,  45),
        "home_team":  (670,   10,  755,  45),
        "away_score": (490,   10,  560,  55),
        "home_score": (575,   10,  645,  55),
        "quarter":    (960,   10, 1010,  55),
        "clock":      (1010,  10, 1090,  55),
        # NEW: split down / distance
        "down":       (1175,  10, 1240,  55),
        "distance":   (1260,  10, 1325,  55),
        "yardline":   (1350,  10, 1425,  55),
        "gameclock":  (1100,  10, 1150,  55),
    },
}


# Top-right label region in FULL FRAME (used for skin detection)
TOPRIGHT_BOX_REF: Tuple[int, int, int, int] = (1850, 30, 1950, 90)

# ---------------------------------------------------------------------------
# ffmpeg frame grab
# ---------------------------------------------------------------------------


def _run_ffmpeg_grab(src: Path, ts: float, out_png: Path) -> bool:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    cmd = (
        f"ffmpeg -y -v error -ss {ts:.3f} "
        f"-i {shlex.quote(str(src))} -frames:v 1 {shlex.quote(str(out_png))}"
    )
    log.info("[_run_ffmpeg_grab] %s", cmd)
    subprocess.run(cmd, shell=True, check=False)
    return out_png.exists()


def _grab_frame(video_path: str, ts: float, out_png: Path) -> Path:
    src = Path(video_path)
    if not src.exists():
        raise FileNotFoundError(f"Video not found: {src}")
    log.info("[_grab_frame] Grabbing frame at t=%.3fs from %s", ts, src)
    ok = _run_ffmpeg_grab(src, ts, out_png)
    if not ok:
        raise RuntimeError(f"Could not grab frame at t={ts:.3f}s")
    log.info("[_grab_frame] Saved frame to %s", out_png)
    return out_png


# ---------------------------------------------------------------------------
# Basic image utilities
# ---------------------------------------------------------------------------

def _ensure_np_rgb(img: Any) -> np.ndarray:
    if isinstance(img, Image.Image):
        arr = np.array(img.convert("RGB"))
    else:
        arr = img
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    return arr


def _scale_fullframe_box_to_actual(
    full_box_ref: Tuple[int, int, int, int],
    actual_w: int,
    actual_h: int,
) -> Tuple[int, int, int, int]:
    lx, ty, rx, by = full_box_ref
    sx = actual_w / float(REF_FRAME_W)
    sy = actual_h / float(REF_FRAME_H)
    X1 = int(round(lx * sx))
    Y1 = int(round(ty * sy))
    X2 = int(round(rx * sx))
    Y2 = int(round(by * sy))
    return X1, Y1, X2, Y2


def _clamp_box_exclusive(
    box: Tuple[int, int, int, int],
    W: int,
    H: int,
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    x1 = max(0, min(x1, W))
    y1 = max(0, min(y1, H))
    x2 = max(0, min(x2, W))
    y2 = max(0, min(y2, H))
    if x2 <= x1:
        x2 = min(W, x1 + 1)
    if y2 <= y1:
        y2 = min(H, y1 + 1)
    return x1, y1, x2, y2


def crop_scorebar(full_frame: Any, skin: str) -> np.ndarray:
    t0 = time.perf_counter()
    skin = (skin or "DEFAULT").upper()
    if skin not in SCOREBAR_BOX_PRESETS:
        skin = "DEFAULT"

    img = _ensure_np_rgb(full_frame)
    H, W, _ = img.shape
    box_ref = SCOREBAR_BOX_PRESETS[skin]
    x1, y1, x2, y2 = _scale_fullframe_box_to_actual(box_ref, W, H)
    x1, y1, x2, y2 = _clamp_box_exclusive((x1, y1, x2, y2), W, H)
    crop = img[y1:y2, x1:x2, :].copy()
    dt_ms = (time.perf_counter() - t0) * 1000.0
    log.info(
        "[crop_scorebar][%s] box=%s -> crop %dx%d in %.3f ms",
        skin,
        (x1, y1, x2, y2),
        crop.shape[1],
        crop.shape[0],
        dt_ms,
    )
    return crop


def to_canonical(scorebar: Any, skin: str) -> np.ndarray:
    t0 = time.perf_counter()
    skin = (skin or "DEFAULT").upper()
    if skin not in CANON_SIZE:
        skin = "DEFAULT"
    target_w, target_h = CANON_SIZE[skin]
    sb = _ensure_np_rgb(scorebar)
    if cv2 is not None:
        canon = cv2.resize(sb, (target_w, target_h), interpolation=cv2.INTER_AREA)
    else:
        pil = Image.fromarray(sb)
        canon = np.array(
            pil.resize((target_w, target_h), resample=Image.Resampling.BICUBIC)
        )
    dt_ms = (time.perf_counter() - t0) * 1000.0
    log.info(
        "[to_canonical][%s] -> %dx%d in %.3f ms",
        skin,
        target_w,
        target_h,
        dt_ms,
    )
    return canon


def extract_rois(
    bar_canon: Any,
    skin: str,
    pad: int = 2,
) -> Dict[str, np.ndarray]:
    t0 = time.perf_counter()
    skin = (skin or "DEFAULT").upper()
    roi_map = ROI_PRESETS_CANON.get(skin) or ROI_PRESETS_CANON["DEFAULT"]

    img = _ensure_np_rgb(bar_canon)
    H, W, _ = img.shape

    patches: Dict[str, np.ndarray] = {}
    for name, (x1, y1, x2, y2) in roi_map.items():
        x1p = max(0, x1 - pad)
        y1p = max(0, y1 - pad)
        x2p = min(W, x2 + pad)
        y2p = min(H, y2 + pad)
        patch = img[y1p:y2p, x1p:x2p, :].copy()
        patches[name] = patch

    dt_ms = (time.perf_counter() - t0) * 1000.0
    log.info(
        "[extract_rois][%s] %d patches in %.3f ms",
        skin,
        len(patches),
        dt_ms,
    )
    return patches


# ---------------------------------------------------------------------------
# Tesseract helpers (for clock fallback)
# ---------------------------------------------------------------------------

def _tess_cfg(whitelist: str, psm: int = 7) -> str:
    """
    Build a Tesseract config string with a character whitelist.
    PSM 7: treat as a single text line.
    """
    return f'--psm {psm} -c tessedit_char_whitelist={whitelist}'


def _resize_for_ocr(pil_img: Image.Image, scale: int = 3) -> Image.Image:
    """
    Upscale small crops so Tesseract has more pixels to work with.
    """
    w, h = pil_img.size
    if w * scale < 300 and h * scale < 150:
        return pil_img.resize(
            (max(1, w * scale), max(1, h * scale)),
            resample=Image.Resampling.BICUBIC,
        )
    return pil_img


def _ocr_raw(
    pil_img: Image.Image,
    whitelist: Optional[str] = None,
    psm: int = 7,
) -> str:
    """
    Run Tesseract on a PIL image, optional whitelist, return stripped text.
    """
    img = _resize_for_ocr(pil_img)
    config = ""
    if whitelist is not None:
        config = _tess_cfg(whitelist, psm=psm)
    text = pytesseract.image_to_string(img, config=config)
    return (text or "").strip()


# ---------------------------------------------------------------------------
# EasyOCR helpers
# ---------------------------------------------------------------------------


def _get_easyocr_reader():
    """
    Lazy-initialize a global EasyOCR reader.
    Returns None if EasyOCR is not available.
    """
    global _easyocr_reader
    if _easyocr is None:
        log.warning("[easyocr] EasyOCR library not available")
        return None
    if _easyocr_reader is None:
        log.info("[easyocr] Initializing EasyOCR reader (en, gpu=False)")
        _easyocr_reader = _easyocr.Reader(["en"], gpu=False)
    return _easyocr_reader


def _easy_ocr_single(
    pil_img: Image.Image,
    *,
    allow_chars: Optional[str] = None,
    detail: bool = False,
    textcase: str = "upper",
) -> str:
    """
    Run EasyOCR on a PIL image and return a single combined text string.

    - allow_chars: if given, restricts allowed characters.
    - detail: if True, return the full EasyOCR results list as a string.
    - textcase: "upper", "lower", or "none".
    """
    reader = _get_easyocr_reader()
    if reader is None:
        return ""

    np_img = np.array(pil_img.convert("RGB"))

    kwargs: Dict[str, Any] = {"detail": 0, "paragraph": False}
    if allow_chars is not None:
        kwargs["allowlist"] = allow_chars

    result = reader.readtext(np_img, **kwargs)  # with detail=0 -> list[str]

    if detail:
        # If caller wants raw detail, just stringify it
        return str(result)

    pieces = [s for s in result if isinstance(s, str)]
    txt = "".join(pieces).strip()

    if textcase == "upper":
        txt = txt.upper()
    elif textcase == "lower":
        txt = txt.lower()

    return txt


def _easyocr_text(
    pil_img: Image.Image,
    allowlist: Optional[str] = None,
) -> str:
    """
    Run EasyOCR on a PIL image and return combined text.

    - Upscales small images to help recognition
    - Uses allowlist when provided
    """
    reader = _get_easyocr_reader()
    if reader is None:
        return ""

    img = pil_img.convert("RGB")

    # Upscale small patches (scoreboard bits are tiny)
    if img.width < 200 or img.height < 80:
        scale = 4
        img = img.resize(
            (img.width * scale, img.height * scale),
            resample=Image.Resampling.BICUBIC,
        )

    arr = np.array(img)
    kwargs: Dict[str, Any] = {"detail": 0, "paragraph": False}
    if allowlist is not None:
        kwargs["allowlist"] = allowlist

    results = reader.readtext(arr, **kwargs)
    if not results:
        return ""

    texts = [r.strip() for r in results if isinstance(r, str) and r.strip()]
    return " ".join(texts)


def _ocr_team(pil_img: Optional[Image.Image]) -> Optional[str]:
    """OCR a team abbreviation (letters only) using EasyOCR."""
    if pil_img is None:
        return None
    txt = _easyocr_text(pil_img, allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    log.debug("[_ocr_team][easyocr] raw='%s'", txt)
    s = re.sub(r"[^A-Z]", "", txt.upper())
    if not s:
        return None
    return s[:4]


def _ocr_digits_int(pil_img: Optional[Image.Image]) -> Optional[int]:
    """
    OCR a small integer (score, down, distance, etc.) using
    EasyOCR as primary and Tesseract as a sanity check.

    This also handles the common 1 ↔ 7 confusion we see in
    your scoreboard fonts.
    """
    if pil_img is None:
        return None

    # ---- EasyOCR pass ----
    txt_easy = _easyocr_text(pil_img, allowlist="0123456789")
    txt_easy = (txt_easy or "").strip()
    if log.isEnabledFor(logging.DEBUG):
        log.debug("[_ocr_digits_int][easyocr] raw='%s'", txt_easy)

    m_easy = re.search(r"\d{1,2}", txt_easy)
    val_easy: Optional[int] = None
    if m_easy:
        try:
            val_easy = int(m_easy.group(0))
        except Exception:
            val_easy = None

    # ---- Tesseract pass ----
    txt_tess = _ocr_raw(pil_img, whitelist="0123456789", psm=7)
    if log.isEnabledFor(logging.DEBUG):
        log.debug("[_ocr_digits_int][tesseract] raw='%s'", txt_tess)

    m_tess = re.search(r"\d{1,2}", txt_tess or "")
    val_tess: Optional[int] = None
    if m_tess:
        try:
            val_tess = int(m_tess.group(0))
        except Exception:
            val_tess = None

    # ---- Combine results ----
    if val_easy is None and val_tess is None:
        return None
    if val_easy is not None and val_tess is None:
        return val_easy
    if val_tess is not None and val_easy is None:
        return val_tess

    # both not None here
    assert val_easy is not None and val_tess is not None

    # If they agree, we're happy
    if val_easy == val_tess:
        return val_easy

    # Special-case the 1 vs 7 confusion:
    # we've seen EasyOCR call a 7 as 1, while Tesseract gets 7.
    if {val_easy, val_tess} == {1, 7}:
        return 7

    # Otherwise, trust EasyOCR as the primary
    return val_easy

def _ocr_clock(pil_img: Optional[Image.Image]) -> Optional[str]:
    """
    OCR a game clock like MM:SS.
    Uses EasyOCR + Tesseract fallback with robust normalization.
    """
    if pil_img is None:
        return None

    # EasyOCR attempt
    raw_easy = _easy_ocr_single(
        pil_img,
        allow_chars="0123456789:",
        detail=False,
        textcase="upper",
    )
    if log.isEnabledFor(logging.DEBUG):
        log.debug("[_ocr_clock][easyocr] raw='%s'", raw_easy)

    clk = _norm_clock_flexible(raw_easy)
    if clk:
        return clk

    # Fallback to Tesseract
    raw_tess = _ocr_raw(pil_img, whitelist="0123456789:", psm=7)
    if log.isEnabledFor(logging.DEBUG):
        log.debug("[_ocr_clock][tesseract] raw='%s'", raw_tess)

    clk2 = _norm_clock_flexible(raw_tess)
    return clk2



def _ocr_generic_text(pil_img: Optional[Image.Image]) -> str:
    """OCR generic text using EasyOCR (no whitelist)."""
    if pil_img is None:
        return ""
    txt = _easyocr_text(pil_img, allowlist=None)
    log.debug("[_ocr_generic_text][easyocr] raw='%s'", txt)
    return txt


# ---------------------------------------------------------------------------
# Skin auto-detection from top-right label (EasyOCR)
# ---------------------------------------------------------------------------


def _detect_profile_from_topright(
    full_pil: Image.Image,
    *,
    viz_dir: Optional[Path] = None,
) -> str:
    """
    Crop the top-right label region and OCR it with EasyOCR to decide
    which skin is active (MNP / TNP / SNP). Falls back to DEFAULT.
    """
    img = _ensure_np_rgb(full_pil)
    H, W, _ = img.shape

    # Scale reference box to actual frame
    x1, y1, x2, y2 = _scale_fullframe_box_to_actual(TOPRIGHT_BOX_REF, W, H)

    # Expand the box a bit to be safe
    w_box = x2 - x1
    h_box = y2 - y1
    pad_x = int(w_box * 0.7) or 10
    pad_y = int(h_box * 0.7) or 10

    x1 -= pad_x
    x2 += pad_x
    y1 -= pad_y
    y2 += pad_y

    x1, y1, x2, y2 = _clamp_box_exclusive((x1, y1, x2, y2), W, H)
    patch_np = img[y1:y2, x1:x2, :]
    patch_pil = Image.fromarray(patch_np)

    # Light preprocessing to help EasyOCR
    proc = ImageOps.autocontrast(ImageOps.grayscale(patch_pil)).convert("RGB")

    if viz_dir is not None:
        viz_dir.mkdir(parents=True, exist_ok=True)
        patch_pil.save(viz_dir / "topright_label_raw.png")
        proc.save(viz_dir / "topright_label_proc.png")

    txt = _easyocr_text(
        proc,
        allowlist="MNPTSABCDEFGHIJKLOQRUVWXYZ",
    )
    txt_u = txt.strip().upper()
    letters = re.sub(r"[^A-Z]", "", txt_u)
    log.info(
        "[detect_profile][easyocr] raw='%s', letters='%s', box=%s",
        txt_u,
        letters,
        (x1, y1, x2, y2),
    )

    if not letters:
        log.info("[detect_profile] no OCR letters, using DEFAULT")
        return "DEFAULT"

    # First, try exact substring
    for label in ("MNP", "TNP", "SNP"):
        if label in letters:
            log.info("[detect_profile] exact match -> %s", label)
            return label

    # Fallback: fuzzy 3-char matching (>=2/3 chars)
    candidates = ["MNP", "TNP", "SNP"]
    best_label = None
    best_score = -1

    for i in range(0, max(0, len(letters) - 2)):
        seg = letters[i : i + 3]
        if len(seg) < 3:
            continue
        for cand in candidates:
            score = sum(seg[j] == cand[j] for j in range(3))
            if score > best_score:
                best_score = score
                best_label = cand

    if best_label is not None and best_score >= 2:
        log.info(
            "[detect_profile] fuzzy match -> %s (score=%d, letters='%s')",
            best_label,
            best_score,
            letters,
        )
        return best_label

    log.info(
        "[detect_profile] no confident match, using DEFAULT (letters='%s')", letters
    )
    return "DEFAULT"


# ---------------------------------------------------------------------------
# Debug drawing
# ---------------------------------------------------------------------------


def _draw_roi_debug(bar_canon: np.ndarray, skin: str) -> Image.Image:
    skin = (skin or "DEFAULT").upper()
    roi_map = ROI_PRESETS_CANON.get(skin) or ROI_PRESETS_CANON["DEFAULT"]

    img = bar_canon.copy()

    if cv2 is not None:
        for name, (x1, y1, x2, y2) in roi_map.items():
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
            cv2.putText(
                img,
                name,
                (x1, max(10, y1 - 2)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 255, 0),
                1,
                lineType=cv2.LINE_AA,
            )
        return Image.fromarray(img)

    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)
    for name, (x1, y1, x2, y2) in roi_map.items():
        draw.rectangle([x1, y1, x2, y2], outline="green", width=1)
        draw.text((x1, max(0, y1 - 8)), name, fill="green")
    return pil_img


# ---------------------------------------------------------------------------
# Normalization and parsing helpers
# ---------------------------------------------------------------------------

_CLOCK_RX = re.compile(r"^\s*(\d{1,2}):([0-5]\d)\s*$", re.I)
_DOWN_DIST_RX = re.compile(
    r"\b(1ST|2ND|3RD|4TH|\d)\s*(?:&|AND)\s*(GOAL|\d{1,2})\b",
    re.I,
)
_YARDLINE_RX = re.compile(
    r"\b([A-Z]{2,4})?\s*(50|[0-4]?\d)\b",
    re.I,
)

_QUARTER_TOKENS = ["1ST", "2ND", "3RD", "4TH", "Q1", "Q2", "Q3", "Q4", "OT"]


def _norm_team(s: str) -> Optional[str]:
    s = (s or "").strip()
    if not s:
        return None
    out = re.sub(r"[^A-Z]", "", s.upper())
    return out or None


def _norm_clock(s: str) -> Optional[str]:
    if not s:
        return None
    x = s.strip().replace(";", ":").replace(" ", "")
    m = _CLOCK_RX.match(x)
    if m:
        return f"{int(m.group(1))}:{m.group(2)}"
    return None

def _norm_clock_flexible(s: str) -> Optional[str]:
    """
    More flexible normalization for OCR'd clocks:
    - Enforces football constraints: minutes 0–15, seconds 0–59.
    - Handles:
      * '5:38'  -> '5:38'
      * '5338'  -> '5:38'
      * '248'   -> '2:48'
      * '40'    -> '0:40'
    """
    if not s:
        return None

    # Clean & normalize
    s = (
        s.strip()
         .replace(" ", "")
         .replace(";", ":")
         .replace(".", ":")
         .replace("O", "0")
         .replace("o", "0")
    )

    # If it already has a colon, treat it as MM:SS
    m = re.fullmatch(r"(\d{1,2}):(\d{2})", s)
    if m:
        mins = int(m.group(1))
        secs = int(m.group(2))
        if 0 <= mins <= 15 and 0 <= secs <= 59:
            return f"{mins}:{secs:02d}"
        return None

    # From here on, only digits
    digits = re.sub(r"\D", "", s)
    if not digits:
        return None

    # 3 digits → M:SS (e.g. '248' -> 2:48)
    if len(digits) == 3:
        mins = int(digits[0])
        secs = int(digits[1:])
        if 0 <= mins <= 15 and 0 <= secs <= 59:
            return f"{mins}:{secs:02d}"

    # 4 digits:
    #  First try MM:SS (e.g. '1234' -> 12:34, but reject 53:38).
    #  If that fails, try M:SS using the first digit as minutes and last two as seconds (e.g. '5338' -> 5:38).
    if len(digits) == 4:
        mins2 = int(digits[:2])
        secs2 = int(digits[2:])
        if 0 <= mins2 <= 15 and 0 <= secs2 <= 59:
            return f"{mins2}:{secs2:02d}"

        mins1 = int(digits[0])
        secs1 = int(digits[2:])
        if 0 <= mins1 <= 15 and 0 <= secs1 <= 59:
            return f"{mins1}:{secs1:02d}"

    # 2 digits → assume :SS, but cap to normal seconds
    if len(digits) == 2:
        secs = int(digits)
        if 0 <= secs <= 59:
            return f"0:{secs:02d}"

    return None

def _score_quarter_candidate(raw: str) -> float:
    if not raw:
        return -1.0
    s = raw.strip().upper()
    s_nospace = re.sub(r"\s+", "", s)
    score = 0.0
    if "OT" in s_nospace or s_nospace == "0T":
        score += 4.0
    if "Q" in s_nospace:
        score += 2.5
    if re.search(r"\b(?:1ST|2ND|3RD|4TH)\b", s):
        score += 2.0
    if re.search(r"\b[1-4]\b", s) or re.search(r"Q[ ]?[1-4]", s):
        score += 2.0
    score -= 0.05 * len(s_nospace)
    return score


def _normalize_quarter(raw: str) -> Optional[str]:
    """
    Make quarter detection more robust to EasyOCR quirks:
    ZND -> 2ND, Ath -> 4TH, etc. Uses fuzzy matching as a fallback.
    """
    if not raw:
        return None

    s = raw.strip().upper()
    s = s.replace("%", "").replace("?", "")

    # First, try the old direct logic
    s_no_space = s.replace(" ", "")
    if s_no_space in {"Q1", "Q2", "Q3", "Q4", "OT"}:
        return s_no_space
    if s_no_space in {"1ST", "2ND", "3RD", "4TH"}:
        return {"1ST": "Q1", "2ND": "Q2", "3RD": "Q3", "4TH": "Q4"}[s_no_space]

    m = re.search(r"Q?([1-4])", s_no_space)
    if m:
        return f"Q{m.group(1)}"
    if s_no_space in {"4T", "4TH.", "Q4.", "Q-4"}:
        return "Q4"
    if s_no_space in {"OT.", "0T"}:
        return "OT"

    # Heuristic fixes for very common OCR mistakes
    if s_no_space in {"ZND", "2NO"}:
        return "Q2"
    if s_no_space in {"ATH", "4H", "A7H"}:
        return "Q4"

    # Fuzzy match against quarter tokens
    best_token = None
    best_score = 0
    for tok in _QUARTER_TOKENS:
        score = fuzz.ratio(s_no_space, tok)
        if score > best_score:
            best_score = score
            best_token = tok

    # Require some minimum confidence
    if best_token and best_score >= 60:
        if best_token in {"Q1", "Q2", "Q3", "Q4", "OT"}:
            return best_token
        if best_token in {"1ST", "2ND", "3RD", "4TH"}:
            return {"1ST": "Q1", "2ND": "Q2", "3RD": "Q3", "4TH": "Q4"}[best_token]

    return None


def _parse_down_distance(s: str) -> tuple[Optional[int], Optional[int], Optional[str]]:
    if not s:
        return None, None, None

    # Clean up to alphanumerics, &, and space
    t = re.sub(r"[^A-Za-z0-9& ]", "", s.upper()).strip()

    # Fix common EasyOCR confusions for down text before parsing.
    replacements = {
        "15T": "1ST",
        "75T": "1ST",
        "IST": "1ST",
        "ZND": "2ND",
        "ZUD": "2ND",
        "2ST": "2ND",
    }
    for bad, good in replacements.items():
        if bad in t:
            t = t.replace(bad, good)

    down_map = {"1ST": 1, "2ND": 2, "3RD": 3, "4TH": 4}

    # --- SPECIAL CASE: INCHES (3RD & INCHES, etc.) ---
    # We often see only the tail of "INCHES" in the distance ROI: CHES, HES, NCHES...
    if any(k in t for k in ("INCH", "NCHES", "CHES", "HES")):
        # We always build s as "down_raw & dist_raw", so split on '&'
        left, _, right = t.partition("&")
        left = left.strip()

        # Parse down from the left part (e.g., "3RDGL" -> "3RD")
        down = None
        m_down = re.search(r"(1ST|2ND|3RD|4TH|[1-4])", left)
        if m_down:
            token = m_down.group(1)
            if token in down_map:
                down = down_map[token]
            elif token.isdigit():
                down = int(token)

        # Represent inches as distance = 1, distance_text = "INCHES"
        return down, 1, "INCHES"

    # --- Normal "X & Y" / "X & GOAL" handling ---
    m = _DOWN_DIST_RX.search(t)
    if not m:
        # Fallback: numeric down & distance ("2 & 9", etc.)
        m2 = re.search(r"\b([1234])\s*(?:&|AND)\s*(GOAL|\d{1,2})\b", t)
        if not m2:
            return None, None, None
        g1, g2 = m2.group(1), m2.group(2)
        down = int(g1)
        if g2 == "GOAL":
            return down, None, "GOAL"
        return down, int(g2), g2

    g1, g2 = m.group(1), m.group(2)
    down = down_map.get(g1, None) if not g1.isdigit() else int(g1)
    if g2 == "GOAL":
        return down, None, "GOAL"
    return down, int(g2), g2

def _parse_yardline(s: str) -> tuple[Optional[str], Optional[int]]:
    """
    Parse yardline strings, handling ugly OCR like '439' (→ 39).
    """
    if not s:
        return None, None

    t = re.sub(r"[^A-Za-z0-9 ]", "", s.upper()).strip()

    # --- Fix common OCR confusions ---
    # 7 ↔ 1, 5 ↔ S, 0 ↔ O
    t = (
        t.replace("75T", "1ST")
         .replace("7ST", "1ST")
         .replace("SND", "2ND")
         .replace("5ND", "2ND")
         .replace("3RO", "3RD")
         .replace("30D", "3RD")
         .replace("40H", "4TH")
         .replace("A7H", "4TH")
    )

    # First try the classic pattern (TEAM 39, CHI 39, etc.)
    m = _YARDLINE_RX.search(t)
    if m:
        side = (m.group(1) or "").strip() or None
        y = int(m.group(2))
        y = max(0, min(50, y))
        return side, y

    # Fallback: search all 1–2 digit chunks and pick the last
    nums = re.findall(r"\d{1,2}", t)
    if nums:
        for candidate in reversed(nums):
            val = int(candidate)
            if 0 <= val <= 50:
                return None, val

    # Absolute last chance: handle weird 3-digit combos like '439'
    digits = re.sub(r"\D", "", t)
    if len(digits) == 3:
        last_two = int(digits[1:])
        if 0 <= last_two <= 50:
            return None, last_two

    return None, None


def _mode_str(values: List[Optional[str]]) -> Optional[str]:
    vals = [v for v in values if v]
    if not vals:
        return None
    from collections import Counter
    return Counter(vals).most_common(1)[0][0]


def _mode_int(values: List[Optional[int]]) -> Optional[int]:
    nums = [v for v in values if isinstance(v, int)]
    if not nums:
        return None
    from collections import Counter
    return Counter(nums).most_common(1)[0][0]


def _compose_legacy_score(
    a_text: Optional[str],
    h_text: Optional[str],
    a_int: Optional[int],
    h_int: Optional[int],
) -> Optional[str]:
    if a_int is not None and h_int is not None:
        return f"{a_int}-{h_int}"
    at = (a_text or "").strip()
    ht = (h_text or "").strip()
    if re.fullmatch(r"\d{1,2}", at or "") and re.fullmatch(r"\d{1,2}", ht or ""):
        return f"{int(at)}-{int(ht)}"
    return None


# ---------------------------------------------------------------------------
# Main public API – EasyOCR + skin auto-detect
# ---------------------------------------------------------------------------

def _ocr_score_int(patch: Optional[Image.Image]) -> Optional[int]:
    """
    Specialized OCR for scoreboard *scores* (not downs/distances).

    Uses the generic digit OCR, then applies a football-specific
    sanity check to fix common 7→1 misreads.
    """
    if patch is None:
        return None

    val = _ocr_digits_int(patch)
    if val is None:
        return None

    # Domain heuristic:
    # A raw score of exactly 1 is almost certainly a misread 7.
    # (You basically never see a team sitting at 1 point.)
    if val == 1:
        return 7

    return val

def extract_scoreboard_from_video_paddle(
    video_path: str,
    *,
    t: float = 0.10,
    attempts: int = 1,
    viz: bool = False,
    profile_key: Optional[str] = None,
) -> Dict[str, Any]:
    t_start = time.perf_counter()
    log.info(
        "=== [extract_scoreboard] Starting on %s (EasyOCR, attempts=%d, viz=%s, profile=%s) ===",
        video_path,
        attempts,
        viz,
        profile_key,
    )

    run_id = f"easyocr_{uuid.uuid4().hex[:8]}"
    run_dir = Path("data/tmp/ocr") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    if viz:
        (run_dir / "scorebar").mkdir(exist_ok=True)
        (run_dir / "rois").mkdir(exist_ok=True)
        (run_dir / "meta").mkdir(exist_ok=True)

    ts_list: List[float] = [max(0.0, t + 0.10 * k) for k in range(max(1, attempts))]

    # Initial preset (from profile_key if given; otherwise DEFAULT)
    chosen_preset = (profile_key or "DEFAULT").upper()
    if chosen_preset not in SCOREBAR_BOX_PRESETS:
        log.warning("Unknown profile_key=%s, will auto-detect from frame", profile_key)
        chosen_preset = "DEFAULT"

    away_team_reads: List[Optional[str]] = []
    home_team_reads: List[Optional[str]] = []
    away_score_text_reads: List[Optional[str]] = []
    home_score_text_reads: List[Optional[str]] = []
    away_score_int_reads: List[Optional[int]] = []
    home_score_int_reads: List[Optional[int]] = []
    clock_reads: List[Optional[str]] = []
    quarter_raw_reads: List[Optional[str]] = []
    quarter_norm_reads: List[Optional[str]] = []
    down_reads: List[Optional[int]] = []
    distance_reads: List[Optional[int]] = []
    distance_text_reads: List[Optional[str]] = []
    yard_side_reads: List[Optional[str]] = []
    yardline_reads: List[Optional[int]] = []

    auto_profile = None

    for i, ts in enumerate(ts_list):
        log.info(
            "=== [extract_scoreboard] Frame %d @ t=%.2fs (preset=%s) ===",
            i + 1,
            ts,
            chosen_preset,
        )
        frame_png = run_dir / f"frame_t{ts:.2f}.png"
        _grab_frame(video_path, ts, frame_png)

        with Image.open(frame_png) as full_pil:
            # Auto-detect skin once from first frame if profile_key not supplied
            if profile_key is None and i == 0:
                auto_profile = _detect_profile_from_topright(
                    full_pil,
                    viz_dir=(run_dir / "meta") if viz else None,
                )
                if auto_profile != chosen_preset:
                    log.info(
                        "[extract_scoreboard] Auto-detected skin %s (was %s)",
                        auto_profile,
                        chosen_preset,
                    )
                    chosen_preset = auto_profile

            sb_crop_np = crop_scorebar(full_pil, chosen_preset)

        sb_canon = to_canonical(sb_crop_np, chosen_preset)
        patches = extract_rois(sb_canon, chosen_preset, pad=2)

        if viz:
            Image.fromarray(sb_canon).save(
                run_dir / "scorebar" / f"{chosen_preset}_scorebar_t{ts:.2f}.png"
            )
            debug_img = _draw_roi_debug(sb_canon, chosen_preset)
            debug_img.save(
                run_dir
                / "scorebar"
                / f"{chosen_preset}_scorebar_debug_t{ts:.2f}.png"
            )
            for name, arr in patches.items():
                Image.fromarray(arr).save(
                    run_dir / "rois" / f"{i:02d}_{chosen_preset}_{name}.png"
                )

        def get_patch(name: str) -> Optional[Image.Image]:
            arr = patches.get(name)
            if arr is None:
                return None
            return Image.fromarray(arr)

        # Teams
        a_team = (
            _norm_team(_ocr_team(get_patch("away_team")) or "")
            if get_patch("away_team")
            else None
        )
        h_team = (
            _norm_team(_ocr_team(get_patch("home_team")) or "")
            if get_patch("home_team")
            else None
        )

        # Scores
                # Scores (use score-specific OCR to fix 7→1 misreads)
        a_sc_int = _ocr_score_int(get_patch("away_score"))
        h_sc_int = _ocr_score_int(get_patch("home_score"))

        a_sc_text = str(a_sc_int) if a_sc_int is not None else None
        h_sc_text = str(h_sc_int) if h_sc_int is not None else None

        # Clock: prefer main 'clock' ROI, then fall back to 'gameclock'
        clk = _ocr_clock(get_patch("clock")) if get_patch("clock") else None
        if not clk and get_patch("gameclock"):
            clk = _ocr_clock(get_patch("gameclock"))

        # Quarter
        q_raw = None
        q_norm = None
        if get_patch("quarter"):
            q_txt_raw = _ocr_generic_text(get_patch("quarter"))
            q_txt_clean = q_txt_raw.strip().upper()

            # Fix common EasyOCR confusions: Z↔2, A↔4, S↔5, etc.
            q_txt_fixed = (
                q_txt_clean
                .replace("Z", "2")
                .replace("A", "4")
                .replace("@", "4")
                .replace("5T", "ST")  # '5T' -> 'ST' in '1ST', '2ND' usually fine
            )

            if q_txt_fixed:
                q_raw = q_txt_fixed
                q_norm = _normalize_quarter(q_txt_fixed)

        # Down & distance – read separate ROIs, then combine for parsing
        down_raw = _ocr_generic_text(get_patch("down")) if get_patch("down") else ""
        dist_raw = _ocr_generic_text(get_patch("distance")) if get_patch("distance") else ""

        if down_raw or dist_raw:
            dd_raw = f"{down_raw} & {dist_raw}".strip(" &")
        else:
            dd_raw = ""

        log.debug(
            "[down/distance][easyocr] raw_down='%s' raw_dist='%s' combined='%s'",
            down_raw,
            dist_raw,
            dd_raw,
        )

        dwn, dist, dist_txt = _parse_down_distance(dd_raw) if dd_raw else (None, None, None)

        # Yardline
        yl_raw = (
            _ocr_generic_text(get_patch("yardline")) if get_patch("yardline") else ""
        )
        y_side, y_line = _parse_yardline(yl_raw) if yl_raw else (None, None)

        # Collect across frames
        away_team_reads.append(a_team)
        home_team_reads.append(h_team)
        away_score_text_reads.append(a_sc_text)
        home_score_text_reads.append(h_sc_text)
        away_score_int_reads.append(a_sc_int)
        home_score_int_reads.append(h_sc_int)
        clock_reads.append(clk)
        quarter_raw_reads.append(q_raw)
        quarter_norm_reads.append(q_norm)
        down_reads.append(dwn)
        distance_reads.append(dist)
        distance_text_reads.append(
            dist_txt or (str(dist) if dist is not None else None)
        )
        yard_side_reads.append(y_side)
        yardline_reads.append(y_line)

    log.info(
        "=== [extract_scoreboard] Finished EasyOCR in %.2f seconds ===",
        time.perf_counter() - t_start,
    )

    # Aggregate
    away_team = _mode_str(away_team_reads)
    home_team = _mode_str(home_team_reads)
    away_score_text = _mode_str(away_score_text_reads)
    home_score_text = _mode_str(home_score_text_reads)
    away_score_best = _mode_int(away_score_int_reads)
    home_score_best = _mode_int(home_score_int_reads)
    clk_best = _mode_str(clock_reads)

    quarter_text = _mode_str(quarter_raw_reads)
    quarter = _mode_str(quarter_norm_reads)

    down_best = _mode_int(down_reads)
    distance_best = _mode_int(distance_reads)
    distance_text_best = _mode_str(distance_text_reads)
    yard_side_best = _mode_str(yard_side_reads)
    yardline_best = _mode_int(yardline_reads)

    legacy_score = _compose_legacy_score(
        away_score_text, home_score_text, away_score_best, home_score_best
    )

    result: Dict[str, Any] = {
        "away_team": away_team,
        "home_team": home_team,
        "away_score_text": away_score_text,
        "home_score_text": home_score_text,
        "away_score": away_score_best,
        "home_score": home_score_best,
        "score": legacy_score,
        "clock": clk_best,
        "quarter_text": quarter_text,
        "quarter": quarter,
        "down": down_best,
        "distance": distance_best,
        "distance_text": distance_text_best,
        "yardline_side": yard_side_best,
        "yardline": yardline_best,
        "used_stub": False,
        "_preset": chosen_preset,
        "_auto_profile": auto_profile,
    }

    if not any(
        (
            away_team,
            home_team,
            away_score_text,
            home_score_text,
            away_score_best is not None,
            home_score_best is not None,
            legacy_score,
            clk_best,
            quarter_text,
            quarter,
            down_best,
            distance_best,
            yardline_best,
        )
    ):
        return {
            "used_stub": True,
            "_preset": chosen_preset,
            "_auto_profile": auto_profile,
        }

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="EasyOCR-based Madden scoreboard OCR")
    parser.add_argument(
        "video", help="Path to the video file (e.g. mr_tai_gameplay/clip.mp4)"
    )
    parser.add_argument(
        "--t",
        type=float,
        default=0.10,
        help="Timestamp (in seconds) for frame capture",
    )
    parser.add_argument(
        "--attempts", type=int, default=1, help="How many frames to sample"
    )
    parser.add_argument(
        "--viz",
        action="store_true",
        help="Enable visualization outputs (save crops, ROIs)",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        help="Skin preset: MNP, TNP, SNP, DEFAULT. If omitted, auto-detect via top-right label.",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Print extra debug info"
    )

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    result = extract_scoreboard_from_video_paddle(
        args.video,
        t=args.t,
        attempts=args.attempts,
        viz=args.viz,
        profile_key=args.profile,
    )

    print(json.dumps(result, indent=2))
