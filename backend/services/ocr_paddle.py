# backend/services/ocr_paddle.py
from __future__ import annotations

import os
import shlex
import subprocess
import time
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import Counter
from threading import Lock

import numpy as np

# Pillow
from PIL import Image, ImageDraw

# Optional OpenCV (faster frame grabs if available)
try:
    import cv2  # type: ignore
except Exception:
    cv2 = None  # fallback to ffmpeg

# PaddleOCR (lazy init)
try:
    from paddleocr import PaddleOCR  # type: ignore
except Exception as e:
    PaddleOCR = None  # type: ignore


# ---------- Reference full-frame -> scoreboard crop ----------
# Reference full-frame: 2047x1155, scoreboard y=[1080,1155)
REF_FRAME_W, REF_FRAME_H = 2047, 1155
SCOREBAR_BOX = (0, 1080, 2047, 1155)  # (left, top, right, bottom) in reference space

# ---------- Your exact pixel ROIs on the scoreboard crop (x0, y0, x1, y1) ----------
# (Keep these as you tuned them for Madden.)
SB_ROIS_PX: Dict[str, Tuple[int, int, int, int]] = {
    "away_team":  (500,  0, 600,  50),
    "home_team":  (900,  0, 1000, 50),
    "away_score": (655,  0, 745,  72),
    "home_score": (765,  0, 855,  72),
    "quarter":    (1285, 20, 1340, 50),
    "clock":      (1325, 20, 1445, 50),
}

# ---------- Regex / normalizers ----------
_SCORE_RX   = re.compile(r"^\s*(\d{1,2})\s*[-:]\s*(\d{1,2})\s*$")
_CLOCK_RX   = re.compile(r"^\s*(\d{1,2}):([0-5]\d)\s*$")
_QTR_RX_QN  = re.compile(r"^(Q[1-4]|OT)$", re.I)
_QTR_RX_MDN = re.compile(r"^(1ST|2ND|3RD|4TH|OT)$", re.I)

def _normalize_quarter(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    x = s.strip().upper()
    x = re.sub(r"1{2,}", "1", x)                 # "111st" -> "1st"
    x = x.replace(" ", "").replace("0T", "OT")   # "0T" -> "OT"
    x = x.replace("QI", "Q1").replace("QT", "Q1")
    m = _QTR_RX_QN.match(x)
    if m:
        return m.group(1).upper()
    m2 = _QTR_RX_MDN.match(x)
    if m2:
        return {"1ST":"Q1","2ND":"Q2","3RD":"Q3","4TH":"Q4","OT":"OT"}[m2.group(1)]
    if re.fullmatch(r"[1-4]", x):
        return f"Q{x}"
    return None

def _normalize_clock(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    x = s.strip()
    if not re.search(r"\d", x):
        return None
    x = x.replace(";", ":").replace(" ", "")
    m = _CLOCK_RX.match(x)
    if m:
        return f"{int(m.group(1))}:{m.group(2)}"
    return None

def _digits1_2(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    m = re.search(r"\d{1,2}", s)
    return m.group(0) if m else None

def _compose_score(away_s: Optional[str], home_s: Optional[str], joined: Optional[str]) -> Optional[str]:
    a = _digits1_2(away_s)
    h = _digits1_2(home_s)
    if a is not None and h is not None:
        return f"{int(a)}-{int(h)}"
    if joined:
        m = _SCORE_RX.match(joined.replace(":", "-"))
        if m:
            return f"{int(m.group(1))}-{int(m.group(2))}"
    return None

def _letters_only(s: str) -> str:
    return re.sub(r"[^A-Z]", "", s.upper())


# ---------- PaddleOCR singleton ----------
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from paddleocr import PaddleOCR

_OCR_SINGLETON: Optional["paddleocr.PaddleOCR"] = None  # type: ignore
_OCR_LOCK = Lock()

def _get_ocr() -> Optional["paddleocr.PaddleOCR"]:  # type: ignore
    """Thread-safe singleton initializer for PaddleOCR"""
    global _OCR_SINGLETON
    if PaddleOCR is None:
        return None
    with _OCR_LOCK:
        if _OCR_SINGLETON is None:
            # Initialize once; MPS on Apple Silicon is handled by paddlepaddle wheels
            _OCR_SINGLETON = PaddleOCR(use_angle_cls=False, lang="en", show_log=False)
        return _OCR_SINGLETON

# ---------- Frame helpers ----------
def grab_frame_at_time(video_path: str, ts: float, out_path: str) -> Path:
    """
    Save a PNG for the frame at timestamp 'ts' (seconds).
    Tries OpenCV first; falls back to ffmpeg -ss.
    """
    src = Path(video_path)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    if cv2 is not None:
        cap = cv2.VideoCapture(str(src))
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, ts * 1000.0))
            ok, frame = cap.read()
            cap.release()
            if ok and frame is not None:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                Image.fromarray(frame_rgb).save(out)
                return out

    cmd = f'ffmpeg -y -v error -ss {ts:.3f} -i {shlex.quote(str(src))} -frames:v 1 {shlex.quote(str(out))}'
    subprocess.run(cmd, shell=True, check=False)
    if out.exists():
        return out

    raise RuntimeError(f"Could not grab frame at t={ts:.3f}s. Install opencv-python or ffmpeg.")

def _scale_box_to_frame(box: Tuple[int, int, int, int], w: int, h: int) -> Tuple[int, int, int, int]:
    lx, ty, rx, by = box
    sx = w / REF_FRAME_W
    sy = h / REF_FRAME_H
    return (int(lx * sx), int(ty * sy), int(rx * sx), int(by * sy))

def crop_scorebar_from_frame(frame_path: str, out_path: str, box: Tuple[int, int, int, int] = SCOREBAR_BOX) -> Path:
    """Crop the scoreboard bar from a full frame; auto-scales box if frame != 2047x1155."""
    ip = Path(frame_path)
    op = Path(out_path)
    op.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(ip) as im:
        w, h = im.size
        if (w, h) != (REF_FRAME_W, REF_FRAME_H):
            lx, ty, rx, by = _scale_box_to_frame(box, w, h)
        else:
            lx, ty, rx, by = box
        crop = im.crop((lx, ty, rx, by))
        crop.save(op)
        return op


# ---------- Per-call ROI scaling & OCR ----------
def _crop_px_from_ref(sb_img: Image.Image,
                      box_px: Tuple[int, int, int, int],
                      ref_size: Tuple[int, int]) -> Image.Image:
    """Scale pixel ROI from per-call ref_size onto current scorebar image."""
    ref_w, ref_h = ref_size
    w, h = sb_img.size
    sx = w / float(ref_w)
    sy = h / float(ref_h)
    x0, y0, x1, y1 = box_px
    X0, Y0 = max(0, int(round(x0 * sx))), max(0, int(round(y0 * sy)))
    X1, Y1 = min(w, int(round(x1 * sx))), min(h, int(round(y1 * sy)))
    return sb_img.crop((X0, Y0, X1, Y1))

def _ocr_text_from_pil(pil_img: Image.Image) -> str:
    """
    Run PaddleOCR on a PIL image without touching the filesystem.
    Paddle expects BGR ndarray; convert from RGB first.
    """
    ocr = _get_ocr()
    if ocr is None:
        return ""
    rgb = np.array(pil_img.convert("RGB"))
    bgr = rgb[:, :, ::-1]  # RGB -> BGR
    res = ocr.ocr(bgr, cls=False)
    if not res or not res[0]:
        return ""
    return " ".join((line[1][0] or "").strip() for line in res[0] if line and line[1])


def _viz_px_rois(scorebar_pil: Image.Image, out_path: str, ref_size: Tuple[int, int]) -> None:
    w, h = scorebar_pil.size
    ref_w, ref_h = ref_size
    sx = w / float(ref_w)
    sy = h / float(ref_h)
    vis = scorebar_pil.copy()
    d = ImageDraw.Draw(vis)
    colors = {
        "away_team": "orange",
        "home_team": "orange",
        "away_score": "lime",
        "home_score": "lime",
        "quarter": "deepskyblue",
        "clock": "deepskyblue",
    }
    for key, (x0, y0, x1, y1) in SB_ROIS_PX.items():
        X0, Y0 = int(round(x0 * sx)), int(round(y0 * sy))
        X1, Y1 = int(round(x1 * sx)), int(round(y1 * sy))
        d.rectangle((X0, Y0, X1, Y1), outline=colors.get(key, "yellow"), width=3)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    vis.save(out_path)


# ---------- Main entry ----------
def extract_scoreboard_from_video_paddle(
    video_path: str,
    *,
    t: float = 0.10,          # start slightly after 0s
    attempts: int = 3,        # quick voting
    viz: bool = False,
) -> Dict[str, Any]:
    """
    Grab a few frames, crop your fixed scoreboard strip, OCR per-ROI via PaddleOCR in-memory,
    and vote for stability. Returns:
      {away_team, home_team, away_score, home_score, score, clock, quarter, used_stub}
    """
    # Early check: if PaddleOCR isn't present, bail with stub so the caller can fall back.
    if _get_ocr() is None:
        return {"used_stub": True}

    ts_list: List[float] = [max(0.0, t + k * 0.10) for k in range(max(1, attempts))]

    # Voting pools
    away_teams: List[str] = []
    home_teams: List[str] = []
    away_scores: List[str] = []
    home_scores: List[str] = []
    clocks: List[str] = []
    quarters: List[str] = []

    first_scorebar: Optional[Image.Image] = None
    ref_size: Optional[Tuple[int, int]] = None

    for idx, ts in enumerate(ts_list):
        # 1) frame -> scorebar crop
        frame_png = grab_frame_at_time(video_path, ts, out_path=f"data/tmp/video_frame_t{ts:.2f}.png")
        scorebar_png = crop_scorebar_from_frame(str(frame_png), out_path=f"data/tmp/scorebar_t{ts:.2f}.png")
        with Image.open(scorebar_png) as sb:
            scorebar_pil = sb.convert("RGB")

        if first_scorebar is None:
            first_scorebar = scorebar_pil
            ref_size = first_scorebar.size
            if viz:
                _viz_px_rois(scorebar_pil, f"data/tmp/scorebar_viz_t{ts:.2f}.png", ref_size)

        assert ref_size is not None

        # 2) crop all ROIs relative to per-call ref_size
        a_team_img  = _crop_px_from_ref(scorebar_pil, SB_ROIS_PX["away_team"],  ref_size)
        h_team_img  = _crop_px_from_ref(scorebar_pil, SB_ROIS_PX["home_team"],  ref_size)
        a_score_img = _crop_px_from_ref(scorebar_pil, SB_ROIS_PX["away_score"], ref_size)
        h_score_img = _crop_px_from_ref(scorebar_pil, SB_ROIS_PX["home_score"], ref_size)
        qtr_img     = _crop_px_from_ref(scorebar_pil, SB_ROIS_PX["quarter"],    ref_size)
        clk_img     = _crop_px_from_ref(scorebar_pil, SB_ROIS_PX["clock"],      ref_size)

        # 3) OCR in-memory
        a_team_txt = _letters_only(_ocr_text_from_pil(a_team_img))
        h_team_txt = _letters_only(_ocr_text_from_pil(h_team_img))
        a_score_txt = _ocr_text_from_pil(a_score_img)
        h_score_txt = _ocr_text_from_pil(h_score_img)
        qtr_txt = _normalize_quarter(_ocr_text_from_pil(qtr_img))
        clk_txt = _normalize_clock(_ocr_text_from_pil(clk_img))

        # 4) normalize scores to digits (keep single/two-digit)
        a_score_d = _digits1_2(a_score_txt)
        h_score_d = _digits1_2(h_score_txt)

        if a_team_txt:
            away_teams.append(a_team_txt)
        if h_team_txt:
            home_teams.append(h_team_txt)
        if a_score_d is not None:
            away_scores.append(a_score_d)
        if h_score_d is not None:
            home_scores.append(h_score_d)
        if qtr_txt:
            quarters.append(qtr_txt)
        if clk_txt:
            clocks.append(clk_txt)

    # Voting helpers
    def _mode_str(vals: List[str]) -> Optional[str]:
        vals = [v for v in vals if v]
        return Counter(vals).most_common(1)[0][0] if vals else None

    away_team = _mode_str(away_teams)
    home_team = _mode_str(home_teams)
    away_sc   = _mode_str(away_scores)
    home_sc   = _mode_str(home_scores)
    clock     = _mode_str(clocks)
    quarter   = _mode_str(quarters)

    score = _compose_score(away_sc, home_sc, None)

    # Build result
    out: Dict[str, Any] = {
        "away_team": away_team,
        "home_team": home_team,
        "away_score": away_sc,
        "home_score": home_sc,
        "score": score,
        "clock": clock,
        "quarter": quarter,
        "used_stub": False,
    }

    # If nothing meaningful was extracted, mark as stub so caller can decide fallback.
    if not any([away_team, home_team, away_sc, home_sc, clock, quarter]):
        return {"used_stub": True}

    return out


if __name__ == "__main__":
    import sys, json
    video = sys.argv[1] if len(sys.argv) > 1 else "Madden Clip.mp4"
    flags = {a for a in sys.argv[2:] if a.startswith("--") and "=" not in a}
    kvs = [a for a in sys.argv[2:] if a.startswith("--") and "=" in a]
    viz = ("--viz" in flags)
    t = 0.10
    attempts = 3
    for kv in kvs:
        if kv.startswith("--t="):
            try: t = float(kv.split("=", 1)[1])
            except: pass
        elif kv.startswith("--attempts="):
            try: attempts = int(kv.split("=", 1)[1])
            except: pass

    data = extract_scoreboard_from_video_paddle(video, t=t, attempts=attempts, viz=viz)
    print(json.dumps(data, indent=2))
