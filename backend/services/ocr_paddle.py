# backend/services/ocr_paddle.py
from __future__ import annotations

from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
import uuid, subprocess, shlex
import re

import numpy as np
from paddleocr import PaddleOCR

# Optional OpenCV (we fall back to ffmpeg if unavailable)
try:
    import cv2  # type: ignore
except Exception:
    cv2 = None

from PIL import Image, ImageDraw, ImageOps

# ------------------------
# Constants
# ------------------------
REF_FRAME_W, REF_FRAME_H = 2047, 1155
SCOREBAR_BOX = (0, 1080, 2047, 1155)  # bottom banner in reference frame

# Your fixed pixel ROIs on the scoreboard crop
SB_ROIS_PX: Dict[str, Tuple[int, int, int, int]] = {
    "away_team":  (500,  0,  600,  50),
    "home_team":  (900,  0, 1000,  50),
    "away_score": (655,  0,  745,  72),
    "home_score": (765,  0,  855,  72),
    "quarter":    (1280, 20, 1335, 50),  # first digit only; we'll normalize to Q1..Q4
    "clock":      (1325, 20, 1445, 50),
}

# ------------------------
# Frame / crop utils
# ------------------------
def _run_ffmpeg_grab(src: Path, ts: float, out_png: Path) -> bool:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    cmd = f'ffmpeg -y -v error -ss {ts:.3f} -i {shlex.quote(str(src))} -frames:v 1 {shlex.quote(str(out_png))}'
    subprocess.run(cmd, shell=True, check=False)
    return out_png.exists()

def _grab_frame(video_path: str, ts: float, out_png: Path) -> Path:
    src = Path(video_path)
    if cv2 is not None:
        cap = cv2.VideoCapture(str(src))
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, ts * 1000.0))
            ok, frame = cap.read()
            cap.release()
            if ok and frame is not None:
                rgb = frame[:, :, ::-1]  # BGR->RGB
                Image.fromarray(rgb).save(out_png)
                return out_png
    if _run_ffmpeg_grab(src, ts, out_png):
        return out_png
    raise RuntimeError(f"Could not grab frame at t={ts:.3f}s")

def _scale_box_to_frame(box: Tuple[int, int, int, int], w: int, h: int) -> Tuple[int, int, int, int]:
    lx, ty, rx, by = box
    sx = w / REF_FRAME_W
    sy = h / REF_FRAME_H
    return (int(lx * sx), int(ty * sy), int(rx * sx), int(by * sy))

def _crop_scorebar(frame_png: Path, out_png: Path) -> Tuple[Path, Tuple[int, int]]:
    with Image.open(frame_png) as im:
        w, h = im.size
        lx, ty, rx, by = _scale_box_to_frame(SCOREBAR_BOX, w, h)
        crop = im.crop((lx, ty, rx, by))
        out_png.parent.mkdir(parents=True, exist_ok=True)
        crop.save(out_png)
        return out_png, crop.size

def _crop_px_from_ref(sb_img: Image.Image, box_px: Tuple[int, int, int, int], ref_size: Tuple[int, int]) -> Image.Image:
    w, h = sb_img.size
    ref_w, ref_h = ref_size
    sx = w / float(ref_w)
    sy = h / float(ref_h)
    x0, y0, x1, y1 = box_px
    X0, Y0 = max(0, int(round(x0 * sx))), max(0, int(round(y0 * sy)))
    X1, Y1 = min(w, int(round(x1 * sx))), min(h, int(round(y1 * sy)))
    return sb_img.crop((X0, Y0, X1, Y1))

def _prep_gray_bw(pil: Image.Image, thresh: int = 160, invert: bool = False) -> Image.Image:
    g = pil.convert("L")
    table = [0] * (thresh + 1) + [255] * (256 - (thresh + 1))
    bw = g.point(table, mode="L")
    if invert:
        bw = ImageOps.invert(bw)
    return bw

# ------------------------
# Visualization
# ------------------------
def _draw_frame_scorebar_outline(frame_png: Path, out_png: Path) -> None:
    with Image.open(frame_png) as im:
        w, h = im.size
        lx, ty, rx, by = _scale_box_to_frame(SCOREBAR_BOX, w, h)
        vis = im.copy()
        d = ImageDraw.Draw(vis)
        d.rectangle((lx, ty, rx, by), outline="lime", width=4)
        out_png.parent.mkdir(parents=True, exist_ok=True)
        vis.save(out_png)

def _viz_rois_on_scorebar(scorebar_png: Path, ref_size: Tuple[int, int], out_png: Path) -> None:
    with Image.open(scorebar_png) as sb:
        w, h = sb.size
        sx = w / float(ref_size[0])
        sy = h / float(ref_size[1])

        vis = sb.copy()
        d = ImageDraw.Draw(vis)
        colors = {
            "away_team":  "orange",
            "home_team":  "orange",
            "away_score": "lime",
            "home_score": "lime",
            "quarter":    "deepskyblue",
            "clock":      "deepskyblue",
        }
        for key, (x0, y0, x1, y1) in SB_ROIS_PX.items():
            X0, Y0 = int(round(x0 * sx)), int(round(y0 * sy))
            X1, Y1 = int(round(x1 * sx)), int(round(y1 * sy))
            d.rectangle((X0, Y0, X1, Y1), outline=colors.get(key, "yellow"), width=3)
            d.text((X0 + 4, max(0, Y0 - 14)), key, fill=colors.get(key, "yellow"))
        out_png.parent.mkdir(parents=True, exist_ok=True)
        vis.save(out_png)

# ------------------------
# OCR init (CACHED SINGLETON)
# ------------------------
_OCR_SINGLETON: Optional[PaddleOCR] = None

def _ocr_init() -> PaddleOCR:
    """
    Create or reuse a single PaddleOCR instance (prevents reloading the model for every request).
    """
    global _OCR_SINGLETON
    if _OCR_SINGLETON is None:
        _OCR_SINGLETON = PaddleOCR(use_angle_cls=False, lang='en', show_log=False)
    return _OCR_SINGLETON

def _ocr_text(ocr: PaddleOCR, pil_img: Image.Image) -> str:
    try:
        arr = np.array(pil_img.convert("RGB"))
        res = ocr.ocr(arr, cls=False)
    except Exception:
        return ""
    if not res or not isinstance(res, list) or not res[0]:
        return ""
    lines = res[0]
    try:
        best = max(lines, key=lambda ln: (ln[1][1] if isinstance(ln, list) and len(ln) > 1 else 0.0))
        return (best[1][0] or "").strip()
    except Exception:
        return ""

# ------------------------
# Normalization + voting
# ------------------------
_CLOCK_RX = re.compile(r"^\s*(\d{1,2}):([0-5]\d)\s*$", re.I)
_Q_TRIM = re.compile(r"\s+")

def _norm_team(s: str) -> Optional[str]:
    s = (s or "").strip()
    if not s:
        return None
    return re.sub(r"[^A-Z]", "", s.upper()) or None

def _digits2(s: str) -> Optional[int]:
    m = re.search(r"\d{1,2}", (s or ""))
    return int(m.group(0)) if m else None

def _norm_clock(s: str) -> Optional[str]:
    if not s:
        return None
    x = s.strip().replace(";", ":").replace(" ", "")
    m = _CLOCK_RX.match(x)
    if m:
        return f"{int(m.group(1))}:{m.group(2)}"
    return None

def _norm_quarter(s: str) -> Optional[str]:
    if not s:
        return None
    x = _Q_TRIM.sub("", s.upper())
    # If it's a single digit from our tight crop, normalize to Q#
    if re.fullmatch(r"[1-4]", x):
        return f"Q{x}"
    # Handle common variants
    x = x.replace("0T", "OT").replace("QI", "Q1").replace("QT", "Q1")
    x = re.sub(r"1{2,}", "1", x)
    if re.fullmatch(r"Q[1-4]|OT", x):
        return x
    if x in {"1ST", "2ND", "3RD", "4TH"}:
        return {"1ST": "Q1", "2ND": "Q2", "3RD": "Q3", "4TH": "Q4"}[x]
    return None

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

# ------------------------
# Public API with aggregation across frames
# ------------------------
def extract_scoreboard_from_video_paddle(
    video_path: str,
    *,
    t: float = 0.10,
    attempts: int = 3,
    viz: bool = False,
) -> Dict[str, Any]:
    """
    Sample several frames; OCR each ROI; aggregate across frames:
      - team names: mode of normalized strings
      - scores: mode of per-side integers
      - clock/quarter: mode of normalized strings
    """
    run_id = f"ocr_{uuid.uuid4().hex[:8]}"
    run_dir = Path("data/tmp/ocr") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    ts_list: List[float] = [max(0.0, t + 0.10 * k) for k in range(max(1, attempts))]
    ocr = _ocr_init()  # <-- cached singleton now

    ref_size: Optional[Tuple[int, int]] = None

    # Collect across frames
    away_team_reads: List[Optional[str]] = []
    home_team_reads: List[Optional[str]] = []
    away_score_reads: List[Optional[int]] = []
    home_score_reads: List[Optional[int]] = []
    clock_reads: List[Optional[str]] = []
    quarter_reads: List[Optional[str]] = []

    for i, ts in enumerate(ts_list):
        frame_png = run_dir / f"frame_t{ts:.2f}.png"
        score_png = run_dir / f"scorebar_t{ts:.2f}.png"

        _grab_frame(video_path, ts, frame_png)
        score_png, (sbw, sbh) = _crop_scorebar(frame_png, score_png)
        if ref_size is None:
            ref_size = (sbw, sbh)

        if viz and i == 0:
            _draw_frame_scorebar_outline(frame_png, run_dir / f"frame_with_scorebar_t{ts:.2f}.png")
            _viz_rois_on_scorebar(score_png, ref_size, run_dir / f"scorebar_rois_t{ts:.2f}.png")

        with Image.open(score_png) as sb:
            def grab_txt(key: str) -> str:
                crop = _crop_px_from_ref(sb, SB_ROIS_PX[key], ref_size)  # type: ignore[arg-type]
                crop_bw = _prep_gray_bw(crop, thresh=160, invert=False)
                return _ocr_text(ocr, crop_bw)

            a_team_raw = grab_txt("away_team")
            h_team_raw = grab_txt("home_team")
            a_sc_raw   = grab_txt("away_score")
            h_sc_raw   = grab_txt("home_score")
            q_raw      = grab_txt("quarter")
            c_raw      = grab_txt("clock")

        away_team_reads.append(_norm_team(a_team_raw))
        home_team_reads.append(_norm_team(h_team_raw))
        away_score_reads.append(_digits2(a_sc_raw))
        home_score_reads.append(_digits2(h_sc_raw))
        clock_reads.append(_norm_clock(c_raw))
        quarter_reads.append(_norm_quarter(q_raw))

    # Aggregate across frames (fixes Codex “completeness per frame” concern)
    away_team = _mode_str(away_team_reads)
    home_team = _mode_str(home_team_reads)
    a_mode = _mode_int(away_score_reads)
    h_mode = _mode_int(home_score_reads)
    clk = _mode_str(clock_reads)
    qtr = _mode_str(quarter_reads)

    result: Dict[str, Any] = {
        "away_team": away_team,
        "home_team": home_team,
        "score": (f"{a_mode}-{h_mode}" if (a_mode is not None and h_mode is not None) else None),
        "clock": clk,
        "quarter": qtr,
        "used_stub": False,
    }

    if not any((away_team, home_team, a_mode is not None, h_mode is not None, clk, qtr)):
        return {"used_stub": True}

    if viz:
        result["_viz_dir"] = str(run_dir)

    return result

if __name__ == "__main__":
    import sys, json
    video = sys.argv[1] if len(sys.argv) > 1 else "Madden Clip.mp4"
    flags = set(a for a in sys.argv[2:] if a.startswith("--"))
    viz = ("--viz" in flags)
    data = extract_scoreboard_from_video_paddle(video, viz=viz)
    print(json.dumps(data, indent=2))
