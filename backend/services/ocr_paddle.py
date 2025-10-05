# backend/services/ocr_paddle.py
from __future__ import annotations

from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
import uuid, subprocess, shlex
import re

import numpy as np
from paddleocr import PaddleOCR

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

# Your fixed pixel ROIs on the scoreboard crop (do not change)
SB_ROIS_PX: Dict[str, Tuple[int, int, int, int]] = {
    "away_team":  (500,  0,  600,  50),
    "home_team":  (900,  0, 1000,  50),
    "away_score": (655,  0,  745,  72),
    "home_score": (765,  0,  855,  72),
    "quarter":    (1280, 20, 1330, 50),   # narrow crop: first digit only
    "clock":      (1325, 20, 1445, 50),
}


# ------------------------
# Utilities
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
# OCR helpers
# ------------------------
def _ocr_init() -> PaddleOCR:
    return PaddleOCR(use_angle_cls=False, lang='en', show_log=False)

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

def _ocr_quarter_digit(ocr: PaddleOCR, pil_img: Image.Image) -> Optional[str]:
    """
    Read a single quarter digit (1..4) from a tight crop.
    Tries a few binarization thresholds and inverted/normal modes; returns first valid.
    Returns 'OT' if that appears explicitly.
    """
    for thresh in (145, 165, 185):
        for inv in (False, True):
            bw = _prep_gray_bw(pil_img, thresh=thresh, invert=inv)
            txt = _ocr_text(ocr, bw) or ""
            if "OT" in txt.upper():
                return "OT"
            txt = txt.replace("I", "1").replace("|", "1").replace("l", "1")
            m = re.search(r"[1-4]", txt)
            if m:
                return m.group(0)
    return None

def _ordinal_from_digit(d: Optional[str]) -> Optional[str]:
    if not d:
        return None
    d = d.strip()
    if d == "1": return "1st"
    if d == "2": return "2nd"
    if d == "3": return "3rd"
    if d == "4": return "4th"
    if d.upper() == "OT": return "OT"
    return None


# ------------------------
# Normalization + voting
# ------------------------
_CLOCK_RX = re.compile(r"^\s*(\d{1,2}):([0-5]\d)\s*$", re.I)

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
      - clock: mode of normalized strings
      - quarter: read first digit/OT and map to ordinal (1st/2nd/3rd/4th/OT)
    """
    run_id = f"ocr_{uuid.uuid4().hex[:8]}"
    run_dir = Path("data/tmp/ocr") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    ts_list: List[float] = [max(0.0, t + 0.10 * k) for k in range(max(1, attempts))]
    ocr = _ocr_init()

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

            # teams & scores via general OCR
            a_team_raw = grab_txt("away_team")
            h_team_raw = grab_txt("home_team")
            a_sc_raw   = grab_txt("away_score")
            h_sc_raw   = grab_txt("home_score")
            c_raw      = grab_txt("clock")

            # quarter: read a single digit (or OT) from the tight crop, then map to ordinal
            q_crop = _crop_px_from_ref(sb, SB_ROIS_PX["quarter"], ref_size)
            q_digit_or_ot = _ocr_quarter_digit(ocr, q_crop)
            q_ordinal = _ordinal_from_digit(q_digit_or_ot)

        away_team_reads.append(_norm_team(a_team_raw))
        home_team_reads.append(_norm_team(h_team_raw))
        away_score_reads.append(_digits2(a_sc_raw))
        home_score_reads.append(_digits2(h_sc_raw))
        clock_reads.append(_norm_clock(c_raw))
        quarter_reads.append(q_ordinal)

    # Aggregate
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
        "quarter": qtr,  # "1st" / "2nd" / "3rd" / "4th" / "OT"
        "used_stub": False,
    }

    # If we didn’t get anything at all, return stub
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
