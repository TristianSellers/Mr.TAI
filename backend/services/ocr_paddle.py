# backend/services/ocr_paddle.py
from __future__ import annotations

from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
import uuid, subprocess, shlex, os, re

import numpy as np
from paddleocr import PaddleOCR

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None

from PIL import Image, ImageDraw, ImageOps

# ------------------------
# Constants (reference full frame)
# ------------------------
REF_FRAME_W, REF_FRAME_H = 2047, 1155

# Original/default bottom banner crop (fallback if preset not detected)
SCOREBAR_BOX_DEFAULT = (0, 1080, 2047, 1155)

# Top-right label region in FULL FRAME (your estimate)
TOPRIGHT_BOX_REF = (1850, 30, 1950, 90)

# Scoreboard crop windows per preset (in FULL FRAME coords)
SCOREBAR_BOX_PRESETS: Dict[str, Tuple[int, int, int, int]] = {
    # MNP Crop guidelines:
    # Y: 1080 to 1150, X: 0 to 2000
    "MNP": (0, 1080, 2000, 1150),
    # TNP Crop guidelines:
    # Y: 1020 to 1145, X: 630 to 1420
    "TNP": (630, 1020, 1420, 1145),
    # SNP Crop guidelines:
    # Y: 1040 to 1140, X: 590 to 1510
    "SNP": (590, 1040, 1510, 1140),
}

# ------------------------
# ROI PRESETS per scoreboard type (coords are on SCOREBOARD CROP)
# ------------------------
ROI_PRESETS: Dict[str, Dict[str, Tuple[int, int, int, int]]] = {
    # --- Default (white bottom banner) ---
    "DEFAULT": {
        "away_team":  (500,  0,  600,  50),
        "home_team":  (900,  0, 1000,  50),
        "away_score": (655,  0,  745,  72),
        "home_score": (765,  0,  855,  72),
        # Quarter ROI here is first-digit area (we normalize to Q1..Q4 later)
        "quarter":    (1280, 20, 1305, 50),
        "clock":      (1325, 20, 1445, 50),
    },

    # --- MNP preset ---
    "MNP": {
        "away_team":  (510, 15, 585, 55),
        "away_score": (625, 10, 725, 60),
        "home_team":  (910, 15, 985, 55),
        "home_score": (775, 10, 875, 60),
        "quarter":    (1260, 15, 1315, 55),
        "clock":      (1315, 15, 1385, 55),
    },

    # --- TNP preset ---
    "TNP": {
        "away_team":  (185, 25, 250, 60),
        "away_score": (250, 0, 335, 65),
        "home_team":  (485, 25, 550, 60),
        "home_score": (400, 0, 485, 65),
        "quarter":    (275, 65, 320, 110),
        "clock":      (325, 65, 400, 110),
    },

    # --- SNP preset ---
    "SNP": {
        "away_team":  (80, 10, 130, 55),
        "away_score": (130, 15, 185, 55),
        "home_team":  (415, 10, 460, 55),
        "home_score": (355, 15, 415, 55),
        "quarter":    (210, 30, 235, 60),
        "clock":      (235, 30, 275, 60),
    },
}

# ------------------------
# OCR singleton
# ------------------------
_OCR_SINGLETON: Optional[PaddleOCR] = None

def _ocr_init() -> PaddleOCR:
    global _OCR_SINGLETON
    if _OCR_SINGLETON is None:
        _OCR_SINGLETON = PaddleOCR(use_angle_cls=False, lang='en', show_log=False)
    return _OCR_SINGLETON

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

def _crop_rect_safe(im: Image.Image, x0: int, y0: int, x1: int, y1: int) -> Image.Image:
    # normalize
    if x1 < x0: x0, x1 = x1, x0
    if y1 < y0: y0, y1 = y1, y0
    # clamp
    w, h = im.size
    x0 = max(0, min(x0, w - 1))
    y0 = max(0, min(y0, h - 1))
    x1 = max(x0 + 1, min(x1, w))
    y1 = max(y0 + 1, min(y1, h))
    return im.crop((x0, y0, x1, y1))

def _crop_from_full(frame_png: Path, full_box: Tuple[int, int, int, int]) -> Tuple[Image.Image, Tuple[int, int]]:
    with Image.open(frame_png) as im:
        w, h = im.size
        lx, ty, rx, by = _scale_box_to_frame(full_box, w, h)
        crop = _crop_rect_safe(im, lx, ty, rx, by)
        return crop, crop.size

def _crop_scorebar(frame_png: Path, full_box: Tuple[int, int, int, int], out_png: Path) -> Tuple[Path, Tuple[int, int]]:
    crop, _ = _crop_from_full(frame_png, full_box)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    crop.save(out_png)
    return out_png, crop.size

def _crop_px_from_ref(sb_img: Image.Image, box_px: Tuple[int, int, int, int], ref_size: Tuple[int, int]) -> Image.Image:
    w, h = sb_img.size
    ref_w, ref_h = ref_size
    if ref_w <= 0 or ref_h <= 0:
        return sb_img.copy()
    sx = w / float(ref_w)
    sy = h / float(ref_h)
    x0, y0, x1, y1 = box_px
    X0, Y0 = int(round(x0 * sx)), int(round(y0 * sy))
    X1, Y1 = int(round(x1 * sx)), int(round(y1 * sy))
    return _crop_rect_safe(sb_img, X0, Y0, X1, Y1)

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
def _draw_frame_scorebar_outline(frame_png: Path, full_box: Tuple[int, int, int, int], out_png: Path) -> None:
    with Image.open(frame_png) as im:
        w, h = im.size
        lx, ty, rx, by = _scale_box_to_frame(full_box, w, h)
        vis = im.copy()
        d = ImageDraw.Draw(vis)
        d.rectangle((lx, ty, rx, by), outline="lime", width=4)
        out_png.parent.mkdir(parents=True, exist_ok=True)
        vis.save(out_png)

def _viz_rois_on_scorebar(scorebar_png: Path, ref_size: Tuple[int, int], roi_map: Dict[str, Tuple[int,int,int,int]], out_png: Path) -> None:
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
        for key, (x0, y0, x1, y1) in roi_map.items():
            X0, Y0 = int(round(x0 * sx)), int(round(y0 * sy))
            X1, Y1 = int(round(x1 * sx)), int(round(y1 * sy))
            d.rectangle((X0, Y0, X1, Y1), outline=colors.get(key, "yellow"), width=3)
            d.text((X0 + 4, max(0, Y0 - 14)), key, fill=colors.get(key, "yellow"))
        out_png.parent.mkdir(parents=True, exist_ok=True)
        vis.save(out_png)

# ------------------------
# Paddle OCR helpers
# ------------------------
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

def _ocr_letters_strict(ocr: PaddleOCR, pil_img: Image.Image, viz_dir: Optional[Path] = None, stamp: str = "") -> str:
    """
    Try a few upscales + thresholds; keep only A-Z; return longest/most confident.
    """
    cand: List[Tuple[float, str]] = []
    idx = 0
    for scale in (2, 3):
        im = pil_img.resize((max(1, pil_img.width * scale), max(1, pil_img.height * scale)))
        for thr in (150, 165, 180, 195):
            for inv in (False, True):
                bw = _prep_gray_bw(im, thresh=thr, invert=inv)
                txt = _ocr_text(ocr, bw)
                txt2 = re.sub(r"[^A-Z]", "", txt.upper())
                if txt2:
                    score = len(txt2) + (0.1 if re.search(r"[A-Za-z]", txt) else 0.0)
                    cand.append((score, txt2))
                if viz_dir:
                    try:
                        bw.save(viz_dir / f"topright_bw_{stamp}_{idx}.png")
                    except Exception:
                        pass
                idx += 1
    if not cand:
        txt = _ocr_text(ocr, pil_img)
        return re.sub(r"[^A-Z]", "", (txt or "").upper())
    cand.sort(key=lambda t: t[0], reverse=True)
    return cand[0][1]

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

def _norm_quarter(s: str) -> Optional[str]:
    if not s:
        return None
    # Since some presets use just the first digit region, convert "1"->"Q1" etc.
    x = re.sub(r"[^0-9A-Z]", "", s.upper())
    # digit-only
    m = re.fullmatch(r"[1-4]", re.sub(r"[^0-9]", "", x))
    if m:
        return f"Q{m.group(0)}"
    # text forms
    if x in {"1ST", "2ND", "3RD", "4TH"}:
        return {"1ST": "Q1", "2ND": "Q2", "3RD": "Q3", "4TH": "Q4"}[x]
    if x in {"Q1", "Q2", "Q3", "Q4", "OT"}:
        return x
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
# Preset detection from top-right label
# ------------------------
def _detect_preset_from_topright(frame_png: Path, ocr: PaddleOCR, viz_dir: Optional[Path]) -> Optional[str]:
    """
    Crop the top-right label, OCR letters, and decide:
      contains 'S' or 'SNP' -> SNP
      contains 'T' or 'TNP' -> TNP
      contains 'M' or 'MNP' -> MNP
    Falls back to None if nothing readable.
    """
    with Image.open(frame_png) as im:
        w, h = im.size
        lx, ty, rx, by = _scale_box_to_frame(TOPRIGHT_BOX_REF, w, h)
        crop = _crop_rect_safe(im, lx, ty, rx, by)

    if viz_dir:
        try:
            (viz_dir / "topright_raw.png").parent.mkdir(parents=True, exist_ok=True)
            crop.save(viz_dir / "topright_raw.png")
        except Exception:
            pass

    letters = _ocr_letters_strict(ocr, crop, viz_dir=viz_dir, stamp="TR")
    if not letters:
        return None

    up = letters.upper()
    if "SNP" in up:
        return "SNP"
    if "TNP" in up:
        return "TNP"
    if "MNP" in up:
        return "MNP"
    if "S" in up:
        return "SNP"
    if "T" in up:
        return "TNP"
    if "M" in up:
        return "MNP"
    return None

# ------------------------
# Public API with aggregation across frames
# ------------------------
def extract_scoreboard_from_video_paddle(
    video_path: str,
    *,
    t: float = 0.10,
    attempts: int = 3,
    viz: bool = False,
    profile_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Sample several frames; OCR each ROI; aggregate across frames:
      - auto-detect scoreboard preset from top-right label (S/T/M => SNP/TNP/MNP)
      - allow explicit profile_key override from API
      - crop scoreboard based on preset window
      - team names: mode of normalized strings
      - scores: mode of per-side integers
      - clock/quarter: mode of normalized strings
    """
    run_id = f"ocr_{uuid.uuid4().hex[:8]}"
    run_dir = Path("data/tmp/ocr") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    ts_list: List[float] = [max(0.0, t + 0.10 * k) for k in range(max(1, attempts))]
    ocr = _ocr_init()

    ref_size: Optional[Tuple[int, int]] = None
    chosen_box_full: Optional[Tuple[int, int, int, int]] = None
    chosen_preset: str = "DEFAULT"

    # Collect across frames
    away_team_reads: List[Optional[str]] = []
    home_team_reads: List[Optional[str]] = []
    away_score_reads: List[Optional[int]] = []
    home_score_reads: List[Optional[int]] = []
    clock_reads: List[Optional[str]] = []
    quarter_reads: List[Optional[str]] = []

    for i, ts in enumerate(ts_list):
        frame_png = run_dir / f"frame_t{ts:.2f}.png"
        _grab_frame(video_path, ts, frame_png)

        # Decide preset once (override has priority)
        if chosen_box_full is None:
            if profile_key and profile_key.upper() in ROI_PRESETS:
                chosen_preset = profile_key.upper()
            else:
                detected = _detect_preset_from_topright(frame_png, ocr, viz_dir=(run_dir if viz else None))
                if detected and detected in ROI_PRESETS:
                    chosen_preset = detected
            chosen_box_full = SCOREBAR_BOX_PRESETS.get(chosen_preset, SCOREBAR_BOX_DEFAULT)

        # viz outline for the chosen preset window
        if viz and i == 0 and chosen_box_full:
            _draw_frame_scorebar_outline(frame_png, chosen_box_full, run_dir / f"frame_with_scorebar_t{ts:.2f}.png")

        # crop scoreboard based on detected/selected preset
        score_png = run_dir / f"scorebar_t{ts:.2f}.png"
        score_png, (sbw, sbh) = _crop_scorebar(frame_png, chosen_box_full, score_png)  # type: ignore[arg-type]
        if ref_size is None:
            ref_size = (sbw, sbh)

        # ROI map for current preset
        roi_map = ROI_PRESETS.get(chosen_preset, ROI_PRESETS["DEFAULT"])

        if viz and i == 0:
            _viz_rois_on_scorebar(score_png, ref_size, roi_map, run_dir / f"scorebar_rois_t{ts:.2f}.png")

        with Image.open(score_png) as sb:
            def grab_txt(key: str) -> str:
                if key not in roi_map:
                    return ""
                crop = _crop_px_from_ref(sb, roi_map[key], ref_size)  # type: ignore[arg-type]
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

    # Aggregate across frames
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
        "_preset": chosen_preset,
    }

    # If nothing at all, return stub
    if not any((away_team, home_team, a_mode is not None, h_mode is not None, clk, qtr)):
        return {"used_stub": True, "_preset": chosen_preset}

    # For debug visibility
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
