# backend/services/ocr_paddle.py
from __future__ import annotations

from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
import uuid, subprocess, shlex, os, re, time, logging
from dataclasses import dataclass

import numpy as np
from paddleocr import PaddleOCR

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None

from PIL import Image, ImageDraw, ImageOps

# ------------------------
# Logging
# ------------------------
log = logging.getLogger(__name__)
if not log.handlers:
    logging.basicConfig(level=logging.INFO)

# ------------------------
# Constants (reference full frame)
# ------------------------
REF_FRAME_W, REF_FRAME_H = 2047, 1155

# Original/default bottom banner crop (fallback if preset not detected)
SCOREBAR_BOX_DEFAULT = (0, 1080, 2047, 1155)

# Top-right label region in FULL FRAME (your estimate)
TOPRIGHT_BOX_REF = (1850, 30, 1950, 90)

# Scoreboard crop windows per preset (in FULL FRAME coords, exclusive x2,y2 semantics)
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
    # Default
    "DEFAULT": (0, 1080, 2047, 1155),
}

# ------------------------
# Canonicalization policy & ROI presets (PLACEHOLDERS for now)
# ------------------------
# Canonical target sizes (W, H) after resize—stretch-to-fit (no letterbox)
CANON_SIZE: Dict[str, Tuple[int, int]] = {
    "MNP":     (2000, 72),
    "TNP":     (800, 128),
    "SNP":     (920, 100),
    "DEFAULT": (1445, 72),
}

# ROI presets tuned *for canonical sizes* (placeholders; replace once you verify overlays)
# Coords are pixel boxes in the canonical image space (x1, y1, x2, y2) with x2,y2 exclusive.
ROI_PRESETS_CANON: Dict[str, Dict[str, Tuple[int, int, int, int]]] = {
    "MNP": {
        "away_team":  (550,  15,  626,  60),
        "away_score": (650,   0,  775,  70),
        "home_team":  (975,  15,  1050, 60),
        "home_score": (825,   0,  950,  70),
        "quarter":    (1350, 15,  1410, 60),
        "clock":      (1410, 15,  1490, 60),
    },
    "TNP": {
        "away_team":  (210,  25,  275,  75),
        "away_score": (275,  10,  365,  75),
        "home_team":  (525,  25,  600,  75),
        "home_score": (440,  10,  525,  75),
        "quarter":    (300,  75,  350, 120),
        "clock":      (350,  75,  450, 120),
    },
    "SNP": {
        "away_team":  (130,  25,  205,  85),
        "away_score": (210,  25,  300,  85),
        "home_team":  (665,  25,  735,  85),
        "home_score": (570,  25,  660,  85),
        "quarter":    (340,  55,  375,  90),
        "clock":      (375,  55,  450,  90),
    },
    "DEFAULT": {
        "away_team":  (375,   10,  465,  45),
        "home_team":  (670,   10,  755,  45),
        "away_score": (490,   10,  560,  55),
        "home_score": (575,   10,  645,  55),
        "quarter":    (960,   10, 1010,  45),
        "clock":      (1010,  10, 1090,  45),
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

def _crop_rect_safe(pil: Image.Image, x0: int, y0: int, x1: int, y1: int) -> Image.Image:
    # normalize
    if x1 < x0: x0, x1 = x1, x0
    if y1 < y0: y0, y1 = y1, y0
    # clamp
    w, h = pil.size
    x0 = max(0, min(x0, w - 1))
    y0 = max(0, min(y0, h - 1))
    x1 = max(x0 + 1, min(x1, w))
    y1 = max(y0 + 1, min(y1, h))
    return pil.crop((x0, y0, x1, y1))

def _prep_gray_bw(pil: Image.Image, thresh: int = 160, invert: bool = False) -> Image.Image:
    g = pil.convert("L")
    table = [0] * (thresh + 1) + [255] * (256 - (thresh + 1))
    bw = g.point(table, mode="L")
    if invert:
        bw = ImageOps.invert(bw)
    return bw

# ------------------------
# Visualization helpers (frame & canonical overlays)
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
    # Legacy visualizer (variable-size scorebar) – not used by canonical flow, kept for reference.
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
# Deterministic Scoreboard Preprocessor (canonicalized ROIs)
# ------------------------
def _ensure_np_rgb(img: Any) -> np.ndarray:
    if isinstance(img, Image.Image):
        return np.array(img.convert("RGB"))
    arr = img
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    return arr

def _clamp_box_exclusive(box: Tuple[int, int, int, int], W: int, H: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    x1 = max(0, min(x1, W))
    y1 = max(0, min(y1, H))
    x2 = max(0, min(x2, W))
    y2 = max(0, min(y2, H))
    if x2 < x1: x1, x2 = x2, x1
    if y2 < y1: y1, y2 = y2, y1
    if x2 == x1 and x2 < W: x2 = x1 + 1
    if y2 == y1 and y2 < H: y2 = y1 + 1
    if x2 == x1 and x2 == W and x1 > 0: x1 -= 1
    if y2 == y1 and y2 == H and y1 > 0: y1 -= 1
    return (x1, y1, x2, y2)

def _scale_fullframe_box_to_actual(full_box_ref: Tuple[int, int, int, int], actual_w: int, actual_h: int) -> Tuple[int, int, int, int]:
    # Scale from REF_FRAME (2047x1155) into the actual frame; keep exclusive semantics with rounding.
    lx, ty, rx, by = full_box_ref
    sx = actual_w / float(REF_FRAME_W)
    sy = actual_h / float(REF_FRAME_H)
    X1 = int(round(lx * sx))
    Y1 = int(round(ty * sy))
    X2 = int(round(rx * sx))
    Y2 = int(round(by * sy))
    return (X1, Y1, X2, Y2)

def crop_scorebar(full_frame: Any, skin: str) -> np.ndarray:
    """
    Input: full_frame (PIL.Image or np.ndarray RGB), skin in {MNP,TNP,SNP,DEFAULT}
    Output: np.ndarray RGB crop (variable size) using SCOREBAR_BOX_PRESETS[skin],
            exclusive x2,y2 semantics, clamped to bounds.
    """
    t0 = time.perf_counter()
    skin = (skin or "DEFAULT").upper()
    if skin not in SCOREBAR_BOX_PRESETS:
        skin = "DEFAULT"

    img = _ensure_np_rgb(full_frame)
    H, W, _ = img.shape

    # 1) scale the configured full-frame box to this actual frame
    box_ref = SCOREBAR_BOX_PRESETS[skin]
    x1, y1, x2, y2 = _scale_fullframe_box_to_actual(box_ref, W, H)

    # 2) clamp (exclusive)
    x1, y1, x2, y2 = _clamp_box_exclusive((x1, y1, x2, y2), W, H)

    # 3) crop via numpy slicing (exclusive)
    crop = img[y1:y2, x1:x2, :].copy()

    log.info("[crop_scorebar][%s] box=%s -> crop %dx%d in %.3f ms",
             skin, (x1, y1, x2, y2), crop.shape[1], crop.shape[0],
             (time.perf_counter() - t0) * 1000.0)
    return crop

def to_canonical(scorebar: Any, skin: str) -> np.ndarray:
    """
    Stretch-to-fit resize to canonical size for the skin.
    Uses INTER_AREA (OpenCV) if available; otherwise PIL BICUBIC.
    """
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
        # ✅ Pillow ≥10 compatibility
        canon = np.array(
            pil.resize((target_w, target_h), resample=Image.Resampling.BICUBIC)
        )

    log.info("[to_canonical][%s] -> %dx%d in %.3f ms",
             skin, target_w, target_h, (time.perf_counter() - t0) * 1000.0)
    return canon

def extract_rois(
    bar_canon: Any,
    skin: str,
    roi_names: Optional[List[str]] = None,
    pad: int = 2
) -> Dict[str, np.ndarray]:
    """
    Convert configured pixel ROIs (exclusive x2,y2) at canonical size into
    np.ndarray patches. Applies ±pad px, clamps to bounds.
    """
    t0 = time.perf_counter()
    skin = (skin or "DEFAULT").upper()
    roi_map = ROI_PRESETS_CANON.get(skin) or ROI_PRESETS_CANON["DEFAULT"]

    img = _ensure_np_rgb(bar_canon)
    H, W, _ = img.shape

    names = roi_names if roi_names else list(roi_map.keys())
    patches: Dict[str, np.ndarray] = {}

    for name in names:
        if name not in roi_map:
            continue
        x1, y1, x2, y2 = roi_map[name]
        # padding while keeping exclusive semantics
        x1p = x1 - pad
        y1p = y1 - pad
        x2p = x2 + pad
        y2p = y2 + pad

        x1p, y1p, x2p, y2p = _clamp_box_exclusive((x1p, y1p, x2p, y2p), W, H)
        patch = img[y1p:y2p, x1p:x2p, :].copy()
        patches[name] = patch

    log.info("[extract_rois][%s] %d patches in %.3f ms",
             skin, len(patches), (time.perf_counter() - t0) * 1000.0)
    return patches

def draw_roi_overlay(bar_canon: Any, skin: str, out_path: str) -> str:
    """
    Draw labeled ROI rectangles over the canonical bar and save PNG.
    """
    skin = (skin or "DEFAULT").upper()
    roi_map = ROI_PRESETS_CANON.get(skin) or ROI_PRESETS_CANON["DEFAULT"]

    img = _ensure_np_rgb(bar_canon)
    pil = Image.fromarray(img)
    d = ImageDraw.Draw(pil)

    color_map = {
        "away_team":  "orange",
        "home_team":  "orange",
        "away_score": "lime",
        "home_score": "lime",
        "quarter":    "deepskyblue",
        "clock":      "deepskyblue",
    }

    for key, (x1, y1, x2, y2) in roi_map.items():
        # Exclusive coords: for drawing, subtract 1px on right/bottom to visualize the exact box.
        x2d = max(x1, x2 - 1)
        y2d = max(y1, y2 - 1)
        col = color_map.get(key, "yellow")
        d.rectangle((x1, y1, x2d, y2d), outline=col, width=3)
        d.text((x1 + 4, max(0, y1 - 14)), key, fill=col)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    pil.save(out_path)
    return out_path

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
    x = re.sub(r"[^0-9A-Z]", "", s.upper())
    m = re.fullmatch(r"[1-4]", re.sub(r"[^0-9]", "", x))
    if m:
        return f"Q{m.group(0)}"
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
# Public API with aggregation across frames (uses canonical pipeline)
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
      - crop scoreboard based on preset full-frame window
      - resize to preset's canonical size (deterministic W×H)
      - extract ROIs by pixel boxes (±2px padding), then OCR
      - aggregate via mode across frames
    """
    run_id = f"ocr_{uuid.uuid4().hex[:8]}"
    run_dir = Path("data/tmp/ocr") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    ts_list: List[float] = [max(0.0, t + 0.10 * k) for k in range(max(1, attempts))]
    ocr = _ocr_init()

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
        if i == 0:
            if profile_key and profile_key.upper() in ROI_PRESETS_CANON:
                chosen_preset = profile_key.upper()
            else:
                detected = _detect_preset_from_topright(frame_png, ocr, viz_dir=(run_dir if viz else None))
                if detected and detected in ROI_PRESETS_CANON:
                    chosen_preset = detected
                else:
                    chosen_preset = "DEFAULT"

        # Visualize the chosen preset crop window on the full frame (debug)
        if viz and i == 0:
            full_box = SCOREBAR_BOX_PRESETS.get(chosen_preset, SCOREBAR_BOX_DEFAULT)
            _draw_frame_scorebar_outline(frame_png, full_box, run_dir / f"frame_with_scorebar_t{ts:.2f}.png")

        # --- Canonical pipeline ---
        with Image.open(frame_png) as full_pil:
            sb_crop_np = crop_scorebar(full_pil, chosen_preset)
        sb_canon = to_canonical(sb_crop_np, chosen_preset)

        # Save canonical bar & overlay once (or each frame if you like)
        if viz:
            Image.fromarray(sb_canon).save(run_dir / f"scorebar_canonical_{chosen_preset}_t{ts:.2f}.png")
            draw_roi_overlay(sb_canon, chosen_preset, str(run_dir / f"scorebar_overlay_{chosen_preset}_t{ts:.2f}.png"))

        # Extract ROI patches (±2px padding)
        patches = extract_rois(sb_canon, chosen_preset, pad=2)

        # OCR helper for ROI dict
        def ocr_roi(name: str) -> str:
            patch = patches.get(name)
            if patch is None:
                return ""
            pil_patch = Image.fromarray(patch)
            patch_bw = _prep_gray_bw(pil_patch, thresh=160, invert=False)
            return _ocr_text(ocr, patch_bw)

        a_team_raw = ocr_roi("away_team")
        h_team_raw = ocr_roi("home_team")
        a_sc_raw   = ocr_roi("away_score")
        h_sc_raw   = ocr_roi("home_score")
        q_raw      = ocr_roi("quarter")
        c_raw      = ocr_roi("clock")

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
