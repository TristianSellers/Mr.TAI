# backend/services/ocr_paddle.py
from __future__ import annotations

from typing import Dict, Any, Optional, Tuple, List, Sequence
from pathlib import Path
import uuid, subprocess, shlex, re, time, logging

import numpy as np
from paddleocr import PaddleOCR

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None

from PIL import Image, ImageDraw, ImageOps, ImageFilter

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
    "MNP": (0, 1080, 2000, 1150),
    "TNP": (630, 1020, 1420, 1145),
    "SNP": (590, 1040, 1510, 1140),
    "DEFAULT": (0, 1080, 2047, 1155),
}

# ------------------------
# Canonicalization policy & ROI presets
# ------------------------
CANON_SIZE: Dict[str, Tuple[int, int]] = {
    "MNP":     (2000, 72),
    "TNP":     (800, 128),
    "SNP":     (920, 100),
    "DEFAULT": (1445, 72),
}

# FINAL ROI presets (x1,y1,x2,y2 exclusive) tuned for canonical sizes
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
        "away_score": (275,   5,  365,  75),  # tweaked y1
        "home_team":  (525,  25,  600,  75),
        "home_score": (440,   5,  525,  75),  # tweaked y1
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
    if x1 < x0: x0, x1 = x1, x0
    if y1 < y0: y0, y1 = y1, y0
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
# Visualization helpers
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
# Canonicalized preprocessor
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
    lx, ty, rx, by = full_box_ref
    sx = actual_w / float(REF_FRAME_W)
    sy = actual_h / float(REF_FRAME_H)
    X1 = int(round(lx * sx))
    Y1 = int(round(ty * sy))
    X2 = int(round(rx * sx))
    Y2 = int(round(by * sy))
    return (X1, Y1, X2, Y2)

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

    log.info("[crop_scorebar][%s] box=%s -> crop %dx%d in %.3f ms",
             skin, (x1, y1, x2, y2), crop.shape[1], crop.shape[0],
             (time.perf_counter() - t0) * 1000.0)
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

    log.info("[to_canonical][%s] -> %dx%d in %.3f ms",
             skin, target_w, target_h, (time.perf_counter() - t0) * 1000.0)
    return canon

def extract_rois(
    bar_canon: Any,
    skin: str,
    roi_names: Optional[List[str]] = None,
    pad: int = 2
) -> Dict[str, np.ndarray]:
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
        x1p = x1 - pad; y1p = y1 - pad
        x2p = x2 + pad; y2p = y2 + pad
        x1p, y1p, x2p, y2p = _clamp_box_exclusive((x1p, y1p, x2p, y2p), W, H)
        patch = img[y1p:y2p, x1p:x2p, :].copy()
        patches[name] = patch

    log.info("[extract_rois][%s] %d patches in %.3f ms",
             skin, len(patches), (time.perf_counter() - t0) * 1000.0)
    return patches

def draw_roi_overlay(bar_canon: Any, skin: str, out_path: str) -> str:
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
# Digit OCR (ensemble) + confidences
# ------------------------
def _only_digits(s: str) -> str:
    return "".join(ch for ch in (s or "") if ch.isdigit())

def _clahe(gray: np.ndarray) -> np.ndarray:
    if cv2 is None:
        return gray
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)

def _ocr_digits_ensemble(
    ocr: PaddleOCR,
    pil_img: Image.Image,
    viz_dir: Optional[Path] = None,
    tag: str = "",
    extra_images: Optional[List[Image.Image]] = None
) -> str:
    """
    Multi-scale, multi-threshold (+/- invert), plus a CLAHE branch.
    Chooses longest digit string; tie-breaker prefers non-inverted.
    extra_images: optional preprocessed variants to include.
    """
    cands: List[Tuple[int, str]] = []
    idx = 0
    bases: List[Image.Image] = []
    for scale in (2, 3, 4):
        bases.append(pil_img.resize((max(1, pil_img.width * scale), max(1, pil_img.height * scale))))
    if cv2 is not None:
        im3 = pil_img.resize((max(1, pil_img.width * 3), max(1, pil_img.height * 3)))
        g3 = np.array(im3.convert("L"))
        g3c = _clahe(g3)
        bases.append(Image.fromarray(g3c).convert("RGB"))
    if extra_images:
        bases.extend(extra_images)

    for im in bases:
        for thr in (110, 130, 150, 170, 190, 210, 230):
            for inv in (False, True):
                bw = _prep_gray_bw(im, thresh=thr, invert=inv)
                txt = _ocr_text(ocr, bw)
                dg = _only_digits(txt)
                if dg:
                    score = (len(dg) * 10) + (0 if inv else 1)
                    cands.append((score, dg))
                if viz_dir:
                    try:
                        bw.save((viz_dir / f"score_ens_{tag}_{idx}.png"))
                    except Exception:
                        pass
                idx += 1
    if cands:
        cands.sort(key=lambda t: t[0], reverse=True)
        return cands[0][1]
    return ""

# ---------- Template fallback helpers (with cache + confidence) ----------
_TPL_CACHE: Dict[str, List[Tuple[str, np.ndarray]]] = {}

def _load_digit_templates(skin: str) -> List[Tuple[str, np.ndarray]]:
    if skin in _TPL_CACHE:
        return _TPL_CACHE[skin]
    base = Path("data/ocr_templates") / skin.upper()
    tpls: List[Tuple[str, np.ndarray]] = []
    if base.exists():
        for d in map(str, range(10)):
            p = base / f"{d}.png"
            if p.exists():
                im = Image.open(p).convert("L")
                arr = np.array(im)
                tpls.append((d, arr))
    _TPL_CACHE[skin] = tpls
    return tpls

def _prep_binary_variants_gray(gray: np.ndarray) -> List[np.ndarray]:
    out: List[np.ndarray] = []
    if cv2 is None:
        pil = Image.fromarray(gray)
        for thr in (120, 150, 180):
            bw = np.array(_prep_gray_bw(pil, thresh=thr, invert=False))
            out.append(bw); out.append(255 - bw)
        return out
    gray2 = _clahe(gray)
    for base in (gray, gray2):
        _, otsu = cv2.threshold(base, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        out += [otsu, cv2.bitwise_not(otsu)]
        ada = cv2.adaptiveThreshold(base, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 11, 2)
        out += [ada, cv2.bitwise_not(ada)]
    return out

def _morph_tidy(bw: np.ndarray) -> np.ndarray:
    if cv2 is None:
        return bw
    k1 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
    k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    closed = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k1, iterations=1)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, k2, iterations=1)
    return opened

def _norm_xcorr(a: np.ndarray, b: np.ndarray) -> float:
    A = a.astype(np.float32); B = b.astype(np.float32)
    A = (A - A.mean()) / (A.std() + 1e-6)
    B = (B - B.mean()) / (B.std() + 1e-6)
    return float((A * B).mean())

def _resize_like(img: np.ndarray, target_hw: Sequence[int]) -> np.ndarray:
    th, tw = int(target_hw[0]), int(target_hw[1])
    if img.shape[0] == th and img.shape[1] == tw:
        return img
    if cv2 is None:
        return np.array(Image.fromarray(img).resize((tw, th), resample=Image.Resampling.BICUBIC))
    return cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)

def _best_split_index(bw: np.ndarray) -> int:
    """
    Find the vertical seam (between 30%..70% of width) with minimal dark pixel count;
    helps separate two digits like '10'/'24'.
    """
    H, W = bw.shape
    fg = (bw < 128).astype(np.uint8)
    col_sums = fg.sum(axis=0)
    lo = int(0.30 * W)
    hi = int(0.70 * W)
    if hi <= lo:
        return W // 2
    segment = col_sums[lo:hi]
    idx = int(np.argmin(segment))
    return lo + idx

def _template_match_digits_with_conf(pil_img: Image.Image, skin: str, max_digits: int = 2) -> Optional[Tuple[str, float]]:
    """
    NCC template match; returns (digits, confidence in [0,1]).
    For 2 digits, searches best seam instead of fixed 50/50 split.
    """
    tpls = _load_digit_templates(skin)
    if not tpls:
        return None

    gray = np.array(pil_img.convert("L"))
    bins = [_morph_tidy(bw) for bw in _prep_binary_variants_gray(gray)]

    def score_digit(glyph: np.ndarray) -> Tuple[str, float]:
        best_s, best_d = -1.0, "0"
        for d, tpl in tpls:
            g = _resize_like(glyph, tpl.shape[:2])
            s = _norm_xcorr(g, tpl)
            s = max(s, _norm_xcorr(255 - g, tpl))
            if s > best_s:
                best_s, best_d = s, d
        return best_d, float(max(0.0, min(1.0, (best_s + 1.0) * 0.5)))

    candidates: List[Tuple[float, str]] = []

    for bw in bins:
        d1, s1 = score_digit(bw)
        candidates.append((s1, d1))
        if max_digits >= 2:
            mid = _best_split_index(bw)
            for shift in (-6, -3, 0, 3, 6):
                m = int(np.clip(mid + shift, 1, bw.shape[1] - 1))
                left, right = bw[:, :m], bw[:, m:]
                if left.size == 0 or right.size == 0:
                    continue
                dL, sL = score_digit(left)
                dR, sR = score_digit(right)
                candidates.append((min(sL, sR), dL + dR))

    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    best_s, best_d = candidates[0]
    return (best_d, best_s)

# ---------- Polarity/illumination helpers ----------
def _illumination_normalize(gray: np.ndarray) -> np.ndarray:
    """Divide by blurred field to flatten shading; clip to 0..255."""
    if cv2 is not None:
        blur = cv2.GaussianBlur(gray, (11, 11), 3)
        norm = (gray.astype(np.float32) / (blur.astype(np.float32) + 1e-6)) * 128.0
        norm = np.clip(norm, 0, 255).astype(np.uint8)
        return norm
    # PIL fallback
    pil = Image.fromarray(gray)
    blur = np.array(pil.filter(ImageFilter.GaussianBlur(radius=3)))
    norm = (gray.astype(np.float32) / (blur.astype(np.float32) + 1e-6)) * 128.0
    return np.clip(norm, 0, 255).astype(np.uint8)

def _edge_energy(bw: np.ndarray) -> float:
    if cv2 is None:
        return float(np.std(bw))
    e = cv2.Canny(bw, 50, 150)
    return float(e.sum()) / 255.0

def _central_occupancy(bw: np.ndarray) -> float:
    H, W = bw.shape
    x0, x1 = int(0.25*W), int(0.75*W)
    y0, y1 = int(0.25*H), int(0.75*H)
    roi = bw[y0:y1, x0:x1]
    if roi.size == 0:
        return 0.0
    fg = (roi < 128).sum()
    return float(fg) / float(roi.size)

def _choose_polarity(gray: np.ndarray) -> np.ndarray:
    """Return the binary image (normal or inverted) with higher edge+occupancy score."""
    variants = _prep_binary_variants_gray(gray)
    best_score = -1.0
    best = variants[0]
    for v in variants:
        bw = v if v.mean() > 127 else (255 - v)
        score = 0.7 * _edge_energy(bw) + 0.3 * _central_occupancy(bw)
        if score > best_score:
            best_score = score
            best = bw
    return best

def _best_subwindow_by_edge(patch_rgb: np.ndarray, max_shift: int = 2) -> Image.Image:
    """Try small shifts, keep window with highest edge energy, return resized back to original size."""
    H, W, _ = patch_rgb.shape
    best_score = -1.0
    best_crop = patch_rgb
    for dy in (-max_shift, 0, max_shift):
        for dx in (-max_shift, 0, max_shift):
            y0 = max(0, 0 + dy); y1 = min(H, H + dy)
            x0 = max(0, 0 + dx); x1 = min(W, W + dx)
            if y1 - y0 < H - abs(dy) or x1 - x0 < W - abs(dx):
                # ensure we keep inside bounds
                y0 = max(0, dy if dy > 0 else 0)
                x0 = max(0, dx if dx > 0 else 0)
                y1 = min(H, H + (dy if dy < 0 else 0))
                x1 = min(W, W + (dx if dx < 0 else 0))
            crop = patch_rgb[y0:y1, x0:x1]
            if crop.size == 0:
                continue
            if cv2 is not None:
                score = _edge_energy(cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY))
            else:
                score = _edge_energy(np.array(Image.fromarray(crop).convert("L")))
            if score > best_score:
                best_score = score
                best_crop = crop
    # resize back to original size
    if cv2 is not None:
        resized = cv2.resize(best_crop, (W, H), interpolation=cv2.INTER_AREA)
        return Image.fromarray(resized)
    return Image.fromarray(best_crop).resize((W, H), resample=Image.Resampling.BICUBIC)

# ---------- Binary selection + geometry cues ----------
def _best_binary(gray: np.ndarray) -> np.ndarray:
    variants: List[np.ndarray] = _prep_binary_variants_gray(gray)
    if not variants:
        pil = Image.fromarray(gray)
        bw = np.array(_prep_gray_bw(pil, thresh=160, invert=False))
        return bw
    if cv2 is None:
        scores = [v.std() for v in variants]
        best = variants[int(np.argmax(scores))]
        return best if best.mean() > 127 else (255 - best)
    best_score = -1.0
    best_bw = variants[0]
    for v in variants:
        bw = v if v.mean() > 127 else (255 - v)
        gx = cv2.Sobel(bw, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(bw, cv2.CV_32F, 0, 1, ksize=3)
        score = float((np.abs(gx) + np.abs(gy)).sum())
        if score > best_score:
            best_score = score
            best_bw = bw
    return best_bw

def _detect_seven_confidence(pil_img: Image.Image) -> float:
    if cv2 is None:
        return 0.0
    gray = np.array(pil_img.convert("L"))
    bw = _best_binary(gray)
    edges = cv2.Canny(bw, 50, 150, apertureSize=3)
    h, w = edges.shape
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=15,
                            minLineLength=max(6, w//5), maxLineGap=3)
    if lines is None:
        return 0.0
    top_strength = 0.0
    diag_strength = 0.0
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        dx, dy = x2 - x1, y2 - y1
        length = max(1.0, float(np.hypot(dx, dy)))
        slope = abs(dy) / length
        angle = np.degrees(np.arctan2(abs(dy), abs(dx)))
        if slope < 0.2 and min(y1, y2) < int(0.4 * h):
            top_strength = max(top_strength, min(1.0, length / (w * 0.8)))
        if 25 <= angle <= 70:
            diag_strength = max(diag_strength, min(1.0, length / (w * 0.8)))
    conf = 0.6 * top_strength + 0.4 * diag_strength
    return float(max(0.0, min(1.0, conf)))

def _detect_zero_confidence(pil_img: Image.Image) -> float:
    if cv2 is None:
        return 0.0
    im = np.array(pil_img.convert("L"))
    best_conf = 0.0
    for inv in (False, True):
        im2 = cv2.bitwise_not(im) if inv else im
        _, bw = cv2.threshold(im2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        h, w = bw.shape[:2]
        if h < 6 or w < 6:
            continue
        contours, hierarchy = cv2.findContours(bw, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is None or len(contours) == 0:
            continue
        best_idx = None
        best_area = 0.0
        for i, cnt in enumerate(contours):
            if hierarchy[0][i][3] != -1:
                continue
            area = cv2.contourArea(cnt)
            if area > best_area:
                best_area = area
                best_idx = i
        if best_idx is None or best_area < 10:
            continue
        has_child = any(hierarchy[0][j][3] == best_idx for j in range(len(contours)))
        if not has_child:
            continue
        x, y, ww, hh = cv2.boundingRect(contours[best_idx])
        aspect = ww / float(hh + 1e-6)
        coverage = best_area / float(ww * hh + 1e-6)
        asp_ok = max(0.0, 1.0 - abs(aspect - 1.1) / 0.8)
        cov_ok = 1.0 - min(abs(coverage - 0.55) / 0.30, 1.0)
        conf = 0.5 * asp_ok + 0.5 * cov_ok
        best_conf = max(best_conf, conf)
    return float(best_conf)

def _zero_center_contrast_conf(pil_img: Image.Image) -> float:
    gray = np.array(pil_img.convert("L"))
    h, w = gray.shape
    if h < 6 or w < 6:
        return 0.0
    g = _clahe(gray) if cv2 is not None else gray
    if cv2 is not None:
        edges = cv2.Canny(g, 50, 150)
        ring_density = float((edges > 0).sum()) / float(max(1, h * w))
    else:
        ring_density = float(np.std(g)) / 255.0
    cx0, cx1 = int(0.35 * w), int(0.65 * w)
    cy0, cy1 = int(0.35 * h), int(0.65 * h)
    center = g[cy0:cy1, cx0:cx1]
    border_mask = np.ones_like(g, dtype=bool)
    border_mask[cy0:cy1, cx0:cx1] = False
    border = g[border_mask]
    if center.size == 0 or border.size == 0:
        return 0.0
    delta = float(center.mean() - border.mean()) / 255.0
    dens_score = min(1.0, ring_density / 0.25)
    bright_score = max(0.0, min(1.0, (delta + 0.2) / 0.5))
    return 0.6 * dens_score + 0.4 * bright_score

def _count_connected_components(bw: np.ndarray) -> int:
    """Count foreground components in a binary image where foreground is dark (0)."""
    if cv2 is None:
        cols = (bw < 128).sum(axis=0) > 0
        return int(max(1, np.diff(cols.astype(np.int32)).clip(min=0).sum()))
    fg = (bw < 128).astype(np.uint8)
    num_labels, _ = cv2.connectedComponents(fg)
    return max(0, num_labels - 1)

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
# Preset detection (top-right)
# ------------------------
def _detect_preset_from_topright(frame_png: Path, ocr: PaddleOCR, viz_dir: Optional[Path]) -> Optional[str]:
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
    if "SNP" in up: return "SNP"
    if "TNP" in up: return "TNP"
    if "MNP" in up: return "MNP"
    if "S" in up: return "SNP"
    if "T" in up: return "TNP"
    if "M" in up: return "MNP"
    return None

# ------------------------
# Public API
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
    Canonical crop → resize → pixel-ROI → OCR:
      * Individual team scores (raw + parsed).
      * Quarter returned raw.
      * Score reading uses OCR ensemble + zero/center/seven/template confidences with safer thresholds.
      * Template split is learned seam (better 2-digit like '10', '24').
      * Targeted rescue path only for TNP-away (darker ROI).
    """
    run_id = f"ocr_{uuid.uuid4().hex[:8]}"
    run_dir = Path("data/tmp/ocr") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    ts_list: List[float] = [max(0.0, t + 0.10 * k) for k in range(max(1, attempts))]
    ocr = _ocr_init()

    chosen_preset: str = "DEFAULT"

    away_team_reads: List[Optional[str]] = []
    home_team_reads: List[Optional[str]] = []
    away_score_text_reads: List[Optional[str]] = []
    home_score_text_reads: List[Optional[str]] = []
    away_score_int_reads: List[Optional[int]] = []
    home_score_int_reads: List[Optional[int]] = []
    clock_reads: List[Optional[str]] = []
    quarter_raw_reads: List[Optional[str]] = []

    # --- Rescue helper (TNP-away only) ---
    def _rescue_read_digits_for_tnp_away(patch_np: np.ndarray) -> Tuple[Optional[str], Optional[int], bool]:
        """
        Returns (text, int, zero_hint). zero_hint=True if geometric zero is moderately likely.
        Runs only when primary pass returned empty text for TNP-away.
        """
        # Illumination normalization
        gray = np.array(Image.fromarray(patch_np).convert("L"))
        norm = _illumination_normalize(gray)
        norm = _clahe(norm)

        # Polarity pick
        best_bw = _choose_polarity(norm)

        # Compose extra candidates for the ensemble (RGB)
        extra_imgs: List[Image.Image] = []
        extra_imgs.append(Image.fromarray(cv2.cvtColor(norm, cv2.COLOR_GRAY2RGB) if cv2 is not None else np.stack([norm]*3, -1)))
        extra_imgs.append(Image.fromarray(cv2.cvtColor(best_bw, cv2.COLOR_GRAY2RGB) if cv2 is not None else np.stack([best_bw]*3, -1)))

        # Micro-ROI nudge: try tiny subwindow with more edges
        nudged = _best_subwindow_by_edge(patch_np, max_shift=2)
        extra_imgs.append(nudged)

        # A slightly heavier single-path upscale for this ROI
        big = Image.fromarray(patch_np).resize(
            (patch_np.shape[1]*5, patch_np.shape[0]*5),
            resample=Image.Resampling.BICUBIC
        )
        extra_imgs.append(big)

        # Ensemble with extras
        dg = _ocr_digits_ensemble(ocr, Image.fromarray(patch_np), viz_dir=(run_dir if viz else None), tag="tnp_away_rescue", extra_images=extra_imgs)
        dg = (dg or "").strip()

        # Template match with slightly relaxed 0 acceptance coupled to geometry
        tpl = _template_match_digits_with_conf(Image.fromarray(patch_np), "TNP", max_digits=2)
        tpl_guess, tpl_conf = (tpl[0], tpl[1]) if tpl else (None, 0.0)

        # Zero geometry hints
        z_conf  = _detect_zero_confidence(Image.fromarray(patch_np))
        zc2     = _zero_center_contrast_conf(Image.fromarray(patch_np))
        zero_hint = (z_conf >= 0.58) or ((z_conf >= 0.52) and (zc2 >= 0.58))

        # If still empty, consider template "0" if geometry agrees
        if not dg:
            if tpl_guess == "0" and tpl_conf >= 0.46 and zero_hint:
                dg = "0"

        # If still empty and zero_hint strong, set int 0 but leave text None
        if not dg and zero_hint:
            return None, 0, True

        return (dg or None, (int(dg) if dg and dg.isdigit() else None), zero_hint)

    for i, ts in enumerate(ts_list):
        frame_png = run_dir / f"frame_t{ts:.2f}.png"
        _grab_frame(video_path, ts, frame_png)

        if i == 0:
            if profile_key and profile_key.upper() in ROI_PRESETS_CANON:
                chosen_preset = profile_key.upper()
            else:
                detected = _detect_preset_from_topright(frame_png, ocr, viz_dir=(run_dir if viz else None))
                chosen_preset = detected if (detected and detected in ROI_PRESETS_CANON) else "DEFAULT"

        if viz and i == 0:
            full_box = SCOREBAR_BOX_PRESETS.get(chosen_preset, SCOREBAR_BOX_DEFAULT)
            _draw_frame_scorebar_outline(frame_png, full_box, run_dir / f"frame_with_scorebar_t{ts:.2f}.png")

        with Image.open(frame_png) as full_pil:
            sb_crop_np = crop_scorebar(full_pil, chosen_preset)
        sb_canon = to_canonical(sb_crop_np, chosen_preset)

        if viz:
            Image.fromarray(sb_canon).save(run_dir / f"scorebar_canonical_{chosen_preset}_t{ts:.2f}.png")
            draw_roi_overlay(sb_canon, chosen_preset, str(run_dir / f"scorebar_overlay_{chosen_preset}_t{ts:.2f}.png"))

        patches = extract_rois(sb_canon, chosen_preset, pad=2)

        def ocr_roi(name: str) -> str:
            patch = patches.get(name)
            if patch is None:
                return ""
            pil_patch = Image.fromarray(patch)
            patch_bw = _prep_gray_bw(pil_patch, thresh=160, invert=False)
            return _ocr_text(ocr, patch_bw)

        a_team_raw = ocr_roi("away_team")
        h_team_raw = ocr_roi("home_team")

        a_sc_patch = patches.get("away_score")
        h_sc_patch = patches.get("home_score")

        # -------- primary score reader (same as before) --------
        def read_score_patch(patch_np: Optional[np.ndarray], tag: str) -> Tuple[Optional[str], Optional[int]]:
            if patch_np is None:
                return None, None
            pil_patch = Image.fromarray(patch_np)

            dg = _ocr_digits_ensemble(ocr, pil_patch, viz_dir=(run_dir if viz else None), tag=tag)
            dg = (dg or "").strip()

            z_conf  = _detect_zero_confidence(pil_patch)
            zc2     = _zero_center_contrast_conf(pil_patch)
            s_conf  = _detect_seven_confidence(pil_patch)
            tpl     = _template_match_digits_with_conf(pil_patch, chosen_preset, max_digits=2)
            tpl_guess, tpl_conf = (tpl[0], tpl[1]) if tpl else (None, 0.0)

            gray = np.array(pil_patch.convert("L"))
            bwv  = _prep_binary_variants_gray(gray)
            H, W = gray.shape
            aspect = W / float(H + 1e-6)

            ratios = []
            for bw in bwv:
                cols = (bw < 128).sum(axis=0)
                ratios.append(cols.mean() / max(1, bw.shape[1]))
            fg_ratio = float(np.median(ratios)) if ratios else 0.0

            cc_counts = []
            for bw in bwv:
                cc_counts.append(_count_connected_components(bw if bw.mean() > 127 else (255 - bw)))
            cc = int(np.median(cc_counts)) if cc_counts else 1

            # Skin-specific thresholds
            is_SNP = (chosen_preset == "SNP")
            SEVEN_STRONG = 0.58 if is_SNP else 0.65
            SEVEN_TPL    = 0.38 if is_SNP else 0.40
            ZERO_STRONG  = 0.70 if is_SNP else 0.72
            ZERO_PAIR    = 0.58 if is_SNP else 0.60
            ZERO_TPL     = 0.52 if is_SNP else 0.55
            TPL_2DIG     = 0.48 if is_SNP else 0.50

            # ZERO handling (safe; never override ≥2 digits)
            strong_zero = (z_conf >= ZERO_STRONG) or ((z_conf >= ZERO_PAIR) and (zc2 >= ZERO_PAIR))
            tpl_zero_ok = (tpl_guess == "0" and tpl_conf >= ZERO_TPL)

            if not dg:
                if strong_zero or tpl_zero_ok:
                    dg = "0"
            elif len(dg) == 1 and dg != "0":
                if strong_zero:
                    dg = "0"

            # Seven vs one
            if (not dg) or (dg == "1"):
                if s_conf >= SEVEN_STRONG:
                    dg = "7"
                elif tpl_guess == "7" and tpl_conf >= SEVEN_TPL:
                    dg = "7"
                else:
                    if fg_ratio >= 0.33 and tpl_guess:
                        dg = tpl_guess

            # Two-digit rescue (e.g., '10','24')
            if (not dg) or (len(dg) == 1):
                if tpl_guess and len(tpl_guess) == 2 and tpl_conf >= TPL_2DIG:
                    if (aspect >= 0.90) or (cc >= 2):
                        dg = tpl_guess

            if dg and len(dg) > 2:
                dg = dg[-2:]

            return (dg or None, (int(dg) if dg and dg.isdigit() else None))

        a_sc_text, a_sc_int = read_score_patch(a_sc_patch, "away")
        h_sc_text, h_sc_int = read_score_patch(h_sc_patch, "home")

        # -------- targeted rescue for TNP-away if missed --------
        if chosen_preset == "TNP" and (a_sc_text is None or a_sc_text == "") and a_sc_patch is not None:
            r_text, r_int, r_zero_hint = _rescue_read_digits_for_tnp_away(a_sc_patch)
            if r_text is not None or r_int is not None:
                a_sc_text = r_text if r_text is not None else a_sc_text
                a_sc_int = r_int if r_int is not None else a_sc_int
            elif r_zero_hint and a_sc_int is None:
                # gentle null->0 mapping ONLY for TNP-away, when geometry suggests zero
                a_sc_int = 0

        q_raw = ocr_roi("quarter")
        c_raw = ocr_roi("clock")

        away_team_reads.append(_norm_team(a_team_raw))
        home_team_reads.append(_norm_team(h_team_raw))

        away_score_text_reads.append(a_sc_text)
        home_score_text_reads.append(h_sc_text)
        away_score_int_reads.append(a_sc_int)
        home_score_int_reads.append(h_sc_int)

        clock_reads.append(_norm_clock(c_raw))
        quarter_raw_reads.append((q_raw or "").strip() or None)

        # Optional instrumentation dump if TNP-away still null
        if viz and chosen_preset == "TNP" and (a_sc_text is None) and a_sc_patch is not None:
            p = Path(run_dir) / f"tnp_away_debug_t{ts:.2f}.png"
            Image.fromarray(a_sc_patch).save(p)
            gray = np.array(Image.fromarray(a_sc_patch).convert("L"))
            best_bw = _choose_polarity(gray)
            Image.fromarray(best_bw).save(Path(run_dir) / f"tnp_away_debug_bestbw_t{ts:.2f}.png")

    # Aggregate across frames
    away_team = _mode_str(away_team_reads)
    home_team = _mode_str(home_team_reads)

    away_score_text = _mode_str(away_score_text_reads)
    home_score_text = _mode_str(home_score_text_reads)

    away_score_best = _mode_int(away_score_int_reads)
    home_score_best = _mode_int(home_score_int_reads)

    clk = _mode_str(clock_reads)
    quarter_text = _mode_str(quarter_raw_reads)
    qtr = quarter_text  # passthrough for now

    result: Dict[str, Any] = {
        "away_team": away_team,
        "home_team": home_team,
        "away_score_text": away_score_text,
        "home_score_text": home_score_text,
        "away_score": away_score_best,
        "home_score": home_score_best,
        "clock": clk,
        "quarter_text": quarter_text,
        "quarter": qtr,
        "used_stub": False,
        "_preset": chosen_preset,
    }

    if not any((away_team, home_team, away_score_text, home_score_text, clk, quarter_text)):
        return {"used_stub": True, "_preset": chosen_preset}

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
