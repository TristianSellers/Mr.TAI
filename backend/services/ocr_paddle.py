# backend/services/ocr_paddle.py
from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, TYPE_CHECKING
import os, re, subprocess, shlex, time

# ----- Optional dependency: PaddleOCR -----
try:
    from paddleocr import PaddleOCR  # type: ignore
except Exception:
    PaddleOCR = None  # type: ignore

if TYPE_CHECKING:
    from paddleocr import PaddleOCR as PaddleOCRType
else:
    PaddleOCRType = object

# Pillow / drawing
from PIL import Image, ImageDraw, ImageOps

# ---------- Reference full-frame → scoreboard crop ----------
# These are the same as your legacy code (bottom strip on 2047x1155 reference)
REF_FRAME_W, REF_FRAME_H = 2047, 1155
SCOREBAR_BOX = (0, 1080, 2047, 1155)  # (left, top, right, bottom) in reference space

# ---------- Your pixel ROIs on the scoreboard crop (x0, y0, x1, y1) ----------
# (Do NOT change unless you want to retune; these are your confirmed boxes)
SB_ROIS_PX: Dict[str, Tuple[int,int,int,int]] = {
    "away_team":  (500,  0,   600,  50),
    "home_team":  (900,  0,  1000,  50),
    "away_score": (655,  0,   745,  72),
    "home_score": (765,  0,   855,  72),
    "quarter":    (1285, 20, 1340,  50),
    "clock":      (1325, 20, 1445,  50),
}

# ------- Regex helpers -------
_SCORE_RX   = re.compile(r"^\s*(\d{1,2})\s*[-:]\s*(\d{1,2})\s*$")
_CLOCK_RX   = re.compile(r"^\s*(\d{1,2}):([0-5]\d)\s*$")
_QTR_RX_QN  = re.compile(r"^(Q[1-4]|OT)$", re.I)
_QTR_RX_MDN = re.compile(r"^(1ST|2ND|3RD|4TH|OT)$", re.I)

# ---------- Paddle singleton ----------
_OCR_SINGLETON: Optional[PaddleOCRType] = None

def _get_paddle_ocr() -> Optional[PaddleOCRType]:
    global _OCR_SINGLETON
    if _OCR_SINGLETON is not None:
        return _OCR_SINGLETON
    if PaddleOCR is None:
        return None
    # English, no angle classifier; quiet logs
    _OCR_SINGLETON = PaddleOCR(use_angle_cls=False, lang="en", show_log=False)
    return _OCR_SINGLETON

# ---------- FFmpeg frame grab + scoreboard crop ----------
def _scale_box_to_frame(box: Tuple[int,int,int,int], w: int, h: int) -> Tuple[int,int,int,int]:
    lx, ty, rx, by = box
    sx = w / REF_FRAME_W
    sy = h / REF_FRAME_H
    return (int(lx * sx), int(ty * sy), int(rx * sx), int(by * sy))

def _grab_frame(video_path: str, t: float, out_png: Path) -> Path:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    cmd = f'ffmpeg -y -v error -ss {t:.3f} -i {shlex.quote(video_path)} -frames:v 1 {shlex.quote(str(out_png))}'
    subprocess.run(cmd, shell=True, check=False)
    if not out_png.exists():
        raise RuntimeError(f"Failed to grab frame at t={t:.3f}s")
    return out_png

def _crop_scorebar(full_frame_png: Path, out_png: Path) -> Path:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(full_frame_png) as im:
        w, h = im.size
        lx, ty, rx, by = _scale_box_to_frame(SCOREBAR_BOX, w, h)
        crop = im.crop((lx, ty, rx, by))
        crop.save(out_png)
        return out_png

# ---------- ROI helpers ----------
_SB_REF_OVERRIDE: Optional[Tuple[int,int]] = None  # set by first crop we see

def _prep_gray_bw(pil: Image.Image, thresh: int = 165, invert: bool = False) -> Image.Image:
    g = pil.convert("L")
    table = [0] * (thresh + 1) + [255] * (256 - (thresh + 1))
    bw = g.point(table, mode="L")
    if invert:
        bw = ImageOps.invert(bw)
    return bw

def _crop_px_from_ref(sb_img: Image.Image, box_px: Tuple[int,int,int,int]) -> Image.Image:
    """
    Scale your pixel coords from the *auto-detected* reference (first crop size)
    to the current crop size; lets you reuse the same boxes across different
    scorebar resolutions.
    """
    global _SB_REF_OVERRIDE
    w, h = sb_img.size
    if _SB_REF_OVERRIDE is None:
        _SB_REF_OVERRIDE = (w, h)
    ref_w, ref_h = _SB_REF_OVERRIDE
    sx = w / float(ref_w)
    sy = h / float(ref_h)
    x0, y0, x1, y1 = box_px
    X0, Y0 = max(0, int(round(x0 * sx))), max(0, int(round(y0 * sy)))
    X1, Y1 = min(w, int(round(x1 * sx))),  min(h, int(round(y1 * sy)))
    return sb_img.crop((X0, Y0, X1, Y1))

def _viz_rois(sb_img_path: Path, out_base: Path) -> None:
    """
    Save:
      - <out_base>_zones.png : scorebar with colored ROI rectangles
      - <out_base>_<key>_raw.png : raw crop per ROI
      - <out_base>_<key>_bw.png  : binarized crop per ROI
    """
    with Image.open(sb_img_path) as sb:
        # ensure ref set
        _ = _crop_px_from_ref(sb, SB_ROIS_PX["away_team"])
        w, h = sb.size
        ref_w, ref_h = _SB_REF_OVERRIDE or (w, h)
        sx = w / float(ref_w)
        sy = h / float(ref_h)

        vis = sb.copy()
        d = ImageDraw.Draw(vis)
        colors = {
            "away_team": "orange",
            "home_team": "orange",
            "away_score":"lime",
            "home_score":"lime",
            "quarter":   "deepskyblue",
            "clock":     "deepskyblue",
        }
        out_base.parent.mkdir(parents=True, exist_ok=True)

        # draw rectangles
        for key, (x0,y0,x1,y1) in SB_ROIS_PX.items():
            X0, Y0 = int(round(x0*sx)), int(round(y0*sy))
            X1, Y1 = int(round(x1*sx)), int(round(y1*sy))
            d.rectangle((X0, Y0, X1, Y1), outline=colors.get(key,"yellow"), width=3)

        vis.save(out_base.with_name(out_base.name + "_zones.png"))

        # per-ROI crops
        for key in SB_ROIS_PX.keys():
            c = _crop_px_from_ref(sb, SB_ROIS_PX[key])
            c.save(out_base.with_name(out_base.name + f"_{key}_raw.png"))
            _prep_gray_bw(c).save(out_base.with_name(out_base.name + f"_{key}_bw.png"))

# ---------- Small text normalizers ----------
def _norm_quarter(s: Optional[str]) -> Optional[str]:
    if not s: return None
    x = s.strip().upper().replace(" ", "")
    x = re.sub(r"1{2,}", "1", x)  # "111st" -> "1st"
    x = x.replace("0T", "OT").replace("QI", "Q1").replace("QT", "Q1")
    m = _QTR_RX_QN.match(x)
    if m: return m.group(1).upper()
    m2 = _QTR_RX_MDN.match(x)
    if m2: return {"1ST":"Q1","2ND":"Q2","3RD":"Q3","4TH":"Q4","OT":"OT"}[m2.group(1)]
    if re.fullmatch(r"[1-4]", x):
        return f"Q{x}"
    return None

def _norm_clock(s: Optional[str]) -> Optional[str]:
    if not s: return None
    x = s.strip().replace(";", ":").replace(" ", "")
    if not re.search(r"\d", x):
        return None
    m = _CLOCK_RX.match(x)
    if m:
        return f"{int(m.group(1))}:{m.group(2)}"
    return None

def _digits(s: Optional[str]) -> Optional[str]:
    if not s: return None
    m = re.search(r"\d{1,2}", s)
    return m.group(0) if m else None

# ---------- OCR routines ----------
def _ocr_text(paddle: PaddleOCRType, pil: Image.Image) -> str:
    # PaddleOCR expects file path or ndarray; we can save temp PNG in ramdisk dir
    tmp = Path("data/tmp/ocr_patch.png")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    pil.save(tmp)
    res = paddle.ocr(str(tmp), cls=False)  # type: ignore
    txts: List[str] = []
    if res and isinstance(res, list) and res[0]:
        for line in res[0]:
            # line: [ [[x,y],...8], (text, conf) ]
            txt = line[1][0]
            if isinstance(txt, str) and txt.strip():
                txts.append(txt.strip())
    try:
        tmp.unlink(missing_ok=True)
    except Exception:
        pass
    return " ".join(txts)

def _ocr_digits(paddle: PaddleOCRType, pil: Image.Image) -> Optional[str]:
    raw = _ocr_text(paddle, pil)
    clean = re.sub(r"[^0-9]", "", raw)
    return clean or None

def _ocr_letters(paddle: PaddleOCRType, pil: Image.Image) -> Optional[str]:
    raw = _ocr_text(paddle, pil)
    clean = re.sub(r"[^A-Za-z]", "", raw).upper()
    return clean or None

def _compose_score(away_s: Optional[str], home_s: Optional[str], joined: Optional[str]) -> Optional[str]:
    if away_s is not None and home_s is not None and away_s != "" and home_s != "":
        return f"{int(away_s)}-{int(home_s)}"
    if joined:
        m = _SCORE_RX.match(joined.replace(":", "-"))
        if m:
            return f"{int(m.group(1))}-{int(m.group(2))}"
    return None

# ---------- Public entry ----------
def extract_scoreboard_from_video_paddle(
    video_path: str,
    *,
    t: float = 0.10,
    attempts: int = 3,
    viz: bool = False,
) -> Dict[str, Any]:
    """
    Grab N frames around t, crop the scoreboard, OCR fixed ROIs with PaddleOCR,
    and return a normalized scoreboard dict. If viz=True, saves ROI overlays and crops
    under data/tmp/.
    """
    paddle = _get_paddle_ocr()
    if paddle is None:
        # Dependency missing → tell caller to fall back or stub
        return {"used_stub": True}

    # reset ROI calibration per call
    global _SB_REF_OVERRIDE
    _SB_REF_OVERRIDE = None

    times = [max(0.0, t + 0.10 * k) for k in range(max(1, attempts))]
    best: Dict[str, Any] = {}
    def completeness(d: Dict[str, Any]) -> int:
        return sum(1 for k in ("away_team","home_team","score","clock","quarter") if d.get(k))

    for idx, ts in enumerate(times):
        frame_png = _grab_frame(video_path, ts, Path(f"data/tmp/paddle_frame_t{ts:.2f}.png"))
        sb_png = _crop_scorebar(frame_png, Path(f"data/tmp/paddle_scorebar_t{ts:.2f}.png"))

        if viz and idx == 0:
            _viz_rois(sb_png, sb_png.with_name(f"paddle_scorebar_viz_t{ts:.2f}"))

        with Image.open(sb_png) as sb_im:
            away_team = _ocr_letters(paddle, _crop_px_from_ref(sb_im, SB_ROIS_PX["away_team"]))
            home_team = _ocr_letters(paddle, _crop_px_from_ref(sb_im, SB_ROIS_PX["home_team"]))

            away_sc = _ocr_digits(paddle, _crop_px_from_ref(sb_im, SB_ROIS_PX["away_score"]))
            home_sc = _ocr_digits(paddle, _crop_px_from_ref(sb_im, SB_ROIS_PX["home_score"]))

            quarter_raw = _ocr_text(paddle, _crop_px_from_ref(sb_im, SB_ROIS_PX["quarter"]))
            clock_raw   = _ocr_text(paddle, _crop_px_from_ref(sb_im, SB_ROIS_PX["clock"]))

            # joined score (rarely used, but try from whole scorebar if needed)
            joined = None  # we keep ROIs tight; joined OCR is noisy on Madden bar

        q_norm = _norm_quarter(quarter_raw)
        c_norm = _norm_clock(clock_raw)
        s_norm = _compose_score(away_sc, home_sc, joined)

        cur = {
            "away_team": away_team,
            "home_team": home_team,
            "score": s_norm,
            "clock": c_norm,
            "quarter": q_norm,
        }

        if completeness(cur) > completeness(best):
            best = cur
            if completeness(best) == 5:
                break
        # tiny pause helps when reading sequential frames from disk
        time.sleep(0.01)

    if not best:
        return {"used_stub": True}
    best["used_stub"] = False
    return best
