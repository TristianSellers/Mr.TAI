# backend/services/ocr_paddle.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import os, uuid, shlex, subprocess, re, shutil
from typing import Any, Optional, TYPE_CHECKING

try:
    from paddleocr import PaddleOCR  # runtime import
except Exception:
    PaddleOCR = None  # type: ignore[assignment]

if TYPE_CHECKING:
    # static type name only visible to the type checker
    from paddleocr import PaddleOCR as _PaddleOCRType
else:
    _PaddleOCRType = Any  # at runtime we don't care

from PIL import Image, ImageDraw

# ---------- Config ----------
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
TMP_ROOT = DATA_DIR / "tmp" / "runs"
TMP_ROOT.mkdir(parents=True, exist_ok=True)

# Reference full frame dims & scoreboard strip at bottom (like your old pipeline)
REF_FRAME_W, REF_FRAME_H = 2047, 1155
SCOREBAR_BOX = (0, 1080, 2047, 1155)  # left, top, right, bottom in reference space

# ROI fractions within SCOREBAR (kept simple/stable; adjust if you like)
# (fx0, fy0, fx1, fy1) in 0..1 of the scorebar crop
Z_TEAMS = (0.03, 0.08, 0.42, 0.92)
Z_SCORE_L = (0.42, 0.12, 0.50, 0.90)
Z_SCORE_R = (0.50, 0.12, 0.58, 0.90)
Z_PERIOD  = (0.62, 0.12, 0.75, 0.90)
Z_CLOCK   = (0.74, 0.12, 0.96, 0.90)

# Regex helpers
_SCORE_JOINED = re.compile(r"\b(\d{1,2})\s*[-:]\s*(\d{1,2})\b")
_CLOCK_RX     = re.compile(r"^\s*(\d{1,2}):([0-5]\d)\s*$")
_QTR_QN       = re.compile(r"^(Q[1-4]|OT)$", re.I)
_QTR_MDN      = re.compile(r"^(1ST|2ND|3RD|4TH|OT)$", re.I)

def _normalize_quarter(txt: Optional[str]) -> Optional[str]:
    if not txt:
        return None
    x = txt.strip().upper()
    x = x.replace(" ", "").replace("0T", "OT").replace("QI", "Q1")
    m = _QTR_QN.match(x)
    if m:
        return m.group(1).upper()
    m = _QTR_MDN.match(x)
    if m:
        return {"1ST": "Q1", "2ND": "Q2", "3RD": "Q3", "4TH": "Q4", "OT": "OT"}[m.group(1)]
    if re.fullmatch(r"[1-4]", x):
        return f"Q{x}"
    return None

def _normalize_clock(txt: Optional[str]) -> Optional[str]:
    if not txt:
        return None
    x = txt.strip().replace(";", ":").replace(" ", "")
    m = _CLOCK_RX.match(x)
    if m:
        return f"{int(m.group(1))}:{m.group(2)}"
    return None

def _scale_scorebar_box_to_frame(box: Tuple[int,int,int,int], w: int, h: int) -> Tuple[int,int,int,int]:
    lx, ty, rx, by = box
    sx = w / REF_FRAME_W
    sy = h / REF_FRAME_H
    return (int(lx * sx), int(ty * sy), int(rx * sx), int(by * sy))

def _crop_frac(im: Image.Image, fx0: float, fy0: float, fx1: float, fy1: float) -> Image.Image:
    w, h = im.size
    x0, y0 = int(fx0 * w), int(fy0 * h)
    x1, y1 = int(fx1 * w), int(fy1 * h)
    return im.crop((x0, y0, x1, y1))

def _grab_frame(video_path: str, ts: float, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = f'ffmpeg -y -v error -ss {ts:.3f} -i {shlex.quote(video_path)} -frames:v 1 {shlex.quote(str(out_path))}'
    subprocess.run(cmd, shell=True, check=False)
    if not out_path.exists():
        raise RuntimeError(f"Could not grab frame at t={ts:.3f}s")
    return out_path

def _crop_scorebar(frame_png: Path, out_path: Path) -> Path:
    with Image.open(frame_png) as im:
        w, h = im.size
        lx, ty, rx, by = _scale_scorebar_box_to_frame(SCOREBAR_BOX, w, h)
        sb = im.crop((lx, ty, rx, by))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sb.save(out_path)
    return out_path

# Lazy OCR instance (heavy to init)
_OCR: Optional[_PaddleOCRType] = None

def _get_ocr() -> Optional[_PaddleOCRType]:
    global _OCR
    if _OCR is None and PaddleOCR is not None:
        _OCR = PaddleOCR(use_angle_cls=False, lang="en", show_log=False)  # type: ignore[call-arg]
    return _OCR

def _ocr_image_text(pil_img: Image.Image) -> List[Tuple[str, float]]:
    """
    Run OCR on a PIL image object (no temp file races).
    Returns list of (text, confidence).
    """
    ocr = _get_ocr()
    if ocr is None:
        return []
    # PaddleOCR accepts ndarray path or file path; we’ll use ndarray via temp PNG in-memory.
    # Convert PIL -> tmp path within run dir is avoided; feed ndarray directly.
    import numpy as np  # local import to avoid global hard deps during import
    arr = np.array(pil_img.convert("RGB"))
    try:
        res = ocr.ocr(arr, cls=False)
    except Exception:
        return []
    out: List[Tuple[str, float]] = []
    if isinstance(res, list) and res and isinstance(res[0], list):
        for line in res[0]:
            if not line or len(line) < 2: 
                continue
            txt = line[1][0]
            conf = float(line[1][1] or 0.0)
            if isinstance(txt, str) and txt.strip():
                out.append((txt.strip(), conf))
    return out

def _join_text(lines: List[Tuple[str, float]]) -> str:
    return " ".join(t for t, _ in lines)

def _parse_score(away_txt: str, home_txt: str, joined_txt: str) -> Optional[str]:
    # Prefer separate digits first
    def _first_int(s: str) -> Optional[int]:
        m = re.search(r"\d{1,2}", s)
        return int(m.group(0)) if m else None
    a = _first_int(away_txt)
    h = _first_int(home_txt)
    if a is not None and h is not None:
        return f"{a}-{h}"
    # Fallback to joined “A-H”
    m = _SCORE_JOINED.search(joined_txt.replace(":", "-"))
    if m:
        return f"{int(m.group(1))}-{int(m.group(2))}"
    return None

def _viz_rois(scorebar_png: Path, out_path: Path) -> None:
    with Image.open(scorebar_png) as sb:
        vis = sb.copy()
        d = ImageDraw.Draw(vis)
        def draw(box, color):
            w, h = sb.size
            fx0, fy0, fx1, fy1 = box
            d.rectangle((int(fx0*w), int(fy0*h), int(fx1*w), int(fy1*h)), outline=color, width=3)
        draw(Z_TEAMS,   "orange")
        draw(Z_SCORE_L, "lime")
        draw(Z_SCORE_R, "lime")
        draw(Z_PERIOD,  "deepskyblue")
        draw(Z_CLOCK,   "deepskyblue")
        vis.save(out_path)

def extract_scoreboard_from_video_paddle(
    video_path: str,
    *,
    t: float = 0.10,
    attempts: int = 3,
    viz: bool = False,
) -> Dict[str, Any]:
    """
    Per-request isolated temp dir; OCR the bottom scorebar with PaddleOCR.
    Returns dict: away_team, home_team, score, clock, quarter, used_stub
    """
    # --- per-run folder ---
    run_id = uuid.uuid4().hex[:10]
    run_dir = TMP_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    keep_tmp = os.getenv("KEEP_TMP", "0") == "1"
    try:
        # Collect a few timestamps
        times = [max(0.0, t + k*0.10) for k in range(max(1, attempts))]
        best: Dict[str, Any] = {}
        def completeness(d: Dict[str, Any]) -> int:
            return sum(1 for k in ("away_team","home_team","score","clock","quarter") if d.get(k))

        for idx, ts in enumerate(times):
            frame_png = run_dir / f"frame_t{ts:.2f}.png"
            score_png = run_dir / f"scorebar_t{ts:.2f}.png"
            _grab_frame(video_path, ts, frame_png)
            _crop_scorebar(frame_png, score_png)

            if viz and idx == 0:
                _viz_rois(score_png, run_dir / f"scorebar_rois_t{ts:.2f}.png")

            # OCR per ROI
            with Image.open(score_png) as sb:
                teams_img = _crop_frac(sb, *Z_TEAMS)
                l_img     = _crop_frac(sb, *Z_SCORE_L)
                r_img     = _crop_frac(sb, *Z_SCORE_R)
                per_img   = _crop_frac(sb, *Z_PERIOD)
                clk_img   = _crop_frac(sb, *Z_CLOCK)

                teams_txt = _join_text(_ocr_image_text(teams_img))
                away_frag = _join_text(_ocr_image_text(l_img))
                home_frag = _join_text(_ocr_image_text(r_img))
                per_txt   = _join_text(_ocr_image_text(per_img))
                clk_txt   = _join_text(_ocr_image_text(clk_img))

            # Basic normalization
            # Try to split teams into two short tokens if possible (left/right zones are digits only, so we trust Z_TEAMS for names)
            away_team = None
            home_team = None
            if teams_txt:
                toks = re.findall(r"[A-Z]{2,4}", teams_txt.upper())
                if len(toks) >= 2:
                    away_team, home_team = toks[0], toks[1]
                elif len(toks) == 1:
                    # best-effort single token
                    away_team = toks[0]

            score = _parse_score(away_frag, home_frag, f"{away_frag} {home_frag}")
            quarter = _normalize_quarter(per_txt)
            clock   = _normalize_clock(clk_txt)

            cur = {
                "away_team": away_team,
                "home_team": home_team,
                "score": score,
                "clock": clock,
                "quarter": quarter,
            }

            if completeness(cur) > completeness(best):
                best = cur
                if completeness(best) == 5:
                    break

        if not best:
            return {"used_stub": True}
        best["used_stub"] = False
        # Drop where we wrote artifacts (useful for debugging)
        if viz:
            best["_debug_dir"] = str(run_dir)
        return best

    finally:
        # Clean up run folder unless explicitly asked to keep or viz requested
        if not viz and not keep_tmp:
            try:
                shutil.rmtree(run_dir, ignore_errors=True)
            except Exception:
                pass
