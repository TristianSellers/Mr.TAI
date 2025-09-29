# backend/services/ocr_from_video.py
from __future__ import annotations
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import subprocess, shlex, json, re
from collections import Counter

from .ocr import extract_scoreboard_from_image, draw_boxes  # legacy extractor (fallback)

# Optional OpenCV for grabbing frames
try:
    import cv2  # type: ignore
except Exception:
    cv2 = None  # fallback to ffmpeg if available

# Pillow for cropping & drawing
try:
    from PIL import Image, ImageDraw, ImageOps  # type: ignore
except Exception as e:
    raise RuntimeError("Pillow is required. pip install Pillow") from e

# Pillow resample constant (compat across Pillow versions)
try:
    RESAMPLE_NEAREST = Image.Resampling.NEAREST   # Pillow ≥ 9.1
except Exception:
    RESAMPLE_NEAREST = 0                          # older Pillow integer code

# Optional pytesseract for fast path
try:
    import pytesseract  # type: ignore
except Exception:
    pytesseract = None

# ----- Reference full-frame -> scoreboard crop -----
REF_FRAME_W, REF_FRAME_H = 2047, 1155
SCOREBAR_BOX = (0, 1080, 2047, 1155)  # (left, top, right, bottom) in reference space

# ----- EXACT pixel ROIs on the scoreboard crop (x0, y0, x1, y1) -----
SB_ROIS_PX: Dict[str, Tuple[int, int, int, int]] = {
    "away_team":  (500,  0, 600, 50),
    "home_team":  (900,  0, 1000,50),
    "away_score": (655,  0, 745, 72),
    "home_score": (765,  0, 855, 72),
    "quarter":    (1285, 20,1340,50),
    "clock":      (1325, 20,1445,50),
}

_SB_REF_OVERRIDE: Optional[Tuple[int, int]] = None  # (ref_w, ref_h)

# ----- regex/validators -----
_SCORE_RX   = re.compile(r"^\s*(\d{1,2})\s*[-:]\s*(\d{1,2})\s*$")
_CLOCK_RX   = re.compile(r"^\s*(\d{1,2}):([0-5]\d)\s*$")
_QTR_RX_MDN = re.compile(r"^(1ST|2ND|3RD|4TH|OT)$", re.I)
_QTR_RX_QN  = re.compile(r"^(Q[1-4]|OT)$", re.I)

def _normalize_quarter(s: Optional[str]) -> Optional[str]:
    if not s: return None
    s = s.strip().upper().replace(" ", "")
    s = s.replace("0T", "OT").replace("QI", "Q1").replace("QT", "Q1")
    m = _QTR_RX_MDN.match(s)
    if m:
        return {"1ST":"Q1","2ND":"Q2","3RD":"Q3","4TH":"Q4","OT":"OT"}[m.group(1)]
    m2 = _QTR_RX_QN.match(s)
    if m2:
        return m2.group(1).upper()
    if re.fullmatch(r"[1-4]", s):
        return f"Q{s}"
    return None

def _mode_int(counts: Dict[int,int]) -> Optional[int]:
    if not counts: return None
    maxc = max(counts.values())
    winners = [k for k, c in counts.items() if c == maxc]
    return min(winners)

def _mode_val(values: List[str]) -> Optional[str]:
    vs = [v for v in values if v]
    return Counter(vs).most_common(1)[0][0] if vs else None

def _to_int_or_none(val: object) -> Optional[int]:
    if val is None:
        return None
    s = str(val).strip()
    if not re.fullmatch(r"\d{1,2}", s):
        return None
    try:
        return int(s)
    except Exception:
        return None

# ----- frame grabbing -----
def grab_frame_at_time(video_path: str, ts: float, out_path: str) -> Path:
    src = Path(video_path); out = Path(out_path)
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

def _scale_box_to_frame(box: Tuple[int,int,int,int], w: int, h: int) -> Tuple[int,int,int,int]:
    lx, ty, rx, by = box
    sx = w / REF_FRAME_W; sy = h / REF_FRAME_H
    return (int(lx * sx), int(ty * sy), int(rx * sx), int(by * sy))

def crop_scorebar_from_frame(frame_path: str, out_path: str, box: Tuple[int,int,int,int] = SCOREBAR_BOX) -> Path:
    ip = Path(frame_path); op = Path(out_path)
    op.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(ip) as im:
        w, h = im.size
        lx, ty, rx, by = _scale_box_to_frame(box, w, h) if (w, h) != (REF_FRAME_W, REF_FRAME_H) else box
        im.crop((lx, ty, rx, by)).save(op)
        return op

def draw_frame_crop_outline(frame_path: str, out_path: str = "data/tmp/frame_with_scorebar_box.png") -> Path:
    ip = Path(frame_path); op = Path(out_path)
    op.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(ip) as im:
        w, h = im.size
        lx, ty, rx, by = _scale_box_to_frame(SCOREBAR_BOX, w, h)
        vis = im.copy(); d = ImageDraw.Draw(vis)
        d.rectangle((lx, ty, rx, by), outline="lime", width=3)
        vis.save(op)
    return op

# ----- ROI helpers -----
def _prep_gray_bw(pil: Image.Image, thresh: int = 160, invert: bool = False) -> Image.Image:
    g = pil.convert("L")
    table = [0] * (thresh + 1) + [255] * (256 - (thresh + 1))
    bw = g.point(table, mode="L")
    return ImageOps.invert(bw) if invert else bw

def _crop_px_from_ref(sb_img: Image.Image, box_px: Tuple[int,int,int,int]) -> Image.Image:
    global _SB_REF_OVERRIDE
    w, h = sb_img.size
    if _SB_REF_OVERRIDE is None:
        _SB_REF_OVERRIDE = (w, h)
    ref_w, ref_h = _SB_REF_OVERRIDE
    sx = w / float(ref_w); sy = h / float(ref_h)
    x0, y0, x1, y1 = box_px
    X0, Y0 = max(0, int(round(x0 * sx))), max(0, int(round(y0 * sy)))
    X1, Y1 = min(w, int(round(x1 * sx))),  min(h, int(round(y1 * sy)))
    return sb_img.crop((X0, Y0, X1, Y1))

def _autotrim_to_content(pil: Image.Image, bg_is_white: bool = True) -> Image.Image:
    bw = _prep_gray_bw(pil, thresh=165, invert=bg_is_white)  # invert=True => text dark on white
    inv = ImageOps.invert(bw)
    box = inv.getbbox()
    if box:
        return pil.crop(box)
    return pil

# ---- OCR drivers ----
def _tess(pil: Image.Image, allow: str, psm: int) -> str:
    if pytesseract is None:
        return ""
    cfg = f'--psm {psm} -c tessedit_char_whitelist={allow}'
    return (pytesseract.image_to_string(pil, config=cfg) or "").strip()

def _single_read_alnum_fast(pil: Image.Image, allow: str) -> str:
    bw = _prep_gray_bw(pil, thresh=160, invert=False)
    s = _tess(bw, allow, psm=7)
    return (s or "").strip()

def _ocr_letters_fast(pil: Image.Image) -> str:
    s = _single_read_alnum_fast(pil, allow="ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    return re.sub(r"[^A-Z]", "", s.upper()).strip()

def _ocr_clock_fast(pil: Image.Image) -> Optional[str]:
    s = _single_read_alnum_fast(pil, allow="0123456789:")
    m = _CLOCK_RX.match(s)
    return f"{int(m.group(1))}:{m.group(2)}" if m else None

# ---- donut heuristic for 0 ----
def _center_white_ratio(img: Image.Image) -> float:
    hist = img.histogram()
    total = sum(hist); white = hist[255] if len(hist) >= 256 else 0
    return (white / total) if total else 0.0

def _looks_like_zero(pil: Image.Image) -> bool:
    """
    Upscale, binarize (both normal & inverted). If the center box has a notably higher
    white ratio than the surrounding ring in either polarity, it’s likely a '0'.
    """
    im = pil.resize((max(1, pil.width*3), max(1, pil.height*3)), resample=RESAMPLE_NEAREST)
    for invert in (False, True):
        bw = _prep_gray_bw(im, thresh=170, invert=invert)
        W, H = bw.size
        cx0, cy0 = int(W*0.30), int(H*0.30)
        cx1, cy1 = int(W*0.70), int(H*0.70)
        center = bw.crop((cx0, cy0, cx1, cy1))
        ring   = bw.crop((int(W*0.15), int(H*0.15), int(W*0.85), int(H*0.85)))
        if (_center_white_ratio(center) - _center_white_ratio(ring)) > 0.08:
            return True
    return False

def _read_digit_strict(pil: Image.Image) -> Optional[str]:
    """
    Single-digit robust read with auto-trim + RANSAC-like voting.
    """
    if pytesseract is None:
        return None
    base = _autotrim_to_content(pil, bg_is_white=True)
    votes: List[str] = []
    for scale in (2, 3):
        im = base.resize((max(1, base.width*scale), max(1, base.height*scale)), resample=RESAMPLE_NEAREST)
        for thresh in (145, 160, 170, 185):
            for inv in (False, True):
                bw = _prep_gray_bw(im, thresh=thresh, invert=inv)
                for psm in (10, 13):  # single char
                    s = _tess(bw, "0123456789", psm)
                    s = re.sub(r"[^0-9]", "", s)
                    if len(s) == 1:
                        votes.append(s)
    if votes:
        pick, _ = Counter(votes).most_common(1)[0]
        if pick in {"7", "2"} and _looks_like_zero(base):
            return "0"
        return pick
    # fallback: allow 'O' -> '0'
    im2 = base.resize((max(1, base.width*3), max(1, base.height*3)), resample=RESAMPLE_NEAREST)
    s2 = _tess(_prep_gray_bw(im2, thresh=165, invert=False), "O0123456789", psm=10)
    s2 = re.sub(r"[^0-9O]", "", s2).replace("O", "0")
    if len(s2) == 1:
        if s2 in {"7", "2"} and _looks_like_zero(base):
            return "0"
        return s2
    return None

def _read_score_1or2_digits(pil: Image.Image) -> Optional[str]:
    """
    Try to read a 1–2 digit score from a single ROI:
      A) direct 1–2 digit read (fast)
      B) fallback: split ROI into halves and read each with _read_digit_strict
    """
    if pytesseract is None:
        return None
    base = _autotrim_to_content(pil, bg_is_white=True)
    # A) direct
    for scale in (2, 3):
        im = base.resize((max(1, base.width*scale), max(1, base.height*scale)), resample=RESAMPLE_NEAREST)
        for thresh in (150, 165, 180):
            for inv in (False, True):
                bw = _prep_gray_bw(im, thresh=thresh, invert=inv)
                for psm in (7, 8, 6):  # line/word
                    s = _tess(bw, "0123456789", psm)
                    s = re.sub(r"[^0-9]", "", s)
                    if 1 <= len(s) <= 2:
                        if len(s) == 1 and s in {"2", "7"} and _looks_like_zero(base):
                            return "0"
                        return s
    # B) halves
    W, H = base.size
    mid = max(1, W // 2)
    left  = base.crop((0, 0, mid, H))
    right = base.crop((mid, 0, W, H))
    d1 = _read_digit_strict(left)
    d2 = _read_digit_strict(right)
    if d1 and d2:
        return f"{d1}{d2}"
    if d1:
        return d1
    if d2:
        return d2
    return None

def _ocr_quarter_light_multi(pil: Image.Image) -> Optional[str]:
    tried: List[str] = []
    for scale in (1, 2):
        im = pil.resize((max(1, pil.width*scale), max(1, pil.height*scale)), resample=RESAMPLE_NEAREST) if scale > 1 else pil
        for thresh in (150, 170):
            for inv in (False, True):
                bw = _prep_gray_bw(im, thresh=thresh, invert=inv)
                for psm in (6, 7, 10, 13):
                    s = _tess(bw, "0123456789QOTNDRSTH", psm)
                    s = re.sub(r"[^A-Z0-9]", "", (s or "").upper())
                    tried.append(s)
                    q = _normalize_quarter(s)
                    if q:
                        return q
    for s in tried:
        s2 = s.replace("0T", "OT").replace("QI", "Q1").replace("QT", "Q1")
        q = _normalize_quarter(s2)
        if q:
            return q
    return None

def _extract_fields_fast_with_px_rois(scorebar_path: str) -> Dict[str, Optional[str]]:
    with Image.open(scorebar_path) as sb:
        away_team_txt = _ocr_letters_fast(_crop_px_from_ref(sb, SB_ROIS_PX["away_team"]))
        home_team_txt = _ocr_letters_fast(_crop_px_from_ref(sb, SB_ROIS_PX["home_team"]))
        away_digit = _read_score_1or2_digits(_crop_px_from_ref(sb, SB_ROIS_PX["away_score"]))
        home_digit = _read_score_1or2_digits(_crop_px_from_ref(sb, SB_ROIS_PX["home_score"]))
        q_norm = _ocr_quarter_light_multi(_crop_px_from_ref(sb, SB_ROIS_PX["quarter"]))
        c_norm = _ocr_clock_fast(_crop_px_from_ref(sb, SB_ROIS_PX["clock"]))
        score_norm = f"{away_digit}-{home_digit}" if (away_digit is not None and home_digit is not None) else None
        return {
            "away_team": away_team_txt or None,
            "home_team": home_team_txt or None,
            "away_score": away_digit,
            "home_score": home_digit,
            "score": score_norm,
            "clock": c_norm,
            "quarter": q_norm,
        }

def _viz_px_rois(scorebar_path: str, out_base: str) -> None:
    with Image.open(scorebar_path) as sb:
        _ = _crop_px_from_ref(sb, SB_ROIS_PX["away_team"])
        w, h = sb.size
        ref_w, ref_h = _SB_REF_OVERRIDE or (w, h)
        sx = w / float(ref_w); sy = h / float(ref_h)
        vis = sb.copy(); d = ImageDraw.Draw(vis)
        colors = {
            "away_team": "orange",
            "home_team": "orange",
            "away_score":"lime",
            "home_score":"lime",
            "quarter":   "deepskyblue",
            "clock":     "deepskyblue",
        }
        Path(out_base).parent.mkdir(parents=True, exist_ok=True)
        for key, (x0,y0,x1,y1) in SB_ROIS_PX.items():
            X0, Y0 = int(round(x0*sx)), int(round(y0*sy))
            X1, Y1 = int(round(x1*sx)), int(round(y1*sy))
            d.rectangle((X0, Y0, X1, Y1), outline=colors.get(key,"yellow"), width=3)
        vis.save(f"{out_base}_zones.png")
        for k in SB_ROIS_PX.keys():
            c = _crop_px_from_ref(sb, SB_ROIS_PX[k])
            c.save(f"{out_base}_{k}_raw.png")
            _prep_gray_bw(c).save(f"{out_base}_{k}_bw.png")

# ---------- LEGACY FALLBACK (no Tesseract) ----------
def _legacy_extract(video_path: str, *, viz: bool, dx: int, dy: int, t: float) -> dict:
    """
    Original multi-frame extractor using extract_scoreboard_from_image with per-side voting.
    """
    attempts: List[float] = [max(0.0, t + k*0.10) for k in range(5)]
    readings: List[Tuple[float, dict]] = []
    away_counts: Dict[int,int] = {}
    home_counts: Dict[int,int] = {}
    last_data: Optional[dict] = None

    for idx, ts in enumerate(attempts):
        frame_png = grab_frame_at_time(video_path, ts, out_path=f"data/tmp/video_frame_t{ts:.2f}.png")
        if viz:
            draw_frame_crop_outline(str(frame_png), out_path=f"data/tmp/frame_with_scorebar_box_t{ts:.2f}.png")
        scorebar_png = crop_scorebar_from_frame(str(frame_png), out_path=f"data/tmp/scorebar_t{ts:.2f}.png")

        # debug crops/boxes only on the last attempt
        debug_flag = viz and (idx == len(attempts) - 1)
        result = extract_scoreboard_from_image(str(scorebar_png),
                                               debug_crops=debug_flag,
                                               viz_boxes_flag=debug_flag,
                                               dx=dx, dy=dy)
        data = result.to_dict()
        last_data = data
        readings.append((ts, data))

        m = _SCORE_RX.match((data.get("score") or "").replace(":", "-"))
        if m:
            try:
                a, b = int(m.group(1)), int(m.group(2))
                away_counts[a] = away_counts.get(a, 0) + 1
                home_counts[b] = home_counts.get(b, 0) + 1
            except Exception:
                pass

    away_mode = _mode_int(away_counts)
    home_mode = _mode_int(home_counts)

    if away_mode is not None and home_mode is not None:
        for _, data in readings:
            m = _SCORE_RX.match((data.get("score") or "").replace(":", "-"))
            if m and (int(m.group(1)), int(m.group(2))) == (away_mode, home_mode):
                return data
        synth = dict(last_data or {})
        synth["score"] = f"{away_mode}-{home_mode}"
        return synth

    if last_data and (away_mode is not None or home_mode is not None):
        m = _SCORE_RX.match((last_data.get("score") or "").replace(":", "-"))
        if m:
            a0, b0 = int(m.group(1)), int(m.group(2))
        else:
            a0, b0 = away_mode or 0, home_mode or 0
        a = away_mode if away_mode is not None else a0
        b = home_mode if home_mode is not None else b0
        patched = dict(last_data)
        patched["score"] = f"{a}-{b}"
        return patched

    return last_data or {"used_stub": True}

# ----- main entry -----
def extract_scoreboard_from_video(
    video_path: str,
    *,
    viz: bool = False,
    dx: int = 0,
    dy: int = 0,
    t: float = 0.10,
    fast_ocr: bool = True,
) -> dict:
    """
    Fast path (pytesseract + pixel ROIs) with legacy fallback if Tesseract unavailable.
    """
    attempts: List[float] = [max(0.0, t + k*0.10) for k in range(3)]
    use_fast = fast_ocr and (pytesseract is not None)

    if not use_fast:
        # Proper fallback to original extractor (no stub)
        return _legacy_extract(video_path, viz=viz, dx=dx, dy=dy, t=t)

    stable_needed = 2
    away_votes: Dict[int,int] = {}
    home_votes: Dict[int,int] = {}

    clocks: List[str] = []
    qtrs:   List[str] = []
    teams_a: List[str] = []
    teams_b: List[str] = []
    first_reading: Optional[dict] = None
    last_reading: Optional[dict] = None

    global _SB_REF_OVERRIDE
    _SB_REF_OVERRIDE = None

    for idx, ts in enumerate(attempts):
        frame_png = grab_frame_at_time(video_path, ts, out_path=f"data/tmp/video_frame_t{ts:.2f}.png")
        if viz and idx == 0:
            draw_frame_crop_outline(str(frame_png), out_path=f"data/tmp/frame_with_scorebar_box_t{ts:.2f}.png")
        scorebar_png = crop_scorebar_from_frame(str(frame_png), out_path=f"data/tmp/scorebar_t{ts:.2f}.png")

        if viz and idx == 0:
            _viz_px_rois(str(scorebar_png), out_base=f"data/tmp/scorebar_viz_t{ts:.2f}")

        fields = _extract_fields_fast_with_px_rois(str(scorebar_png))

        if idx == 0:
            reading = {
                "away_team": fields.get("away_team"),
                "home_team": fields.get("home_team"),
                "away_score": fields.get("away_score"),
                "home_score": fields.get("home_score"),
                "score": fields.get("score"),
                "clock": fields.get("clock"),
                "quarter": fields.get("quarter"),
            }
            if reading["away_team"]: teams_a.append(reading["away_team"])
            if reading["home_team"]: teams_b.append(reading["home_team"])
            if reading.get("clock"): clocks.append(reading["clock"])  # type: ignore[arg-type]
            if reading.get("quarter"): qtrs.append(reading["quarter"])  # type: ignore[arg-type]
        else:
            reading = {
                "away_team": None, "home_team": None,
                "away_score": fields.get("away_score"),
                "home_score": fields.get("home_score"),
                "score": fields.get("score"),
                "clock": None, "quarter": None,
            }

        if first_reading is None:
            first_reading = dict(reading)
        last_reading = dict(reading)

        av = _to_int_or_none(reading.get("away_score"))
        if av is not None:
            away_votes[av] = away_votes.get(av, 0) + 1
        hv = _to_int_or_none(reading.get("home_score"))
        if hv is not None:
            home_votes[hv] = home_votes.get(hv, 0) + 1

        a_mode = _mode_int(away_votes)
        b_mode = _mode_int(home_votes)

        if a_mode is not None and b_mode is not None:
            if away_votes[a_mode] >= stable_needed and home_votes[b_mode] >= stable_needed:
                out = dict(first_reading or {})
                out["away_team"] = _mode_val(teams_a) or out.get("away_team")
                out["home_team"] = _mode_val(teams_b) or out.get("home_team")
                out["away_score"] = str(a_mode)
                out["home_score"] = str(b_mode)
                out["score"] = f"{a_mode}-{b_mode}"
                clk_mode = _mode_val(clocks)
                q_mode   = _mode_val(qtrs)
                if clk_mode: out["clock"] = clk_mode
                if q_mode:   out["quarter"] = q_mode
                return out

    a_mode = _mode_int(away_votes)
    b_mode = _mode_int(home_votes)
    out = dict(first_reading or last_reading or {})
    out["away_team"] = _mode_val(teams_a) or out.get("away_team")
    out["home_team"] = _mode_val(teams_b) or out.get("home_team")
    if a_mode is not None: out["away_score"] = str(a_mode)
    if b_mode is not None: out["home_score"] = str(b_mode)
    if a_mode is not None and b_mode is not None: out["score"] = f"{a_mode}-{b_mode}"
    clk_mode = _mode_val(clocks); q_mode = _mode_val(qtrs)
    if clk_mode: out["clock"] = clk_mode
    if q_mode:   out["quarter"] = q_mode
    return out or {"used_stub": True}

if __name__ == "__main__":
    import sys
    video = sys.argv[1] if len(sys.argv) > 1 else "Madden Clip.mp4"
    data = extract_scoreboard_from_video(video, viz="--viz" in sys.argv)
    print(json.dumps(data, indent=2))
