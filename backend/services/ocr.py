# backend/services/ocr.py
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Optional, Dict, TYPE_CHECKING, Any, List, Tuple, Set
import os, re, logging
from pathlib import Path

if TYPE_CHECKING:
    from PIL.Image import Image as _PILImageType
    PILImageLike = _PILImageType
else:
    PILImageLike = Any  # type: ignore[assignment]

log = logging.getLogger(__name__)

# ---- Pillow resample constant (Pylance-safe, Pillow <10 and >=10) ----
try:
    from PIL import Image as _PILImageModule  # type: ignore
    if hasattr(_PILImageModule, "Resampling"):  # Pillow >= 10
        RESAMPLE_LANCZOS = _PILImageModule.Resampling.LANCZOS
    else:  # Pillow < 10, numeric fallback (ANTIALIAS/LANCZOS == 1)
        RESAMPLE_LANCZOS = 1
except Exception:
    RESAMPLE_LANCZOS = None

# --------------------------- Model ---------------------------

@dataclass
class OCRScoreboard:
    home_team: Optional[str] = None
    away_team: Optional[str] = None
    score:     Optional[str] = None   # "away-home"
    quarter:   Optional[str] = None   # "1st", "OT"
    clock:     Optional[str] = None   # "2:36"
    # Debug helpers
    ocr_text:  Optional[str] = None
    used_stub: bool = False

    def to_dict(self) -> Dict[str, Optional[str]]:
        return asdict(self)

# --------------------------- Reference layout (pixels) ---------------------------

REF_W, REF_H = 2047, 75
REF_BOXES = {
    "away_team":  (550, 0, 650, 50),
    "home_team":  (950, 0, 1100, 50),  # widened for safety
    "quarter":    (1350, 0, 1430, 50),
    "away_score": (700, 0, 775, 72),
    "home_score": (840, 0, 900, 72),
    "clock":      (1430, 0, 1525, 50),
}

# --------------------------- Utilities ---------------------------

TEAM_ABBREVS = {
    "ARI","ATL","BAL","BUF","CAR","CHI","CIN","CLE","DAL","DEN","DET",
    "GB","HOU","IND","JAX","KC","LV","LAC","LAR","MIA","MIN","NE","NO",
    "NYG","NYJ","PHI","PIT","SEA","SF","TB","TEN","WSH",
}

def _lev1(a: str, b: str) -> int:
    if a == b: return 0
    if abs(len(a)-len(b)) > 2: return 3
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b))
    if len(a)+1 == len(b):
        for i in range(len(b)):
            if a == b[:i] + b[i+1:]: return 1
        return 2
    if len(b)+1 == len(a):
        for i in range(len(a)):
            if b == a[:i] + a[i+1:]: return 1
        return 2
    return 2

def _nearest_team_abbrev(tok: Optional[str], prefer: Set[str] | None = None) -> Optional[str]:
    if not tok: return None
    t = tok.strip().upper()
    if t in TEAM_ABBREVS:
        return t
    best = None
    best_d = 3
    for cand in TEAM_ABBREVS:
        if abs(len(cand) - len(t)) > 1:
            continue
        d = _lev1(t, cand)
        if (d < best_d) or (d == best_d and prefer and cand in prefer):
            best, best_d = cand, d
    if best is not None and best_d <= 1:
        return best
    return t

# Quarter tokens & regexes
_QUARTER_MAP_IN  = {"1ST":"1st","2ND":"2nd","3RD":"3rd","4TH":"4th","OT":"OT"}
_Q_FROM_QDIGIT   = {"1":"1st","2":"2nd","3":"3rd","4":"4th"}
_QUARTER_RX      = re.compile(r"\b(Q[1-4]|OT|1ST|2ND|3RD|4TH)\b", re.I)
_QUARTER_WORD_RX = re.compile(r"\b([1-4])\s*(?:ST|ND|RD|TH)\b", re.I)
_CLOCK_RX        = re.compile(r"\b([0-5]?\d:[0-5]\d)\b")
_ALLOWED_Q_TOKENS = ["1ST","2ND","3RD","4TH","OT","Q1","Q2","Q3","Q4","1st","2nd","3rd","4th","ot","q1","q2","q3","q4"]

def _lev(a: str, b: str) -> int:
    la, lb = len(a), len(b)
    dp = list(range(lb+1))
    for i in range(1, la+1):
        prev = dp[0]
        dp[0] = i
        ca = a[i-1]
        for j in range(1, lb+1):
            temp = dp[j]
            cost = 0 if ca == b[j-1] else 1
            dp[j] = min(dp[j] + 1, dp[j-1] + 1, prev + cost)
            prev = temp
    return dp[lb]

def _snap_quarter_token(s: str) -> Optional[str]:
    if not s:
        return None
    s = s.strip()
    best = None
    best_d = 99
    for tok in _ALLOWED_Q_TOKENS:
        d = _lev(s, tok)
        if d < best_d:
            best, best_d = tok, d
    if best is not None and best_d <= 2:
        up = best.upper()
        if up in _QUARTER_MAP_IN: return _QUARTER_MAP_IN[up]
        if up.startswith("Q") and len(up) == 2 and up[1] in "1234":
            return _Q_FROM_QDIGIT[up[1]]
    return None

def _norm_quarter(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    s_up = re.sub(r"\s+", "", s.upper())
    if s_up.startswith(("I", "L")):
        s_up = "1" + s_up[1:]
    s_up = (s_up
        .replace("IST","1ST")
        .replace("2ST","2ND")
        .replace("3ST","3RD")
        .replace("4ST","4TH")
    )
    if s_up in {"1ST","2ND","3RD","4TH","OT"}:
        return {"1ST":"1st","2ND":"2nd","3RD":"3rd","4TH":"4th","OT":"OT"}[s_up]
    m = re.match(r"Q([1-4])", s_up)
    if m:
        return _Q_FROM_QDIGIT[m.group(1)]
    m2 = re.search(r"\b([1-4])\s*(?:ST|ND|RD|TH)\b", s, flags=re.I)
    if m2:
        return _Q_FROM_QDIGIT[m2.group(1)]
    snap = _snap_quarter_token(s)
    if snap:
        return snap
    if s_up == "OT":
        return "OT"
    return None

def _best_token(txt: str, pat: re.Pattern[str]) -> Optional[str]:
    if not txt: return None
    m = pat.search(txt)
    return m.group(1) if m else None

def _first_abbrev(txt: str) -> Optional[str]:
    toks = re.findall(r"\b[A-Z]{2,4}\b", (txt or "").upper())
    return toks[0] if toks else None

def _clean_team(s: Optional[str]) -> Optional[str]:
    if not s: return None
    t = s.strip().upper()
    if not re.fullmatch(r"[A-Z]{2,4}", t): return None
    if t in {"WJ"}: return None
    return t

# --------------------------- OCR core ---------------------------

def _ocr_text(img: Any, config: str) -> str:
    try:
        import pytesseract  # type: ignore
        txt = pytesseract.image_to_string(img, lang="eng", config=config) or ""
        return txt.strip()
    except Exception:
        return ""

def _ocr_try_many(img: PILImageLike, cfgs: List[str]) -> str:
    best = ""
    for cfg in cfgs:
        t = _ocr_text(img, cfg)
        if len(t) > len(best):
            best = t
    return best

# --------------------------- Preprocessing ---------------------------

def _preprocess_banner(img: PILImageLike) -> PILImageLike:
    """Generic upscale+sharpen for thin HUD fonts."""
    try:
        from PIL import ImageOps, ImageFilter  # type: ignore
        w, h = img.size
        if RESAMPLE_LANCZOS is not None:
            big = img.resize((w*3, h*3), resample=RESAMPLE_LANCZOS)
        else:
            big = img.resize((w*3, h*3))
        g = ImageOps.grayscale(big)
        g = ImageOps.autocontrast(g)
        g = g.filter(ImageFilter.UnsharpMask(radius=1.0, percent=150, threshold=2))
        return g
    except Exception:
        return img

def _prep_digits_mask(img: PILImageLike, invert: bool = False) -> PILImageLike:
    """
    White + yellow/amber text isolation -> clean binary -> upscale.
    Falls back to banner preprocess if cv2 unavailable.
    """
    try:
        import numpy as np, cv2  # type: ignore
        arr = np.array(img.convert("RGB"))
        hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)

        # White-ish text on dark HUD
        white_lo = np.array([0,   0, 180], dtype=np.uint8)
        white_hi = np.array([179, 40, 255], dtype=np.uint8)
        mask_white = cv2.inRange(hsv, white_lo, white_hi)

        # Yellow/amber digits (common HUD tint)
        yellow_lo = np.array([15,  60, 170], dtype=np.uint8)
        yellow_hi = np.array([40, 255, 255], dtype=np.uint8)
        mask_yellow = cv2.inRange(hsv, yellow_lo, yellow_hi)

        mask = cv2.bitwise_or(mask_white, mask_yellow)
        if invert:
            mask = cv2.bitwise_not(mask)

        mask = cv2.GaussianBlur(mask, (3,3), 0)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        mask = cv2.resize(mask, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

        from PIL import Image  # type: ignore
        return Image.fromarray(mask)
    except Exception:
        return _preprocess_banner(img)

# --------------------------- Scaling + Cropping ---------------------------

def scale_box(box: Tuple[int,int,int,int], w: int, h: int,
              ref_w: int = REF_W, ref_h: int = REF_H,
              dx: int = 0, dy: int = 0) -> Tuple[int,int,int,int]:
    lx, ty, rx, by = box
    sx = w / ref_w
    sy = h / ref_h
    return (int(lx * sx + dx), int(ty * sy + dy), int(rx * sx + dx), int(by * sy + dy))

def crop_regions(img: PILImageLike, dx: int = 0, dy: int = 0) -> Dict[str, PILImageLike]:
    w, h = img.size
    crops: Dict[str, PILImageLike] = {}
    for name, ref_box in REF_BOXES.items():
        box = scale_box(ref_box, w, h, dx=dx, dy=dy)
        crops[name] = img.crop(box)
    return crops

def draw_boxes(img: PILImageLike, out_path: Path, dx: int = 0, dy: int = 0) -> None:
    from PIL import ImageDraw  # type: ignore
    vis = img.copy()
    d = ImageDraw.Draw(vis)
    for name, ref_box in REF_BOXES.items():
        w, h = img.size
        box = scale_box(ref_box, w, h, dx=dx, dy=dy)
        d.rectangle(box, outline="red", width=2)
        d.text((box[0] + 2, box[1] + 2), name, fill="red")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vis.save(out_path)

# --------------------------- Tesseract wrappers ---------------------------

def _ocr_digits_core(img: PILImageLike, whitelist: str) -> str:
    # numeric mode biases to digits (reduces O/0/3 swaps)
    cfgs = [
        f"--oem 1 --psm 10 -c tessedit_char_whitelist={whitelist} -c classify_bln_numeric_mode=1",
        f"--oem 1 --psm 7  -c tessedit_char_whitelist={whitelist} -c classify_bln_numeric_mode=1",
        f"--oem 1 --psm 6  -c tessedit_char_whitelist={whitelist} -c classify_bln_numeric_mode=1",
        f"--oem 1 --psm 13 -c tessedit_char_whitelist={whitelist} -c classify_bln_numeric_mode=1",
    ]
    return _ocr_try_many(img, cfgs)

def _ocr_digits(img: PILImageLike, allow_colon: bool = False, allow_minus: bool = False) -> str:
    wl = "0123456789"
    if allow_colon: wl += ":"
    if allow_minus: wl += "-"
    t = _ocr_digits_core(_prep_digits_mask(img, invert=False), wl)
    if not t.strip():
        t = _ocr_digits_core(_prep_digits_mask(img, invert=True), wl)
    return t

def _ocr_letters(img: PILImageLike) -> str:
    g = _preprocess_banner(img)
    cfgs = [
        "--oem 1 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        "--oem 1 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        "--oem 1 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    ]
    return (_ocr_try_many(g, cfgs) or "").upper()

# --------------------------- Core extraction helpers ---------------------------

def _expand_box(box: Tuple[int,int,int,int], pad: int, max_w: int, max_h: int) -> Tuple[int,int,int,int]:
    lx, ty, rx, by = box
    return (max(0, lx - pad), max(0, ty - pad), min(max_w, rx + pad), min(max_h, by + pad))

def _recover_quarter_from_clock(base: PILImageLike, dx: int, dy: int) -> Optional[str]:
    """If quarter box fails, scan a wider strip left of the clock box and combine digits+suffix."""
    w, h = base.size
    c_lx, c_ty, c_rx, c_by = scale_box(REF_BOXES["clock"], w, h, dx=dx, dy=dy)
    lx = max(0, c_lx - 220)
    neighbor_box = (lx, c_ty, c_lx - 4, c_by)
    region = base.crop(_expand_box(neighbor_box, 6, w, h))
    q = _quarter_from_parts(region)
    if q:
        return q
    for cfg in (
        "--oem 1 --psm 8 -c tessedit_char_whitelist=Qq01234OoTtSsNnDdRrTtHh",
        "--oem 1 --psm 7 -c tessedit_char_whitelist=Qq01234OoTtSsNnDdRrTtHh",
        "--oem 1 --psm 10 -c tessedit_char_whitelist=Qq01234OoTtSsNnDdRrTtHh",
    ):
        q_raw = _ocr_text(region, cfg)
        norm = _norm_quarter(q_raw)
        if norm:
            return norm
    return None

# -------- NEW: hole-shape detector for 0 vs 2/3 --------

def _digit_has_hole(img: PILImageLike) -> Optional[bool]:
    """
    Returns True if the digit crop clearly contains an internal hole (0/6/8/9), False if not (2/3/5/7/1/4),
    or None if we can't tell.
    """
    try:
        import numpy as np, cv2  # type: ignore
        bin_img = _prep_digits_mask(img, invert=False)
        arr = np.array(bin_img)
        if arr.ndim == 3:
            arr = arr[:, :, 0]
        # Ensure binary 0/255
        _, bw = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        # Find contours with hierarchy so we can count holes (child contours)
        contours, hierarchy = cv2.findContours(bw, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is None:
            return False
        holes = 0
        for i, h in enumerate(hierarchy[0]):
            parent_idx = h[3]
            if parent_idx != -1:
                holes += 1
        return holes >= 1
    except Exception:
        return None

# -------- robust score via micro-jitter voting + hole override --------

def _read_score_microvote(base: PILImageLike, ref_key: str, dx: int, dy: int,
                          debug_dir: Optional[Path]) -> Optional[str]:
    """
    Vote the score by sampling tiny jitters around the box.
    Offsets: x in {-2,0,2}, y in {-1,0,1}; pads in {0,3}.
    Applies a shape override: if OCR says {2,3} but the glyph has a hole -> count as 0.
    """
    w, h = base.size
    box = scale_box(REF_BOXES[ref_key], w, h, dx=dx, dy=dy)
    lx0, ty0, rx0, by0 = box

    counts: Dict[int, int] = {}
    def _bump(n: int, weight: int = 1): counts[n] = counts.get(n, 0) + weight

    for pad in (0, 3):
        for ox in (-2, 0, 2):
            for oy in (-1, 0, 1):
                lx = max(0, lx0 + ox); rx = min(w, rx0 + ox)
                ty = max(0, ty0 + oy); by = min(h, by0 + oy)
                crop = base.crop(_expand_box((lx,ty,rx,by), pad, w, h))

                # OCR read (single/two-digit)
                txt = _ocr_digits(crop)
                norm = txt.replace("O","0").replace("o","0").replace("D","0").replace("Q","0")
                m = re.search(r"\d{1,2}", norm)

                # Shape hint: does it have a hole?
                hole = _digit_has_hole(crop)

                if m:
                    val = int(m.group(0))
                    # Only apply override for single-digit readings where 2/3 are common mistakes for 0
                    if val in (2,3) and len(m.group(0)) == 1 and hole is True:
                        _bump(0, weight=2)  # strong vote for 0 if hole exists
                    else:
                        _bump(val)
                    continue

                # If OCR failed but the shape says "has hole", it's very likely 0 in this HUD context
                if hole is True:
                    _bump(0)
                    continue

                # As a last resort, try preprocessed grayscale OCR
                txt2 = _ocr_digits_core(_preprocess_banner(crop), "0123456789")
                m2 = re.search(r"\d{1,2}", txt2.replace("O","0").replace("o","0").replace("D","0").replace("Q","0"))
                if m2:
                    _bump(int(m2.group(0)))

    if counts:
        maxc = max(counts.values())
        # choose modal; break ties by smaller value (stable)
        best_n = min([k for k, c in counts.items() if c == maxc])
        return str(best_n)

    return None

def _extract_single_score(crop: PILImageLike, base: PILImageLike, ref_key: str, dx: int, dy: int,
                          debug_dir: Optional[Path]) -> Optional[str]:
    """Backup path if micro-vote yields nothing. Applies the same hole override."""
    def _norm_digits(s: Optional[str]) -> Optional[str]:
        if not s: return None
        return s.replace("O", "0").replace("o", "0").replace("D", "0").replace("Q", "0")

    # OCR
    txt = _ocr_digits(crop)
    m = re.search(r"\d{1,2}", _norm_digits(txt) or "")
    hole = _digit_has_hole(crop)

    if m:
        val = int(m.group(0))
        if val in (2,3) and len(m.group(0)) == 1 and hole is True:
            return "0"
        return str(int(m.group(0)))

    # Preprocessed OCR
    txt2 = _ocr_digits_core(_preprocess_banner(crop), "0123456789")
    m = re.search(r"\d{1,2}", _norm_digits(txt2) or "")
    if m:
        val = int(m.group(0))
        if val in (2,3) and len(m.group(0)) == 1 and hole is True:
            return "0"
        return str(int(m.group(0)))

    # Expand box + retry
    w, h = base.size
    box = scale_box(REF_BOXES[ref_key], w, h, dx=dx, dy=dy)
    exp_box = _expand_box(box, 6, w, h)
    exp_crop = base.crop(exp_box)
    if debug_dir:
        try:
            exp_crop.save(debug_dir / f"{ref_key}_expanded.png")
        except Exception:
            pass
    txt3 = _ocr_digits(exp_crop)
    m = re.search(r"\d{1,2}", _norm_digits(txt3) or "")
    hole2 = _digit_has_hole(exp_crop)
    if m:
        val = int(m.group(0))
        if val in (2,3) and len(m.group(0)) == 1 and (hole2 is True or hole is True):
            return "0"
        return str(int(m.group(0)))

    # Last resort: single-char forced read
    txt4 = _ocr_digits_core(_prep_digits_mask(exp_crop, invert=False), "0123456789")
    m = re.search(r"\d", _norm_digits(txt4) or "")
    if m:
        val = int(m.group(0))
        hole3 = _digit_has_hole(exp_crop)
        if val in (2,3) and hole3 is True:
            return "0"
        return str(int(m.group(0)))

    return None

def _quarter_from_parts(region: PILImageLike) -> Optional[str]:
    """Combine digits-only + suffix-only OCR to form 1st/2nd/3rd/4th."""
    d_txt = _ocr_digits(region)
    digit = None
    m = re.search(r"[1-4]", d_txt or "")
    if not m:
        m = re.search(r"[IlL]", d_txt or "")
        if m:
            digit = "1"
    else:
        digit = m.group(0)
    s_txt = _ocr_text(region, "--oem 1 --psm 7 -c tessedit_char_whitelist=STNDRDTHstndrdth")
    suf = None
    ms = re.search(r"(ST|ND|RD|TH)", (s_txt or "").upper())
    if ms:
        suf = ms.group(1)
    if digit and suf:
        return _norm_quarter(f"{digit}{suf}")
    if digit:
        return _norm_quarter(f"Q{digit}")
    snap = _snap_quarter_token((d_txt or "") + " " + (s_txt or ""))
    if snap:
        return snap
    return None

def _ocr_quarter(q_crop: PILImageLike, base: PILImageLike, dx: int, dy: int) -> Optional[str]:
    for cfg in (
        "--oem 1 --psm 8 -c tessedit_char_whitelist=Qq01234OoTtSsNnDdRrTtHh",
        "--oem 1 --psm 7 -c tessedit_char_whitelist=Qq01234OoTtSsNnDdRrTtHh",
        "--oem 1 --psm 10 -c tessedit_char_whitelist=Qq01234OoTtSsNnDdRrTtHh",
        "--oem 1 --psm 6 -c tessedit_char_whitelist=Qq01234OoTtSsNnDdRrTtHh",
    ):
        q_raw = _ocr_text(q_crop, cfg)
        norm = _norm_quarter(q_raw)
        if norm:
            return norm
        snap = _snap_quarter_token(q_raw)
        if snap:
            return snap
    for masked in (_prep_digits_mask(q_crop, invert=False), _prep_digits_mask(q_crop, invert=True)):
        for cfg in (
            "--oem 1 --psm 8 -c tessedit_char_whitelist=Qq01234OoTtSsNnDdRrTtHh",
            "--oem 1 --psm 7 -c tessedit_char_whitelist=Qq01234OoTtSsNnDdRrTtHh",
            "--oem 1 --psm 10 -c tessedit_char_whitelist=Qq01234OoTtSsNnDdRrTtHh",
        ):
            q_raw = _ocr_text(masked, cfg)
            norm = _norm_quarter(q_raw)
            if norm:
                return norm
            snap = _snap_quarter_token(q_raw)
            if snap:
                return snap
    comb = _quarter_from_parts(q_crop)
    if comb:
        return comb
    w, h = base.size
    q_box = scale_box(REF_BOXES["quarter"], w, h, dx=dx, dy=dy)
    q_box = _expand_box(q_box, 12, w, h)
    q_exp = base.crop(q_box)
    comb2 = _quarter_from_parts(q_exp)
    if comb2:
        return comb2
    for cfg in ("--oem 1 --psm 8 -c tessedit_char_whitelist=Qq01234OoTtSsNnDdRrTtHh",
                "--oem 1 --psm 7 -c tessedit_char_whitelist=Qq01234OoTtSsNnDdRrTtHh"):
        q_raw = _ocr_text(q_exp, cfg)
        norm = _norm_quarter(q_raw)
        if norm:
            return norm
        snap = _snap_quarter_token(q_raw)
        if snap:
            return snap
    return _recover_quarter_from_clock(base, dx, dy)

def _choose_home_by_seen_tokens(away: Optional[str], home: Optional[str], seen: Set[str]) -> Optional[str]:
    """
    If the global OCR across the strip ('seen') contains exactly two team tokens
    and one of them is the away team, pick the other as home.
    """
    if not away or not seen:
        return home
    teams = {t for t in seen if t in TEAM_ABBREVS}
    if away in teams and len(teams) == 2:
        other = (teams - {away})
        if len(other) == 1:
            return next(iter(other))
    return home

# --------------------------- Core extraction ---------------------------

def _extract_from_crops(
    crops: Dict[str, PILImageLike],
    base: PILImageLike,
    dx: int,
    dy: int,
    debug_dir: Path | None = None,
    prefer_tokens: Set[str] | None = None,
) -> Dict[str, Optional[str]]:
    if debug_dir:
        for k, im in crops.items():
            try:
                debug_dir.mkdir(parents=True, exist_ok=True)
                im.save(debug_dir / f"{k}.png")
            except Exception:
                pass

    # Teams
    away_raw = _clean_team(_first_abbrev(_ocr_letters(crops["away_team"])))
    home_raw = _clean_team(_first_abbrev(_ocr_letters(crops["home_team"])))
    away = _nearest_team_abbrev(away_raw, prefer=prefer_tokens)
    home = _nearest_team_abbrev(home_raw, prefer=prefer_tokens)
    home = _choose_home_by_seen_tokens(away, home, prefer_tokens or set())

    # Scores (explicit away-home) with micro-vote + hole override
    away_num = _read_score_microvote(base, "away_score", dx, dy, debug_dir) \
               or _extract_single_score(crops["away_score"], base, "away_score", dx, dy, debug_dir)
    home_num = _read_score_microvote(base, "home_score", dx, dy, debug_dir) \
               or _extract_single_score(crops["home_score"], base, "home_score", dx, dy, debug_dir)
    score = f"{away_num}-{home_num}" if (away_num is not None and home_num is not None) else None

    # Quarter & clock
    quarter = _ocr_quarter(crops["quarter"], base, dx, dy)
    clock = _best_token(_ocr_digits(crops["clock"], allow_colon=True), _CLOCK_RX)

    return {"home": home, "away": away, "score": score, "clock": clock, "quarter": quarter}

# --------------------------- Public API ---------------------------

def extract_scoreboard_from_image(
    image_path: str | os.PathLike[str],
    debug_crops: bool = False,
    viz_boxes_flag: bool = False,
    dx: int = 0,
    dy: int = 0,
) -> OCRScoreboard:
    try:
        from PIL import Image  # type: ignore
        import pytesseract  # type: ignore
    except Exception:
        return OCRScoreboard(used_stub=True)

    tess_cmd = os.getenv("TESSERACT_CMD")
    if tess_cmd:
        pytesseract.pytesseract.tesseract_cmd = tess_cmd

    p = Path(image_path)
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {p}")

    try:
        with Image.open(p) as im:
            if viz_boxes_flag:
                draw_boxes(im, Path("data/tmp/ocr_debug/boxes.png"), dx=dx, dy=dy)

            base = _preprocess_banner(im)

            general = _ocr_text(
                base,
                "--oem 1 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789: "
            )
            prefer_tokens: Set[str] = set(re.findall(r"\b[A-Z]{2,4}\b", general or "")) & TEAM_ABBREVS

            crops = crop_regions(base, dx=dx, dy=dy)
            dbg_dir = Path("data/tmp/ocr_debug") if debug_crops else None

            fields = _extract_from_crops(crops, base, dx, dy, debug_dir=dbg_dir, prefer_tokens=prefer_tokens)

            if not all([fields["home"], fields["away"], fields["score"], fields["quarter"], fields["clock"]]):
                fields["quarter"] = fields["quarter"] or _norm_quarter(
                    _best_token(general, _QUARTER_RX) or _best_token(general, _QUARTER_WORD_RX)
                ) or _snap_quarter_token(general)
                fields["clock"] = fields["clock"] or _best_token(general, _CLOCK_RX)

                if not fields["score"]:
                    nums = re.findall(r"\d{1,2}", general.replace("O","0").replace("o","0"))
                    if len(nums) >= 2:
                        a = str(int(nums[0])); b = str(int(nums[1]))
                        fields["score"] = f"{a}-{b}"

                if prefer_tokens:
                    if fields["home"] and fields["home"] not in prefer_tokens:
                        cands = [t for t in prefer_tokens if t != fields.get("away")]
                        if len(cands) == 1:
                            fields["home"] = cands[0]
                    if fields["away"] and fields["away"] not in prefer_tokens:
                        cands = [t for t in prefer_tokens if t != fields.get("home")]
                        if len(cands) == 1:
                            fields["away"] = cands[0]

            fields["away"] = _nearest_team_abbrev(fields.get("away"), prefer=prefer_tokens)
            fields["home"] = _nearest_team_abbrev(fields.get("home"), prefer=prefer_tokens)
            fields["home"] = _choose_home_by_seen_tokens(fields["away"], fields["home"], prefer_tokens or set())

            return OCRScoreboard(
                home_team=fields["home"],
                away_team=fields["away"],
                score=fields["score"],
                quarter=fields["quarter"],
                clock=fields["clock"],
                ocr_text=str(fields),
                used_stub=False,
            )
    except Exception:
        return OCRScoreboard(used_stub=True)

# --------------------------- CLI ---------------------------

if __name__ == "__main__":
    import sys, json
    # Usage:
    #   python -m backend.services.ocr <image> [--debug] [--viz] [--dx INT] [--dy INT]
    img = sys.argv[1] if len(sys.argv) > 1 else "data/tmp/frame.png"
    flags = set(a for a in sys.argv[2:] if a.startswith("--") and "=" not in a)
    kvs = [a for a in sys.argv[2:] if a.startswith("--") and "=" in a]
    debug = ("--debug" in flags)
    viz   = ("--viz" in flags)
    dx = dy = 0
    for kv in kvs:
        if kv.startswith("--dx="):
            try: dx = int(kv.split("=",1)[1])
            except: pass
        if kv.startswith("--dy="):
            try: dy = int(kv.split("=",1)[1])
            except: pass

    res = extract_scoreboard_from_image(img, debug_crops=debug, viz_boxes_flag=viz, dx=dx, dy=dy)
    print(json.dumps(res.to_dict(), indent=2))
