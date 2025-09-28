# backend/services/ocr_from_video.py
from __future__ import annotations
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import subprocess, shlex, json, re

from .ocr import extract_scoreboard_from_image, draw_boxes

# Optional OpenCV for grabbing frames
try:
    import cv2  # type: ignore
except Exception:
    cv2 = None  # fallback to ffmpeg if available

# Pillow for cropping & drawing
try:
    from PIL import Image, ImageDraw  # type: ignore
except Exception as e:
    raise RuntimeError("Pillow is required. pip install Pillow") from e

# ----- Reference full-frame -> scoreboard crop -----
# Reference full-frame: 2047x1155, scoreboard y=[1080,1155)
REF_FRAME_W, REF_FRAME_H = 2047, 1155
SCOREBAR_BOX = (0, 1080, 2047, 1155)  # (left, top, right, bottom) in reference space

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

def _scale_box_to_frame(box: Tuple[int,int,int,int], w: int, h: int) -> Tuple[int,int,int,int]:
    lx, ty, rx, by = box
    sx = w / REF_FRAME_W
    sy = h / REF_FRAME_H
    return (int(lx * sx), int(ty * sy), int(rx * sx), int(by * sy))

def crop_scorebar_from_frame(frame_path: str, out_path: str, box: Tuple[int,int,int,int] = SCOREBAR_BOX) -> Path:
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

def draw_frame_crop_outline(frame_path: str, out_path: str = "data/tmp/frame_with_scorebar_box.png") -> Path:
    """Draw a green rectangle showing where SCOREBAR_BOX maps on the full frame."""
    ip = Path(frame_path)
    op = Path(out_path)
    op.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(ip) as im:
        w, h = im.size
        lx, ty, rx, by = _scale_box_to_frame(SCOREBAR_BOX, w, h)
        vis = im.copy()
        d = ImageDraw.Draw(vis)
        d.rectangle((lx, ty, rx, by), outline="lime", width=3)
        vis.save(op)
    return op

# -------- voting helpers --------

_SCORE_RX = re.compile(r"^\s*(\d{1,2})\s*-\s*(\d{1,2})\s*$")

def _parse_score_tuple(s: Optional[str]) -> Optional[Tuple[int,int]]:
    if not s: return None
    m = _SCORE_RX.match(s)
    if not m: return None
    try:
        return (int(m.group(1)), int(m.group(2)))
    except Exception:
        return None

def _mode_int(counts: Dict[int,int]) -> Optional[int]:
    if not counts: return None
    maxc = max(counts.values())
    # choose the smallest value among ties (stable)
    winners = [k for k, c in counts.items() if c == maxc]
    return min(winners)

def extract_scoreboard_from_video(
    video_path: str,
    *,
    viz: bool = False,
    dx: int = 0,
    dy: int = 0,
    t: float = 0.10,   # start a bit after 0s to avoid overlay fade-in
) -> dict:
    """
    Pipeline with per-side score voting:
      sample frames at [t, t+0.10, t+0.20, t+0.30, t+0.40] -> crop -> OCR
      vote for AWAY and HOME scores separately, then return the earliest reading matching both.
    If viz=True, saves:
      data/tmp/video_frame_t{ts}.png
      data/tmp/frame_with_scorebar_box_t{ts}.png
      data/tmp/scorebar_t{ts}.png
      data/tmp/ocr_debug/boxes.png (from the last attempt)
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

        # Enable debug crops + boxes overlay only on the last attempt (to avoid overwriting per try)
        debug_flag = viz and (idx == len(attempts) - 1)
        result = extract_scoreboard_from_image(
            str(scorebar_png),
            debug_crops=debug_flag,
            viz_boxes_flag=debug_flag,
            dx=dx,
            dy=dy,
        )
        data = result.to_dict()
        last_data = data
        readings.append((ts, data))

        st = _parse_score_tuple(data.get("score"))
        if st:
            a, b = st
            away_counts[a] = away_counts.get(a, 0) + 1
            home_counts[b] = home_counts.get(b, 0) + 1

    # Compute modal AWAY and HOME separately
    away_mode = _mode_int(away_counts)
    home_mode = _mode_int(home_counts)

    if away_mode is not None and home_mode is not None:
        # Return earliest reading that matches both modal sides
        for ts, data in readings:
            st = _parse_score_tuple(data.get("score"))
            if st == (away_mode, home_mode):
                return data
        # If no reading exactly matches both, synthesize a coherent result from the earliest reading
        # (teams/quarter/clock are consistent across these few frames)
        synth = dict(last_data or {})
        synth["score"] = f"{away_mode}-{home_mode}"
        return synth

    # Fallback: if only one side got a mode, patch that side into the last reading
    if last_data and (away_mode is not None or home_mode is not None):
        st = _parse_score_tuple(last_data.get("score")) or (away_mode or 0, home_mode or 0)
        a = away_mode if away_mode is not None else st[0]
        b = home_mode if home_mode is not None else st[1]
        patched = dict(last_data)
        patched["score"] = f"{a}-{b}"
        return patched

    # Last resort
    return last_data or {"used_stub": True}

if __name__ == "__main__":
    import sys
    # Usage:
    #   python -m backend.services.ocr_from_video "Madden Clip.mp4" [--viz] [--dx=<int>] [--dy=<int>] [--t=<seconds>]
    video = sys.argv[1] if len(sys.argv) > 1 else "Madden Clip.mp4"
    flags = {a for a in sys.argv[2:] if a.startswith("--") and "=" not in a}
    kvs = [a for a in sys.argv[2:] if a.startswith("--") and "=" in a]
    viz = ("--viz" in flags)
    dx = dy = 0
    t = 0.10
    for kv in kvs:
        if kv.startswith("--dx="):
            try: dx = int(kv.split("=",1)[1])
            except: pass
        elif kv.startswith("--dy="):
            try: dy = int(kv.split("=",1)[1])
            except: pass
        elif kv.startswith("--t="):
            try: t = float(kv.split("=",1)[1])
            except: pass

    data = extract_scoreboard_from_video(video, viz=viz, dx=dx, dy=dy, t=t)
    print(json.dumps(data, indent=2))
