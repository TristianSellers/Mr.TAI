from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
from typing import Optional, List, Dict, Any
import json

from .event_segmenter import EventSegmenter
from .gameplay_classifier import GameplayClassifier
from .schemas import Event, GameplayPrediction, ScoreboardFacts, BannerFacts
from .fuse_for_tts import fuse, choose_primary
from .script_builder import build_script
from .utils_io import save_json

def run_inference(
    video_path: str,
    out_json: str,
    clip_id: Optional[str] = None,
    scoreboard_json: Optional[str] = None,
    banner_json: Optional[str] = None,
    device: str = "cpu",
    ckpt: Optional[str] = None,
):
    """
    Single-window inference (kept for backward compatibility).
    Uses flow-based segment() to find one play and fuses to a one-liner TTS.
    """
    clip_id = clip_id or Path(video_path).stem

    seg = EventSegmenter()
    windows = seg.segment(video_path)
    if not windows:
        raise ValueError("No event window detected or video unreadable.")
    t_start, t_end = windows[0]

    clf = GameplayClassifier(device=device, ckpt_path=ckpt)
    NUM_FRAMES = 32
    probs = clf.predict(video_path, t_start, t_end, num_frames=NUM_FRAMES)
    primary = choose_primary(probs)

    event = Event(
        t_start=float(t_start),
        t_end=float(t_end),
        primary_label=primary,
        confidence=float(probs[primary]),
        alt_labels=[
            {"label": k, "p": float(v)}
            for k, v in sorted(probs.items(), key=lambda x: x[1], reverse=True)[:3]
        ],
        evidence={"frames": NUM_FRAMES},
    )
    gp = GameplayPrediction(clip_id=clip_id, events=[event])

    sb = None
    if scoreboard_json and Path(scoreboard_json).exists():
        sb = ScoreboardFacts(**json.loads(Path(scoreboard_json).read_text()))
    bn = None
    if banner_json and Path(banner_json).exists():
        bn = BannerFacts(**json.loads(Path(banner_json).read_text()))

    fused = fuse(gp, sb, bn)

    save_json(
        {
            "gameplay_only": {
                "clip_id": gp.clip_id,
                "events": [asdict(event)],
            },
            "tts": fused,
        },
        out_json,
    )


def run_inference_multi(
    video_path: str,
    out_json: str,
    clip_id: Optional[str] = None,
    scoreboard_json: Optional[str] = None,
    banner_json: Optional[str] = None,
    device: str = "cpu",
    ckpt: Optional[str] = None,
    frames_per_window: int = 32,
):
    """
    Multi-segment inference:
      - Splits the full clip into ~5s windows (segment_all)
      - Classifies each window (gameplay-only)
      - Builds a timed script with event lines and gap fillers
    """
    clip_id = clip_id or Path(video_path).stem

    seg = EventSegmenter()
    windows = seg.segment_all(video_path)
    if not windows:
        raise ValueError("No event windows detected or video unreadable.")

    clf = GameplayClassifier(device=device, ckpt_path=ckpt)

    events: List[Event] = []
    for (t_start, t_end) in windows:
        probs = clf.predict(video_path, t_start, t_end, num_frames=frames_per_window)
        primary = choose_primary(probs)
        ev = Event(
            t_start=float(t_start),
            t_end=float(t_end),
            primary_label=primary,
            confidence=float(probs[primary]),
            alt_labels=[
                {"label": k, "p": float(v)}
                for k, v in sorted(probs.items(), key=lambda x: x[1], reverse=True)[:3]
            ],
            evidence={"frames": frames_per_window},
        )
        events.append(ev)

    gp = GameplayPrediction(clip_id=clip_id, events=events)

    # Optional (kept for future use): scoreboard & banner facts
    sb = None
    if scoreboard_json and Path(scoreboard_json).exists():
        sb = ScoreboardFacts(**json.loads(Path(scoreboard_json).read_text()))
    bn = None
    if banner_json and Path(banner_json).exists():
        bn = BannerFacts(**json.loads(Path(banner_json).read_text()))

    # Build script with timing; later we can inject yards from sb/bn per window
    tts_payload: Dict[str, Any] = {"clip_id": gp.clip_id, **build_script([asdict(e) for e in events])}

    save_json(
        {
            "gameplay_only": {
                "clip_id": gp.clip_id,
                "events": [asdict(e) for e in events],
            },
            "tts": tts_payload,
        },
        out_json,
    )


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("video")
    p.add_argument("--out", default="out/prediction.json")
    p.add_argument("--clip_id", default=None)
    p.add_argument("--scoreboard_json", default=None)
    p.add_argument("--banner_json", default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--ckpt", default=None)
    p.add_argument("--multi", action="store_true", help="Use multi-segment inference")
    p.add_argument("--frames_per_window", type=int, default=32)
    args = p.parse_args()

    if args.multi:
        run_inference_multi(
            args.video,
            args.out,
            args.clip_id,
            args.scoreboard_json,
            args.banner_json,
            device=args.device,
            ckpt=args.ckpt,
            frames_per_window=args.frames_per_window,
        )
    else:
        run_inference(
            args.video,
            args.out,
            args.clip_id,
            args.scoreboard_json,
            args.banner_json,
            device=args.device,
            ckpt=args.ckpt,
        )
