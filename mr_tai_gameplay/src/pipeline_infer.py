from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
from typing import List, Dict, Any, Union, Optional
import json

from .event_segmenter import EventSegmenter
from .gameplay_classifier import GameplayClassifier
from .schemas import Event, GameplayPrediction, ScoreboardFacts, BannerFacts
from .fuse_for_tts import fuse, choose_primary
from .utils_io import save_json

def run_inference(video_path: str,
                out_json: str,
                clip_id: Optional[str] = None,
                scoreboard_json: Optional[str] = None,
                banner_json: Optional[str] = None,
                device: str = "cpu",
                ckpt: Optional[str] = None):
    clip_id = clip_id or Path(video_path).stem

    seg = EventSegmenter()
    windows = seg.segment(video_path)
    if not windows:
        raise SystemExit("No event window detected.")
    t_start, t_end = windows[0]

    clf = GameplayClassifier(device=device, ckpt_path=ckpt)
    probs = clf.predict(video_path, t_start, t_end)
    primary = choose_primary(probs)

    event = Event(t_start=float(t_start), t_end=float(t_end),
                primary_label=primary, confidence=float(probs[primary]),
                alt_labels=[{"label": k, "p": float(v)} for k, v in sorted(probs.items(), key=lambda x: x[1], reverse=True)[:3]],
                evidence={"frames": 16})
    gp = GameplayPrediction(clip_id=clip_id, events=[event])

    sb = None
    if scoreboard_json and Path(scoreboard_json).exists():
        sb = ScoreboardFacts(**json.loads(Path(scoreboard_json).read_text()))
    bn = None
    if banner_json and Path(banner_json).exists():
        bn = BannerFacts(**json.loads(Path(banner_json).read_text()))

    fused = fuse(gp, sb, bn)

    save_json({
        "gameplay_only": {
            "clip_id": gp.clip_id,
            "events": [asdict(event)]
        },
        "tts": fused
    }, out_json)


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
    args = p.parse_args()


    run_inference(args.video, args.out, args.clip_id,
                args.scoreboard_json, args.banner_json,
                device=args.device, ckpt=args.ckpt)