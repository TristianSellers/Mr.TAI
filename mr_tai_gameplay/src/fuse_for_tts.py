from __future__ import annotations
from typing import Dict, Any, Optional
from .schemas import GameplayPrediction, ScoreboardFacts, BannerFacts, LABELS
from .templates import one_liner

PRIMARY_MAP = [
"touchdown", "interception", "fumble",
"sack", "completion", "incomplete",
"qb_scramble", "run", "tackle_big_hit", "pass_attempt"
]

def choose_primary(probs: Dict[str, float]) -> str:
    # Greedy by a custom priority so decisive outcomes beat generic attempts
    best = max(probs, key=lambda k: probs[k])
    # If a decisive label is reasonably high, prefer it
    for lbl in PRIMARY_MAP:
        if probs.get(lbl, 0.0) >= 0.40:
            return lbl
    return best


def fuse(gameplay: GameplayPrediction,
        scoreboard: Optional[ScoreboardFacts] = None,
        banner: Optional[BannerFacts] = None) -> Dict[str, Any]:
    lines = []
    for ev in gameplay.events:
        yards = banner.yard_gain if banner else None
        text = one_liner(ev.primary_label, yards, scoreboard, banner)
        t_say = ev.t_end + 0.30 # speak shortly after the tackle
        lines.append({"t_say": round(t_say, 2), "text": text})
    return {"clip_id": gameplay.clip_id, "lines": lines}