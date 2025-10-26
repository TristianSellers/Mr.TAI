from __future__ import annotations
from typing import Dict, Any, Optional
from .schemas import GameplayPrediction, ScoreboardFacts, BannerFacts, LABELS
from .templates import one_liner

PRIMARY_MAP = [
"touchdown", "interception", "fumble",
"sack", "completion", "incomplete",
"qb_scramble", "run", "tackle_big_hit", "pass_attempt"
]

def choose_primary(probs):
    best, pbest = max(probs.items(), key=lambda kv: kv[1])
    # Prefer decisive outcomes ≥ 0.45
    for lbl in PRIMARY_MAP:
        if probs.get(lbl, 0.0) >= 0.45:
            return lbl
    # If 'sack' is low and 'run' is close, favor 'run'
    if probs.get("sack", 0.0) < 0.30 and (probs.get("run", 0.0) + 0.05) >= probs.get("sack", 0.0):
        return "run"
    # Margin check: if top-2 too close, default to pass_attempt (neutral)
    top2 = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)[:2]
    if len(top2) == 2 and (top2[0][1] - top2[1][1]) < 0.07:
        return "pass_attempt"
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