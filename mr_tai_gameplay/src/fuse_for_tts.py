from __future__ import annotations
from typing import Dict, Any, Optional
from .schemas import GameplayPrediction, ScoreboardFacts, BannerFacts
from .templates import one_liner

# Expanded to include scoring + meta for clarity when confident
PRIMARY_MAP = [
    "touchdown", "interception", "fumble",
    "sack", "complete_pass", "incomplete_pass",
    "qb_scramble", "run", "tackle_big_hit", "pass_attempt"
]

def choose_primary(probs: Dict[str, float]) -> str:
    best, pbest = max(probs.items(), key=lambda kv: kv[1])

    # Strong, decisive picks take priority
    for lbl in [
    "touchdown", "field_goal_good", "field_goal_missed",
    "extra_point_good", "two_point_good",
    "interception", "fumble",
    "sack", "complete_pass", "incomplete_pass",
    "run", "qb_scramble"
    ]:
        if probs.get(lbl, 0.0) >= 0.45:
            return lbl

    # Handoff bias: if completion only slightly above run, prefer run
    comp = probs.get("complete_pass", probs.get("completion", 0.0))
    runp = probs.get("run", 0.0)
    if comp < 0.35 and (comp - runp) <= 0.10:
        return "run"

    # Near-tie fallback goes to neutral "pass_attempt"
    top2 = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)[:2]
    if len(top2) == 2 and (top2[0][1] - top2[1][1]) < 0.07:
        return "pass_attempt"

    return best


def fuse(
    gameplay: GameplayPrediction,
    scoreboard: Optional[ScoreboardFacts] = None,
    banner: Optional[BannerFacts] = None
) -> Dict[str, Any]:
    lines = []
    for ev in gameplay.events:
        yards = None
        # Prefer scoreboard-derived yards if you store them there, else banner
        if banner and getattr(banner, "yard_gain", None) is not None:
            yards = banner.yard_gain
        # TODO: If later you compute yards from scoreboard deltas, set `yards` here.
        text = one_liner(ev.primary_label, yards, scoreboard, banner)
        t_say = ev.t_end + 0.30  # speak shortly after the tackle/finish
        lines.append({"t_say": round(t_say, 2), "text": text})
    return {"clip_id": gameplay.clip_id, "lines": lines}
