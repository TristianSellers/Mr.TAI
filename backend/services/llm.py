# backend/services/llm.py
from textwrap import shorten

def generate_commentary(context: dict, tone: str = "play-by-play") -> str:
    away = (context.get("away_team") or "the visitors").strip()
    home = (context.get("home_team") or "the home side").strip()
    quarter = (context.get("quarter") or "Q?").strip()
    clock = (context.get("clock") or "?:??").strip()
    score = (context.get("score") or "—").strip()

    # short, hypey but not cheesy; under ~120 words later enforced by shorten
    line = (
        f"{quarter}, {clock} on the clock — {away} against {home}. "
        f"Here we go! The snap... pressure coming... throws — CAUGHT! "
        f"Breaking free to the pylon — TOUCHDOWN! "
        f"Momentum swings and the score stands {score}. "
        f"{away} just flipped this game on its head."
    )
    return shorten(line, width=750, placeholder="…")
