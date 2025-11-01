from __future__ import annotations
from typing import Optional
from .schemas import ScoreboardFacts, BannerFacts

def _down_dist(sb: Optional[ScoreboardFacts]) -> str:
    if not sb or not sb.down or not sb.distance:
        return ""
    ords = {1:"1st", 2:"2nd", 3:"3rd", 4:"4th"}
    return f"{ords.get(sb.down, str(sb.down)+'th')} & {sb.distance}"

def _to_int(x):
    try:
        return int(x)
    except Exception:
        # Sometimes OCR gives strings like "+7", "−2", "7yd"
        try:
            return int(str(x).replace("yd", "").replace("+", "").replace("−", "-").strip())
        except Exception:
            return None

def one_liner(label: str, yards=None, scoreboard=None, banner=None) -> str:
    """
    Short natural line for TTS. Uses yards (if available) to add
    positive/negative/zero gain phrasing for runs (and could do the same for other labels later).
    """

    y = _to_int(yards)

    # ----- Scoring / turnovers -----
    if label == "touchdown":
        return "Touchdown!"
    if label == "interception":
        return "Picked off!"
    if label == "fumble":
        return "Ball is out!"

    # ----- Passing -----
    if label in ("complete_pass", "completion"):  # support older label name
        if y is not None and y > 0:
            return f"Completed pass for {y} yards."
        return "Completed pass."
    if label in ("incomplete_pass", "incomplete"):
        return "Incomplete pass."
    if label == "pass_attempt":
        return "Pass attempt."

    # ----- Rushing (use yards if provided) -----
    if label == "run":
        if y is None:
            return "The QB hands it off."
        if y > 0:
            return f"Run for {y} yards."
        if y == 0:
            return "No gain on the run."
        return f"Run stuffed for a loss of {abs(y)}."

    # ----- QB scramble / pressure -----
    if label == "qb_scramble":
        if y is not None and y > 0:
            return f"Quarterback scramble for {y}."
        return "Quarterback scramble."
    if label == "sack":
        return "Quarterback sacked."
    if label == "tackle_big_hit":
        return "Big hit on the tackle."

    # ----- Kicks / conversions -----
    if label == "field_goal_good":
        return "Field goal is good."
    if label == "field_goal_missed":
        return "Field goal is no good."
    if label == "extra_point_good":
        return "Extra point is good."
    if label == "two_point_good":
        return "Two-point try is successful."

    # ----- Meta / fallback -----
    if label == "no_event":
        return ""
    return label.replace("_", " ").strip()
