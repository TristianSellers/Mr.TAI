from __future__ import annotations
from typing import Optional
from .schemas import ScoreboardFacts, BannerFacts


def one_liner(primary: str,
            yards: Optional[int],
            sb: Optional[ScoreboardFacts],
            banner: Optional[BannerFacts]) -> str:
    # Minimal, deterministic phrasing. You can later swap with prompting.
    down_dist = ""
    if sb and sb.down and sb.distance:
        down_dist = f"{sb.down}rd & {sb.distance} " if sb.down==3 else \
        f"{sb.down}th & {sb.distance} " if sb.down==4 else \
        f"{sb.down}nd & {sb.distance} " if sb.down==2 else \
        f"1st & {sb.distance} "
    yard_txt = f"for {yards} yards " if (yards is not None and yards>0) else ""

    if primary == "run":
        return f"{down_dist}hand-off up the middle {yard_txt}to move the chains.".strip()
    if primary == "pass_attempt":
        return f"{down_dist}quarterback drops and fires; pass underway.".strip()
    if primary == "completion":
        fd = "for a first down" if banner and banner.result == "first_down" else ""
        return f"{down_dist}quick strike {yard_txt}{fd}.".strip()
    if primary == "incomplete":
        return f"{down_dist}pass falls incomplete.".strip()
    if primary == "sack":
        return f"{down_dist}quarterback taken down in the backfield.".strip()
    if primary == "qb_scramble":
        return f"{down_dist}quarterback tucks it and runs {yard_txt}.".strip()
    if primary == "interception":
        return f"{down_dist}picked off! Change of possession.".strip()
    if primary == "fumble":
        return f"{down_dist}ball is out—scramble on the turf.".strip()
    if primary == "touchdown":
        return f"{down_dist}touchdown!".strip()
    if primary == "tackle_big_hit":
        return f"{down_dist}monster hit stops the drive.".strip()
    return "Play in progress."