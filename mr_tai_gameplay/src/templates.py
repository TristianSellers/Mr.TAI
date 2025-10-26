from __future__ import annotations
from typing import Optional
from .schemas import ScoreboardFacts, BannerFacts

def _down_dist(sb: Optional[ScoreboardFacts]) -> str:
    if not sb or not sb.down or not sb.distance:
        return ""
    ords = {1:"1st", 2:"2nd", 3:"3rd", 4:"4th"}
    return f"{ords.get(sb.down, str(sb.down)+'th')} & {sb.distance}"

def one_liner(primary: str,
              yards: Optional[int],
              sb: Optional[ScoreboardFacts],
              banner: Optional[BannerFacts]) -> str:
    dd = _down_dist(sb)
    lead = f"{dd}: " if dd else ""
    ytxt = f" for {yards} yards" if (yards is not None and yards > 0) else ""

    if primary == "run":
        return f"{lead}The QB hands it off{ytxt}."
    if primary == "pass_attempt":
        return f"{lead}The QB drops and fires."
    if primary == "completion":
        fd = " for a first down" if banner and banner.result == "first_down" else ""
        return f"{lead}Completion{ytxt}{fd}."
    if primary == "incomplete":
        return f"{lead}Pass falls incomplete."
    if primary == "sack":
        return f"{lead}Quarterback sacked."
    if primary == "qb_scramble":
        return f"{lead}Quarterback tucks it and runs{ytxt}."
    if primary == "interception":
        return f"{lead}Intercepted!"
    if primary == "fumble":
        return f"{lead}Ball is out—fumble."
    if primary == "touchdown":
        return f"{lead}Touchdown!"
    if primary == "tackle_big_hit":
        return f"{lead}Big hit finishes the play."
    return f"{lead}Play in progress."
