# backend/services/context_from_ocr.py
from __future__ import annotations
from typing import Dict, Optional, Tuple, Any

def _split_score(score: Optional[str]) -> Tuple[Optional[int], Optional[int]]:
    if not score or "-" not in score:
        return (None, None)
    a, b = score.split("-", 1)
    try:
        return (int(a.strip()), int(b.strip()))
    except Exception:
        return (None, None)

def ocr_to_commentary_context(ocr: Dict[str, Any]) -> Dict[str, Any]:
    """
    Transform OCR dict (home_team, away_team, score, quarter, clock) into the
    commentary context shape used by /analyze_commentate.
    """
    # Narrow type for score to str|None for Pylance
    score_val = ocr.get("score")
    raw_score: Optional[str] = score_val if isinstance(score_val, str) else None
    away_score, home_score = _split_score(raw_score)

    ctx: Dict[str, Any] = {
        "home_team": ocr.get("home_team"),
        "away_team": ocr.get("away_team"),
        "quarter":   ocr.get("quarter"),
        "clock":     ocr.get("clock"),
        "home_score": home_score,
        "away_score": away_score,
        # You can set defaults; your prompt builder normalizes tone later.
        "tone": "play-by-play",
        "bias": "neutral",
        "_ocr_raw": ocr,
    }
    return ctx
