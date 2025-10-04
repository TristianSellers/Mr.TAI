# backend/services/tone_profiles.py
from __future__ import annotations
from typing import Dict, Any

# ---- Tone presets (unchanged, but easy to extend) ----
TONE_PRESETS: Dict[str, Dict[str, Any]] = {
    "hype": {
        "system": (
            "You are an energetic TV play-by-play commentator. "
            "Speak in short, punchy clauses with vivid verbs. "
            "Use at most 1–2 exclamations. Avoid filler. Keep momentum high."
        ),
        "length_hint": "12–18 seconds",
    },
    "radio": {
        "system": (
            "You are a classic radio baseball-style commentator. "
            "Measured cadence, descriptive imagery, minimal exclamations. "
            "Prefer complete sentences and situational detail for listeners."
        ),
        "length_hint": "12–18 seconds",
    },
    "neutral": {
        "system": (
            "You are a professional broadcast commentator. "
            "Concise, balanced, and clear delivery with neutral tone."
        ),
        "length_hint": "10–15 seconds",
    },
}

def normalize_tone(tone: str | None) -> str:
    t = (tone or "").strip().lower()
    return t if t in TONE_PRESETS else "neutral"

# ---- Bias normalization + small guideline snippets ----
def normalize_bias(bias: str | None) -> str:
    b = (bias or "").strip().lower()
    return b if b in {"neutral", "home", "away"} else "neutral"

def _bias_guideline(bias: str, home: str, away: str) -> str:
    if bias == "home":
        return (
            f"Prioritize {home} events. Favor their perspective subtly (occasional 'we/us' ok). "
            f"Downplay {away} successes. Keep it believable."
        )
    if bias == "away":
        return (
            f"Prioritize {away} events. Favor their perspective subtly (occasional 'we/us' ok). "
            f"Downplay {home} successes. Keep it believable."
        )
    return "Call both sides fairly; avoid loaded adjectives; keep balance and clarity."

def build_llm_prompt(context: dict, transcript: str | None = None) -> str:
    """
    Builds a compact, tone- and bias-aware prompt.
    - Tone: selects system style + target length
    - Bias: shapes POV/guidelines (neutral/home/away)
    """
    tone = normalize_tone(context.get("tone"))
    bias = normalize_bias(context.get("bias"))

    preset = TONE_PRESETS[tone]
    sport = (context.get("sport") or "football").strip() or "football"
    home = (context.get("home_team") or "Home").strip() or "Home"
    away = (context.get("away_team") or "Away").strip() or "Away"

    pov = "balanced, impartial" if bias == "neutral" else (f"favor {home}" if bias == "home" else f"favor {away}")
    bias_rules = _bias_guideline(bias, home, away)

    # Keep transcript short for latency; still useful for context
    snippet = (transcript or "").strip()
    if len(snippet) > 1200:
        snippet = snippet[:1200] + " …"

    sys = preset["system"]
    length_hint = preset["length_hint"]

    # Single-string prompt (works with current llm.generate(prompt))
    return (
        f"{sys}\n"
        f"Sport: {sport}. Tone: {tone}. POV: {pov}. Target length: {length_hint}.\n"
        f"Bias guideline: {bias_rules}\n"
        f"If team names are known, mention them naturally. These are NFL Teams, going by at most three letter abbrieviation.\n\n"
        f"CONTEXT: {context}\n\n"
        f"TRANSCRIPT: {snippet}\n\n"
        f"Commentary:"
        f"Be sure to explicitly mention all parameters in your response. Do not improvise."
    )
