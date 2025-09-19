# backend/services/tone_profiles.py
from __future__ import annotations
from typing import Dict, Any

# Expandable library of tone presets for LLM prompting.
# Keep these short and opinionated so style changes are obvious.
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

def build_llm_prompt(context: dict, transcript: str | None = None) -> str:
    """
    Builds a compact, tone-aware prompt. Keeps transcript short for latency.
    """
    tone = normalize_tone(context.get("tone"))
    preset = TONE_PRESETS[tone]
    sport = (context.get("sport") or "football").strip() or "football"

    snippet = (transcript or "").strip()
    if len(snippet) > 1200:
        snippet = snippet[:1200] + " …"

    # System guidance (not a separate chat role here—embedded since we call .generate(prompt))
    sys = preset["system"]
    length_hint = preset["length_hint"]

    return (
        f"{sys}\n"
        f"Sport: {sport}. Tone: {tone}. Target length: {length_hint}.\n"
        f"Use the provided context if relevant. Avoid profanity.\n\n"
        f"CONTEXT: {context}\n\n"
        f"TRANSCRIPT: {snippet}\n\n"
        f"Commentary:"
    )
