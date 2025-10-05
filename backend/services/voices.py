# backend/services/voices.py
from __future__ import annotations
from typing import Any, Dict, List

# ✅ Your verified 6 voices (unchanged)
VOICE_CATALOG: List[Dict[str, Any]] = [
    {"id": "tc_673eb45cdc1073aef51e6b90", "name": "Dean",    "gender": "male",   "emotions": ["tonedown", "normal", "toneup"]},
    {"id": "tc_6412c42d733f60ab8ad369a9", "name": "Caitlyn", "gender": "female", "emotions": ["normal"]},
    {"id": "tc_6837b58f80ceeb17115bb771", "name": "Walter",  "gender": "male",   "emotions": ["normal"]},
    {"id": "tc_684a5a7ba2ce934624b59c6e", "name": "Nia",     "gender": "female", "emotions": ["normal", "tonedown"]},
    {"id": "tc_623145940c2c1ff30c30f3a9", "name": "Matthew", "gender": "male",   "emotions": ["normal", "shout"]},
    {"id": "tc_630494521f5003bebbfdafef", "name": "Rachel",  "gender": "female", "emotions": ["normal", "toneup", "happy"]},
]

# ✅ Preferred voice per (tone, gender)
VOICE_BY_TONE_GENDER = {
    "neutral": {"male": "tc_673eb45cdc1073aef51e6b90", "female": "tc_6412c42d733f60ab8ad369a9"},  # Dean / Caitlyn
    "radio":   {"male": "tc_6837b58f80ceeb17115bb771", "female": "tc_684a5a7ba2ce934624b59c6e"},  # Walter / Nia
    "hype":    {"male": "tc_623145940c2c1ff30c30f3a9", "female": "tc_630494521f5003bebbfdafef"},  # Matthew / Rachel
}

# ✅ Emotion preference per tone (for previews & fallback)
PREFERRED_BY_TONE = {
    "neutral": ["normal"],
    "radio":   ["tonedown", "normal", "tonemid"],
    "hype":    ["shout", "toneup", "happy", "normal"],
}

def pick_emotion_for_tone(tone: str, supported: list[str]) -> str:
    for e in PREFERRED_BY_TONE.get(tone, ["normal"]):
        if e in supported:
            return e
    return "normal"
