# backend/services/ocr_llm.py
from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Any, Dict, List, cast

from openai import OpenAI
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionContentPartParam,
    ChatCompletionContentPartTextParam,
    ChatCompletionContentPartImageParam,
)

# ---------------------------
# Environment loading (optional)
# ---------------------------
def _load_env_for_ocr_llm() -> None:
    """Load .env files so running this module directly works."""
    try:
        from dotenv import load_dotenv  # type: ignore
    except Exception:
        return
    ROOT = Path(__file__).resolve().parents[2]
    for f in [
        ROOT / ".env",
        ROOT / ".env.local",
        ROOT / "backend" / ".env",
        ROOT / "backend" / ".env.local",
    ]:
        if f.exists():
            load_dotenv(f, override=True)

# call on import so CLIs work
_load_env_for_ocr_llm()

# ---------------------------
# Helpers / normalization
# ---------------------------
NULLISH = {"", "null", "none", "nil", "n/a", "na", "-", "–", "—"}

def _n(v: Any) -> str | None:
    """Convert null-ish strings to None; strip whitespace, stringify numbers."""
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return str(v)
    s = str(v).strip()
    return None if s.lower() in NULLISH else s

def _clean_fields(d: dict) -> dict:
    """
    Normalize model output:
      - null-ish -> None
      - score to '##-##'
      - quarter to 'Q#' or 'OT'
      - clock to 'M:SS'
    """
    import re
    home = _n(d.get("home_team"))
    away = _n(d.get("away_team"))
    score = _n(d.get("score"))
    quarter = _n(d.get("quarter"))
    clock = _n(d.get("clock"))

    if isinstance(score, list) and len(score) >= 2:
        score = f"{score[0]}-{score[1]}"
    if isinstance(score, str):
        m = re.search(r"(\d{1,2})\s*[-:]\s*(\d{1,2})", score)
        score = f"{m.group(1)}-{m.group(2)}" if m else None

    if isinstance(quarter, str):
        m = re.search(r"\b(Q[1-4]|OT)\b", quarter.upper())
        if not m:
            m2 = re.search(r"\b([1-4])\b", quarter)
            quarter = f"Q{m2.group(1)}" if m2 else None
        else:
            quarter = m.group(1)

    if isinstance(clock, str):
        m = re.search(r"\b([0-5]?\d)[:\.]([0-5]\d)\b", clock)
        clock = f"{int(m.group(1))}:{m.group(2)}" if m else None

    return {
        "home_team": home,
        "away_team": away,
        "score": score,
        "quarter": quarter,
        "clock": clock,
    }

# ---------------------------
# Prompting
# ---------------------------
_SYS = (
    "You read a Madden NFL scoreboard banner from an image that is ALREADY cropped "
    "to the thin HUD bar at the bottom. Return ONLY a JSON object with keys: "
    "home_team, away_team, score, quarter, clock. Use null for unknowns. Example: "
    '{"home_team":"USC","away_team":"UCLA","score":"21-24","quarter":"Q4","clock":"0:42"}'
)

_USER_HINT = (
    "The image is a pre-cropped scoreboard strip. Extract fields: "
    "home_team, away_team, score as '##-##', quarter like 'Q1..Q4' or 'OT', and clock 'M:SS'. "
    "If a field is not visible, set it to null."
)

MODEL = os.getenv("OPENAI_VISION_MODEL", "gpt-4o-mini")

def _image_to_data_uri(p: Path) -> str:
    b = p.read_bytes()
    b64 = base64.b64encode(b).decode("ascii")
    # Use 'image/jpeg' or 'image/png'—jpeg is fine for your frame.jpg
    mime = "image/jpeg" if p.suffix.lower() in {".jpg", ".jpeg"} else "image/png"
    return f"data:{mime};base64,{b64}"

# ---------------------------
# Core API call (with fallback)
# ---------------------------
def _call_chat_json(client: OpenAI, model: str, messages: List[ChatCompletionMessageParam]) -> Dict[str, Any]:
    """
    Try to force a JSON object response. If the server rejects the response_format,
    the caller should fall back to plain text.
    """
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
        response_format=cast(Any, {"type": "json_object"}),  # keep type-checkers happy
    )
    content = resp.choices[0].message.content or ""
    return cast(Dict[str, Any], json.loads(content))

def extract_scoreboard_with_llm(
    image_path: str | os.PathLike[str] = "frame.jpg",
) -> Dict[str, Any]:
    """
    Reads a PRE-CROPPED Madden scoreboard banner image (default: ./frame.jpg),
    sends it to the OpenAI vision model, and returns normalized fields.

    Returns:
      {
        "home_team": str|None,
        "away_team": str|None,
        "score": str|None,
        "quarter": str|None,
        "clock": str|None,
        "raw_json": str
      }
    """
    key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not key:
        raise RuntimeError("OPENAI_API_KEY missing")

    client = OpenAI(api_key=key)

    p = Path(image_path)
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {p}")

    data_uri = _image_to_data_uri(p)

    content_parts: List[ChatCompletionContentPartParam] = [
        ChatCompletionContentPartTextParam(type="text", text=_USER_HINT),
        ChatCompletionContentPartImageParam(type="image_url", image_url={"url": data_uri}),
    ]

    messages: List[ChatCompletionMessageParam] = [
        {"role": "system", "content": _SYS},
        {"role": "user", "content": content_parts},
    ]

    # Try JSON mode; if quota/rate/format issues happen, fall back to plain text then parse.
    model = MODEL
    last_err: Exception | None = None
    for attempt in range(2):
        try:
            if attempt == 0:
                obj = _call_chat_json(client, model, messages)
            else:
                # Fallback: no response_format, then parse whatever came back.
                resp = client.chat.completions.create(model=model, messages=messages, temperature=0)
                content = (resp.choices[0].message.content or "").strip()
                try:
                    obj = cast(Dict[str, Any], json.loads(content))
                except Exception:
                    obj = {}
            # success path
            norm = _clean_fields(obj)
            return {
                **norm,
                "raw_json": json.dumps(
                    {
                        "home_team": obj.get("home_team"),
                        "away_team": obj.get("away_team"),
                        "score": obj.get("score"),
                        "quarter": obj.get("quarter"),
                        "clock": obj.get("clock"),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
            }
        except Exception as e:
            last_err = e
            # On first failure, try fallback loop iteration
            continue

    # If both attempts failed, surface the last error
    assert last_err is not None
    raise last_err

# ---------------------------
# CLI
# ---------------------------
if __name__ == "__main__":
    import sys
    img = sys.argv[1] if len(sys.argv) > 1 else "frame.jpg"
    result = extract_scoreboard_with_llm(img)
    print(json.dumps(result, indent=2))
