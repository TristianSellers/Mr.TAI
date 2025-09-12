# backend/services/llm_providers.py
from __future__ import annotations
from typing import Protocol, Dict, Any
import os

# ---------- Interface ----------
class LLMClient(Protocol):
    def generate(self, prompt: str, meta: Dict[str, Any] | None = None) -> str: ...

# ---------- Stub (default/offline) ----------
class StubLLM:
    def generate(self, prompt: str, meta: Dict[str, Any] | None = None) -> str:
        tone = (meta or {}).get("tone", "hype")
        return f"[{tone} demo] Incredible play! The offense strings it together for a highlight moment."

# ---------- OpenAI ----------
class OpenAILLM:
    def __init__(self):
        from openai import OpenAI
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY not set")
        self.client = OpenAI(api_key=key)
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.temperature = float(os.getenv("LLM_TEMP", "0.8"))
        self.max_tokens = int(os.getenv("LLM_MAX_TOKENS", "350"))

    def generate(self, prompt: str, meta: Dict[str, Any] | None = None) -> str:
        sys = "You are an energetic sports commentator. Keep output ~12–18 seconds."
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": sys},
                      {"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return (resp.choices[0].message.content or "").strip()

# ---------- Anthropic ----------
class AnthropicLLM:
    def __init__(self):
        import anthropic
        key = os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        self.client = anthropic.Anthropic(api_key=key)
        self.model = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest")
        self.temperature = float(os.getenv("LLM_TEMP", "0.8"))
        self.max_tokens = int(os.getenv("LLM_MAX_TOKENS", "350"))

    def generate(self, prompt: str, meta: Dict[str, Any] | None = None) -> str:
        msg = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        # concat text blocks only
        parts = []
        for b in getattr(msg, "content", []) or []:
            if getattr(b, "type", "") == "text":
                parts.append(b.text)
        return "".join(parts).strip()

# ---------- Factory ----------
def get_llm() -> LLMClient:
    provider = os.getenv("LLM_PROVIDER", "stub").lower()
    try:
        if provider == "openai":
            return OpenAILLM()
        if provider == "anthropic":
            return AnthropicLLM()
        return StubLLM()
    except Exception:
        # Fallback hard to stub if misconfigured keys/models
        return StubLLM()
