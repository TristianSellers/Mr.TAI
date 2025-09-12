# backend/services/llm_providers.py
from __future__ import annotations
from typing import Protocol, Dict, Any
import os

class LLMClient(Protocol):
    def generate(self, prompt: str, meta: Dict[str, Any] | None = None) -> str: ...

class StubLLM:
    """
    Offline-safe default: returns a short canned line so the rest of
    the pipeline can run without network/API keys.
    """
    def generate(self, prompt: str, meta: Dict[str, Any] | None = None) -> str:
        tone = (meta or {}).get("tone", "hype")
        return f"[{tone} demo] Incredible play! The offense strings it together for a highlight moment."

def get_llm() -> LLMClient:
    """
    Factory. Real providers will be added next and selected via LLM_PROVIDER.
    For now, always return StubLLM.
    """
    _provider = os.getenv("LLM_PROVIDER", "stub").lower()
    return StubLLM()
