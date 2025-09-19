# backend/services/tts_providers.py
from __future__ import annotations
from typing import Protocol
from functools import lru_cache
import pathlib, uuid, os, requests


class TTSClient(Protocol):
    def synth_to_file(
        self, text: str, out_dir: str | pathlib.Path, tone: str | None = None
    ) -> str: ...


# -------- ElevenLabs (default, tone-aware) --------
class ElevenLabsTTS:
    TONE_SETTINGS = {
        "hype":   {"stability": 0.45, "similarity_boost": 0.85},
        "radio":  {"stability": 0.65, "similarity_boost": 0.75},
        "neutral":{"stability": 0.55, "similarity_boost": 0.80},
    }

    def __init__(self):
        self.key = (os.getenv("ELEVENLABS_API_KEY") or "").strip()
        if not self.key:
            raise RuntimeError("ELEVENLABS_API_KEY missing")
        self.voice_id = (os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM") or "").strip()
        self.model = (os.getenv("ELEVENLABS_MODEL", "eleven_multilingual_v2") or "").strip()

    def synth_to_file(
        self, text: str, out_dir: str | pathlib.Path, tone: str | None = None
    ) -> str:
        out_dir = pathlib.Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / f"tts_{uuid.uuid4().hex}.mp3"

        tone_key = (tone or "neutral").lower()
        settings = self.TONE_SETTINGS.get(tone_key, self.TONE_SETTINGS["neutral"])

        url = f"https://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}"
        headers = {
            "xi-api-key": self.key,
            "accept": "audio/mpeg",
            "content-type": "application/json",
        }
        payload = {"text": text, "model_id": self.model, "voice_settings": settings}

        r = requests.post(url, json=payload, headers=headers, timeout=60)
        if r.status_code != 200:
            snippet = r.text[:200].replace(self.key, "***")
            raise RuntimeError(f"ElevenLabs {r.status_code}: {snippet}")
        out.write_bytes(r.content)
        return str(out)


# -------- OpenAI (optional, tone accepted but ignored) --------
class OpenAITTS:
    def __init__(self):
        from openai import OpenAI
        key = (os.getenv("OPENAI_API_KEY") or "").strip()
        if not key:
            raise RuntimeError("OPENAI_API_KEY missing for OpenAI TTS")
        self.client = OpenAI(api_key=key)
        self.model = (os.getenv("OPENAI_TTS_MODEL", "gpt-4o-mini-tts") or "").strip()
        self.voice = (os.getenv("OPENAI_TTS_VOICE", "alloy") or "").strip()

    def synth_to_file(
        self, text: str, out_dir: str | pathlib.Path, tone: str | None = None
    ) -> str:
        out_dir = pathlib.Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / f"tts_{uuid.uuid4().hex}.mp3"

        # Prefer streaming API (OpenAI v1 SDK)
        try:
            with self.client.audio.speech.with_streaming_response.create(
                model=self.model,
                voice=self.voice,
                input=text,
            ) as resp:
                resp.stream_to_file(str(out))
                return str(out)
        except AttributeError:
            # Fallback: write bytes if exposed
            resp = self.client.audio.speech.create(
                model=self.model,
                voice=self.voice,
                input=text,
            )
            data = getattr(resp, "content", None)
            if data is None and hasattr(resp, "read"):
                data = resp.read()
            if not data:
                raise RuntimeError("OpenAI TTS response did not contain audio bytes")
            out.write_bytes(data)
            return str(out)


@lru_cache(maxsize=1)
def get_tts():
    provider = (os.getenv("TTS_PROVIDER", "elevenlabs") or "").lower()
    if provider == "openai":
        return OpenAITTS()
    return ElevenLabsTTS()
