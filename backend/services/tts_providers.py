# backend/services/tts_providers.py
from __future__ import annotations
from typing import Protocol
from functools import lru_cache
import pathlib, uuid, os, requests

class TTSClient(Protocol):
    def synth_to_file(self, text: str, out_dir: str | pathlib.Path) -> str: ...

# -------- ElevenLabs (default) --------
class ElevenLabsTTS:
    def __init__(self):
        self.key = os.getenv("ELEVENLABS_API_KEY")
        if not self.key:
            raise RuntimeError("ELEVENLABS_API_KEY missing")
        self.voice_id = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
        self.model = os.getenv("ELEVENLABS_MODEL", "eleven_multilingual_v2")

    def synth_to_file(self, text: str, out_dir: str | pathlib.Path) -> str:
        out_dir = pathlib.Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / f"tts_{uuid.uuid4().hex}.mp3"

        url = f"https://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}"
        headers = {
            "xi-api-key": self.key,
            "accept": "audio/mpeg",
            "content-type": "application/json",
        }
        payload = {
            "text": text,
            "model_id": self.model,
            "voice_settings": {"stability": 0.5, "similarity_boost": 0.8},
        }
        r = requests.post(url, json=payload, headers=headers, timeout=60)
        r.raise_for_status()
        out.write_bytes(r.content)
        return str(out)

# -------- OpenAI (optional) --------
class OpenAITTS:
    def __init__(self):
        from openai import OpenAI
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY missing for OpenAI TTS")
        self.client = OpenAI(api_key=key)
        self.model = os.getenv("OPENAI_TTS_MODEL", "gpt-4o-mini-tts")
        self.voice = os.getenv("OPENAI_TTS_VOICE", "alloy")

    def synth_to_file(self, text: str, out_dir: str | pathlib.Path) -> str:
        out_dir = pathlib.Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / f"tts_{uuid.uuid4().hex}.mp3"
        resp = self.client.audio.speech.create(model=self.model, voice=self.voice, input=text)
        with open(out, "wb") as f:
            f.write(resp.read())
        return str(out)

@lru_cache
def get_tts():
    provider = (os.getenv("TTS_PROVIDER", "elevenlabs") or "").lower()
    if provider == "openai":
        return OpenAITTS()
    return ElevenLabsTTS()
