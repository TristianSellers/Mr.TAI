# backend/services/tts_providers.py
from __future__ import annotations
from typing import Protocol
from functools import lru_cache
import pathlib, uuid, os, requests

class TTSClient(Protocol):
    # Accept **kwargs so callers can pass tone/bias without breaking providers.
    def synth_to_file(self, text: str, out_dir: str | pathlib.Path, **kwargs) -> str: ...


# -------- ElevenLabs (tone-aware) --------
class ElevenLabsTTS:
    # Tunable per-tone settings (light touch so it still sounds natural).
    TONE_SETTINGS = {
        "hype":    {"stability": 0.45, "similarity_boost": 0.85},
        "radio":   {"stability": 0.65, "similarity_boost": 0.75},
        "neutral": {"stability": 0.55, "similarity_boost": 0.80},
    }

    def __init__(self):
        self.key = (os.getenv("ELEVENLABS_API_KEY") or "").strip()
        if not self.key:
            raise RuntimeError("ELEVENLABS_API_KEY missing")
        self.voice_id = (os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM") or "").strip()
        self.model = (os.getenv("ELEVENLABS_MODEL", "eleven_multilingual_v2") or "").strip()

    def synth_to_file(self, text: str, out_dir: str | pathlib.Path, **kwargs) -> str:
        """
        kwargs:
          - tone: 'hype' | 'radio' | 'neutral' (optional; defaults to neutral)
          - bias: accepted but ignored here (bias primarily handled in LLM prompt)
        """
        out_dir = pathlib.Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / f"tts_{uuid.uuid4().hex}.mp3"

        tone_key = (kwargs.get("tone") or "neutral").lower()
        settings = self.TONE_SETTINGS.get(tone_key, self.TONE_SETTINGS["neutral"])

        url = f"https://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}"
        headers = {
            "xi-api-key": self.key,
            "accept": "audio/mpeg",
            "content-type": "application/json",
        }
        payload = {
            "text": text,
            "model_id": self.model,
            "voice_settings": settings,
        }

        r = requests.post(url, json=payload, headers=headers, timeout=60)
        if r.status_code != 200:
            # Mask the key if the server echoed it back (defensive).
            snippet = r.text[:200].replace(self.key, "***")
            raise RuntimeError(f"ElevenLabs {r.status_code}: {snippet}")

        out.write_bytes(r.content)
        return str(out)


# -------- OpenAI (optional) --------
class OpenAITTS:
    def __init__(self):
        from openai import OpenAI
        key = (os.getenv("OPENAI_API_KEY") or "").strip()
        if not key:
            raise RuntimeError("OPENAI_API_KEY missing for OpenAI TTS")
        self.client = OpenAI(api_key=key)
        self.model = (os.getenv("OPENAI_TTS_MODEL", "gpt-4o-mini-tts") or "").strip()
        self.voice = (os.getenv("OPENAI_TTS_VOICE", "alloy") or "").strip()

    def synth_to_file(self, text: str, out_dir: str | pathlib.Path, **kwargs) -> str:
        """
        kwargs accepted (tone/bias) but currently unused by OpenAI TTS.
        """
        out_dir = pathlib.Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / f"tts_{uuid.uuid4().hex}.mp3"

        # Preferred: streaming response API (OpenAI v1 SDK)
        try:
            with self.client.audio.speech.with_streaming_response.create(
                model=self.model,
                voice=self.voice,
                input=text,
            ) as resp:
                resp.stream_to_file(str(out))
                return str(out)
        except AttributeError:
            # Fallback: older/newer variants that expose .create() with bytes-like content
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


# -------- TypeCast.AI (stub w/ real API support when configured) --------
class TypeCastTTS:
    """
    When TYPECAST_API_KEY is set, call TypeCast API.
    If not set, gracefully fallback to ElevenLabs (so you can test the flow today).
    Env:
      - TYPECAST_API_KEY
      - TYPECAST_BASE_URL (default: https://api.typecast.ai)
      - TYPECAST_DEFAULT_VOICE
      - TYPECAST_VOICE_MAP (JSON: {"hype|neutral": "voice_id", "radio|neutral": "voice_id", ...})
    """
    def __init__(self):
        import json
        self.key = (os.getenv("TYPECAST_API_KEY") or "").strip()
        self.base = (os.getenv("TYPECAST_BASE_URL", "https://api.typecast.ai") or "").rstrip("/")
        self.voice_default = os.getenv("TYPECAST_DEFAULT_VOICE", "typecast_default")
        try:
            self.voice_map = json.loads(os.getenv("TYPECAST_VOICE_MAP", "{}"))
        except Exception:
            self.voice_map = {}
        # Prepare ElevenLabs fallback
        self._fallback_eleven: ElevenLabsTTS | None = None
        try:
            self._fallback_eleven = ElevenLabsTTS()
        except Exception:
            self._fallback_eleven = None

    def _pick_voice(self, tone: str | None, bias: str | None) -> str:
        key = f"{(tone or 'neutral').lower()}|{(bias or 'neutral').lower()}"
        return self.voice_map.get(key, self.voice_default)

    def synth_to_file(self, text: str, out_dir: str | pathlib.Path, **kwargs) -> str:
        # If no API key, fallback to ElevenLabs (keeps your pipeline working)
        if not self.key:
            if self._fallback_eleven:
                return self._fallback_eleven.synth_to_file(text, out_dir, **kwargs)
            raise RuntimeError("TypeCast not configured and ElevenLabs fallback unavailable")

        out_dir = pathlib.Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / f"tts_{uuid.uuid4().hex}.mp3"

        voice_id = self._pick_voice(kwargs.get("tone"), kwargs.get("bias"))
        url = f"{self.base}/v1/tts"
        headers = {"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"}
        payload = {
            "voice_id": voice_id,
            "text": text,
            # Hint: you can extend with model/emotion/pitch/tempo when you have exact API spec.
            "metadata": {"tone": kwargs.get("tone"), "bias": kwargs.get("bias")},
        }

        r = requests.post(url, headers=headers, json=payload, timeout=60)
        if r.status_code >= 400:
            raise RuntimeError(f"TypeCast {r.status_code}: {r.text[:200]}")

        # If TypeCast returns JSON with a URL, download; else assume audio bytes.
        ct = r.headers.get("content-type", "")
        if "application/json" in ct:
            data = r.json()
            audio_url = data.get("audio_url")
            if not audio_url:
                raise RuntimeError("TypeCast: no audio_url in response")
            r2 = requests.get(audio_url, timeout=120)
            if r2.status_code >= 400:
                raise RuntimeError(f"TypeCast download {r2.status_code}")
            out.write_bytes(r2.content)
        else:
            out.write_bytes(r.content)

        return str(out)


# -------- Hybrid router --------
@lru_cache(maxsize=1)
def get_tts():
    """
    Hybrid routing:
      - tone=neutral  -> ElevenLabs
      - tone=hype/radio -> TypeCast (or fallback to ElevenLabs if TypeCast not configured)
    """
    eleven = None
    typecast = None
    # Initialize providers (best-effort)
    try:
        eleven = ElevenLabsTTS()
    except Exception:
        eleven = None
    try:
        typecast = TypeCastTTS()
    except Exception:
        typecast = None

    class HybridTTS:
        def synth_to_file(self, text: str, out_dir: str | pathlib.Path, **kwargs) -> str:
            tone = (kwargs.get("tone") or "neutral").lower()
            # Route by tone
            if tone in {"hype", "radio"} and typecast is not None:
                try:
                    return typecast.synth_to_file(text, out_dir, **kwargs)
                except Exception:
                    # Fallback to ElevenLabs if TypeCast fails
                    if eleven is not None:
                        return eleven.synth_to_file(text, out_dir, **kwargs)
                    raise
            # Default → ElevenLabs
            if eleven is not None:
                return eleven.synth_to_file(text, out_dir, **kwargs)
            # Last resort: TypeCast if ElevenLabs unavailable
            if typecast is not None:
                return typecast.synth_to_file(text, out_dir, **kwargs)
            raise RuntimeError("No TTS provider available")

    return HybridTTS()
