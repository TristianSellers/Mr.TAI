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


# -------- Typecast --------
class TypecastTTS:
    """
    Minimal provider that matches your TTSClient protocol (synth_to_file).
    Supports 'tone' kwarg mapped to Typecast emotion presets and allows
    per-call voice override via 'voice_id' kwarg.

    Env:
      TYPECAST_API_KEY    (required)
      TYPECAST_VOICE_ID   (default voice; can be overridden per call)
      TYPECAST_MODEL      (default: 'ssfm-v21')
      TYPECAST_AUDIO_FORMAT (default: 'mp3')  # 'mp3' or 'wav'
    """
    # Map your app's tones -> Typecast emotion presets
    # (Tweak as you like; safe defaults)
    TONE_TO_EMOTION = {
        "play-by-play": "toneup",  # energetic / excited
        "hype":         "toneup",
        "radio":        "tonemid", # smoother/announcer
        "neutral":      "normal",
        "serious":      "normal",
        "sad":          "sad",
        "angry":        "angry",
        "happy":        "happy",
    }

    def __init__(self):
        self.key = (os.getenv("TYPECAST_API_KEY") or "").strip()
        if not self.key:
            raise RuntimeError("TYPECAST_API_KEY missing")
        self.voice_id = (os.getenv("TYPECAST_VOICE_ID") or "").strip()
        self.model = (os.getenv("TYPECAST_MODEL", "ssfm-v21") or "").strip()
        self.audio_format = (os.getenv("TYPECAST_AUDIO_FORMAT", "mp3") or "mp3").strip().lower()
        if self.audio_format not in ("mp3", "wav"):
            self.audio_format = "mp3"

    def synth_to_file(self, text: str, out_dir: str | pathlib.Path, **kwargs) -> str:
        """
        kwargs:
          - tone: string (maps to Typecast emotion preset)
          - bias: accepted but unused
          - voice_id: override env default
          - emotion_preset: pass-through to force a specific preset
          - emotion_intensity: float (0.0..2.0), default 1.0
          - volume: int (0..200), default 100
          - audio_pitch: int (-12..+12), default 0
          - audio_tempo: float (0.5..2.0), default 1.0
          - language: string (e.g., 'eng'), default 'eng'
          - seed: int, optional
          - model: override the model if needed
          - audio_format: 'mp3'|'wav' (overrides env)
        """
        out_dir = pathlib.Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        tone = (kwargs.get("tone") or "neutral").lower()
        emotion_preset = kwargs.get("emotion_preset") or self.TONE_TO_EMOTION.get(tone, "normal")
        emotion_intensity = float(kwargs.get("emotion_intensity", 1.0))

        payload_model = (kwargs.get("model") or self.model)
        voice_id = (kwargs.get("voice_id") or self.voice_id).strip()
        if not voice_id:
            raise RuntimeError("Typecast voice_id required (set TYPECAST_VOICE_ID or pass voice_id=...)")

        language = kwargs.get("language", "eng")
        volume = int(kwargs.get("volume", 100))
        audio_pitch = int(kwargs.get("audio_pitch", 0))
        audio_tempo = float(kwargs.get("audio_tempo", 1.0))
        seed = kwargs.get("seed")
        fmt = (kwargs.get("audio_format") or self.audio_format).lower()
        if fmt not in ("mp3", "wav"):
            fmt = "mp3"

        url = "https://api.typecast.ai/v1/text-to-speech"
        headers = {
            "X-API-KEY": self.key,
            "Content-Type": "application/json",
            "Accept": "*/*",  # audio bytes
        }
        body = {
            "voice_id": voice_id,
            "text": text,
            "model": payload_model,
            "language": language,
            "prompt": {
                "emotion_preset": emotion_preset,
                "emotion_intensity": emotion_intensity,
            },
            "output": {
                "volume": volume,
                "audio_pitch": audio_pitch,
                "audio_tempo": audio_tempo,
                "audio_format": fmt,
            },
        }
        if seed is not None:
            body["seed"] = int(seed)

        r = requests.post(url, json=body, headers=headers, timeout=120)
        if r.status_code != 200:
            snippet = r.text[:200].replace(self.key, "***")
            raise RuntimeError(f"Typecast {r.status_code}: {snippet}")

        out_path = out_dir / f"tts_{uuid.uuid4().hex}.{fmt}"
        out_path.write_bytes(r.content)
        return str(out_path)

@lru_cache(maxsize=1)
def get_tts():
    # default to typecast now, not elevenlabs
    provider = (os.getenv("TTS_PROVIDER") or "typecast").lower()

    if provider == "typecast":
        return TypecastTTS()   # <-- matches your class name

    if provider == "openai":
        return OpenAITTS()

    if provider == "elevenlabs":
        return ElevenLabsTTS()

    raise RuntimeError(f"Unknown TTS_PROVIDER: {provider!r}")