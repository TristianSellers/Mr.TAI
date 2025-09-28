# backend/services/tts_providers.py
from __future__ import annotations
from typing import Protocol
from functools import lru_cache
import pathlib, uuid, os, time, json, requests

class TTSClient(Protocol):
    def synth_to_file(self, text: str, out_dir: str | pathlib.Path, **kwargs) -> str: ...


# -------- ElevenLabs (tone-aware) --------
class ElevenLabsTTS:
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
        payload = {"text": text, "model_id": self.model, "voice_settings": settings}

        r = requests.post(url, json=payload, headers=headers, timeout=60)
        if r.status_code != 200:
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
        out_dir = pathlib.Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / f"tts_{uuid.uuid4().hex}.mp3"

        try:
            with self.client.audio.speech.with_streaming_response.create(
                model=self.model, voice=self.voice, input=text
            ) as resp:
                resp.stream_to_file(str(out))
                return str(out)
        except AttributeError:
            resp = self.client.audio.speech.create(model=self.model, voice=self.voice, input=text)
            data = getattr(resp, "content", None)
            if data is None and hasattr(resp, "read"):
                data = resp.read()
            if not data:
                raise RuntimeError("OpenAI TTS response did not contain audio bytes")
            out.write_bytes(data)
            return str(out)


# -------- TypeCast.AI (real integration) --------
class TypeCastTTS:
    """
    Two strategies:
      1) POST /v1/text-to-speech (sync) with X-API-KEY (preferred)
      2) POST /api/speak -> poll GET /api/speak/v2/{id} -> download audio (Bearer token)
    Env:
      - TYPECAST_API_KEY (required)
      - TYPECAST_BASE_URL (default https://typecast.ai)
      - TYPECAST_DEFAULT_VOICE
      - TYPECAST_VOICE_MAP (JSON: {"hype|neutral":"voice_x", "radio|neutral":"voice_y", ...})
      - TYPECAST_MODEL (optional; e.g., "ssfm-v21")
      - TYPECAST_POLL_INTERVAL_MS (default 1000)
      - TYPECAST_POLL_TIMEOUT_S (default 60)
    """
    def __init__(self):
        self.key = (os.getenv("TYPECAST_API_KEY") or "").strip()
        if not self.key:
            raise RuntimeError("TYPECAST_API_KEY missing")
        self.base = (os.getenv("TYPECAST_BASE_URL", "https://typecast.ai") or "").rstrip("/")
        self.voice_default = os.getenv("TYPECAST_DEFAULT_VOICE", "typecast_default")
        self.model = os.getenv("TYPECAST_MODEL", "ssfm-v21")
        try:
            self.voice_map = json.loads(os.getenv("TYPECAST_VOICE_MAP", "{}"))
        except Exception:
            self.voice_map = {}
        self.poll_interval_ms = int(os.getenv("TYPECAST_POLL_INTERVAL_MS", "1000"))
        self.poll_timeout_s = int(os.getenv("TYPECAST_POLL_TIMEOUT_S", "60"))

    def _pick_voice(self, tone: str | None, bias: str | None) -> str:
        key = f"{(tone or 'neutral').lower()}|{(bias or 'neutral').lower()}"
        return self.voice_map.get(key, self.voice_default)

    # ---------- Strategy 1: /v1/text-to-speech (sync) ----------
    def _tts_v1_sync(self, text: str, voice_id: str) -> bytes | None:
        url = f"{self.base}/v1/text-to-speech"
        headers = {
            "X-API-KEY": self.key,               # per docs
            "Authorization": f"Bearer {self.key}",  # some docs/examples use Bearer
            "Content-Type": "application/json",
            "Accept": "audio/mpeg,application/json",
        }
        payload = {
            "text": text,
            "voice_id": voice_id,
            "model": self.model,
            # You can add "prompt"/emotion here if desired.
        }
        r = requests.post(url, json=payload, headers=headers, timeout=90)
        if r.status_code >= 400:
            return None
        # Either bytes directly or JSON with URL
        ct = r.headers.get("content-type", "")
        if "application/json" in ct:
            try:
                data = r.json()
                audio_url = data.get("audio_url")
                if audio_url:
                    r2 = requests.get(audio_url, timeout=120)
                    if r2.status_code < 400:
                        return r2.content
            except Exception:
                return None
            return None
        return r.content

    # ---------- Strategy 2: /api/speak (async) ----------
    def _speak_async(self, text: str, voice_id: str) -> bytes:
        # Step 1: request synthesis job
        url = f"{self.base}/api/speak"
        headers = {
            "Authorization": f"Bearer {self.key}",
            "X-API-KEY": self.key,  # belt & suspenders
            "Content-Type": "application/json",
        }
        payload = {
            "text": text,
            "actor_id": voice_id,   # docs use "actor"/"voice" terminology; many examples use actor_id
            "model": self.model,
        }
        r = requests.post(url, json=payload, headers=headers, timeout=60)
        if r.status_code >= 400:
            raise RuntimeError(f"TypeCast speak error {r.status_code}: {r.text[:200]}")
        data = r.json()
        speak_id = data.get("result", {}).get("id") or data.get("id") or data.get("result_id")
        if not speak_id:
            raise RuntimeError("TypeCast /api/speak: missing job id")

        # Step 2: poll status
        status_url = f"{self.base}/api/speak/v2/{speak_id}"
        t0 = time.time()
        while True:
            r2 = requests.get(status_url, headers={"Authorization": f"Bearer {self.key}"}, timeout=30)
            if r2.status_code >= 400:
                raise RuntimeError(f"TypeCast poll {r2.status_code}: {r2.text[:200]}")
            info = r2.json()
            status = info.get("result", {}).get("status") or info.get("status")
            if status == "done":
                # Expect audio_download_url in result
                dl = (
                    info.get("result", {}).get("audio_download_url")
                    or info.get("audio_download_url")
                    or info.get("result", {}).get("download_url")
                )
                if not dl:
                    raise RuntimeError("TypeCast: done but no audio_download_url")
                r3 = requests.get(dl, timeout=120)
                if r3.status_code >= 400:
                    raise RuntimeError(f"TypeCast download {r3.status_code}")
                return r3.content
            if (time.time() - t0) > self.poll_timeout_s:
                raise RuntimeError("TypeCast poll timeout")
            time.sleep(self.poll_interval_ms / 1000.0)

    def synth_to_file(self, text: str, out_dir: str | pathlib.Path, **kwargs) -> str:
        out_dir = pathlib.Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / f"tts_{uuid.uuid4().hex}.mp3"

        voice_id = self._pick_voice(kwargs.get("tone"), kwargs.get("bias"))

        # Try synchronous v1 first (if available)
        blob = self._tts_v1_sync(text, voice_id)
        if blob is None:
            # Fallback to async job flow
            blob = self._speak_async(text, voice_id)

        out.write_bytes(blob)
        return str(out)


# -------- Hybrid router --------
# -------- Hybrid router --------
class _HybridTTS:
    def __init__(self, eleven=None, typecast=None, openai=None):
        self.eleven = eleven
        self.typecast = typecast
        self.openai = openai

    def synth_to_file(self, text: str, out_dir: str | pathlib.Path, **kwargs) -> str:
        tone = (kwargs.get("tone") or "neutral").lower()
        # Route by tone first
        if tone in {"hype", "radio"} and self.typecast is not None:
            try:
                return self.typecast.synth_to_file(text, out_dir, **kwargs)
            except Exception:
                # fallback: ElevenLabs -> OpenAI
                if self.eleven is not None:
                    return self.eleven.synth_to_file(text, out_dir, **kwargs)
                if self.openai is not None:
                    return self.openai.synth_to_file(text, out_dir, **kwargs)
                raise
        # Neutral / default path
        if self.eleven is not None:
            return self.eleven.synth_to_file(text, out_dir, **kwargs)
        if self.openai is not None:
            return self.openai.synth_to_file(text, out_dir, **kwargs)
        if self.typecast is not None:
            return self.typecast.synth_to_file(text, out_dir, **kwargs)
        raise RuntimeError("No TTS provider available (configure ELEVENLABS_API_KEY / OPENAI_API_KEY / TYPECAST_API_KEY).")


from functools import lru_cache

@lru_cache(maxsize=1)
def get_tts():
    """
    Provider selection precedence:
      - If TTS_PROVIDER is explicitly set to:  openai | elevenlabs | typecast | hybrid
        honor it (backward compatible).
      - Otherwise default to 'hybrid' routing (neutral→ElevenLabs, hype/radio→TypeCast),
        with graceful fallbacks to OpenAI if others are missing.
    """
    provider = (os.getenv("TTS_PROVIDER", "hybrid") or "").lower()

    # Lazy init: try each provider and ignore missing creds
    eleven = typecast = openai = None
    try:
        eleven = ElevenLabsTTS()
    except Exception:
        pass
    try:
        typecast = TypeCastTTS()
    except Exception:
        pass
    try:
        openai = OpenAITTS()
    except Exception:
        pass

    if provider == "openai":
        if openai is None:
            raise RuntimeError("OPENAI_API_KEY missing for OpenAI TTS")
        return openai
    if provider == "elevenlabs":
        if eleven is None:
            raise RuntimeError("ELEVENLABS_API_KEY missing for ElevenLabs TTS")
        return eleven
    if provider == "typecast":
        if typecast is None:
            raise RuntimeError("TYPECAST_API_KEY missing for TypeCast TTS")
        return typecast

    # Default: hybrid with graceful fallbacks
    return _HybridTTS(eleven=eleven, typecast=typecast, openai=openai)
