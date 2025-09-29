// frontend/src/lib/api.js
// Centralized API helpers for Mr. TAI frontend

// Base URL (no trailing slash)
const RAW_API_BASE = import.meta?.env?.VITE_API_BASE ?? "http://localhost:8000";
export const API_BASE = RAW_API_BASE.replace(/\/+$/, "");

// ---- Utilities ----
export const staticUrl = (u) => (u ? `${API_BASE}${u}` : "");

// Basic JSON fetch wrapper
async function httpJson(url, init = {}) {
  const res = await fetch(url, init);
  if (!res.ok) {
    const msg = await res.text().catch(() => "");
    throw new Error(`${init.method || "GET"} ${url} failed: ${res.status} ${msg}`);
  }
  return res.json();
}

// ---- Health ----
export async function getHealth() {
  return httpJson(`${API_BASE}/health`);
}

// ---- Upload (optional legacy helper) ----
export async function uploadFile(file) {
  const fd = new FormData();
  fd.append("file", file);
  return httpJson(`${API_BASE}/upload`, { method: "POST", body: fd });
}

// ---- One-shot pipeline (video → OCR → LLM → TTS → mux) ----
export async function runCommentaryFromVideo({ file, tone = "neutral", bias = "neutral", gender = "male" }) {
  if (!file) throw new Error("Video file is required");
  const fd = new FormData();
  fd.append("video", file);
  fd.append("tone", tone);
  fd.append("bias", bias);
  fd.append("gender", gender);
  return httpJson(`${API_BASE}/pipeline/run-commentary-from-video`, { method: "POST", body: fd });
}

// ---- Voice preview / options ----
export async function getVoiceOptions({ tone = "neutral", gender } = {}) {
  const u = new URL(`${API_BASE}/pipeline/voice-options`);
  if (tone) u.searchParams.set("tone", tone);
  if (gender) u.searchParams.set("gender", gender);
  const res = await fetch(u);
  if (!res.ok) throw new Error(`voice-options failed: ${res.status}`);
  return res.json(); // { voices:[{id,name,gender,emotions,suggested_emotion}], tone, gender }
}

export async function previewVoice({ tone = "neutral", bias = "neutral", gender = "male", text = "Mic check. One-two.", voiceId, emotion } = {}) {
  const fd = new FormData();
  fd.append("tone", tone);
  fd.append("bias", bias);
  fd.append("gender", gender);
  fd.append("text", text);
  if (voiceId) fd.append("voice_id", voiceId);
  if (emotion) fd.append("emotion_preset", emotion);
  return httpJson(`${API_BASE}/pipeline/voice-preview`, { method: "POST", body: fd });
}

// ---- Legacy: direct analyze endpoint (kept for compatibility) ----
export async function analyzeCommentate(
  file,
  {
    home_team = null,
    away_team = null,
    score = null,
    quarter = null,
    clock = null,
    tone = "play-by-play",
    bias = "neutral",
    voice = "default",
    audio_only = false,
  } = {}
) {
  const fd = new FormData();
  if (file) fd.append("file", file);
  if (home_team != null) fd.append("home_team", String(home_team));
  if (away_team != null) fd.append("away_team", String(away_team));
  if (score != null) fd.append("score", String(score));
  if (quarter != null) fd.append("quarter", String(quarter));
  if (clock != null) fd.append("clock", String(clock));
  fd.append("tone", tone);
  fd.append("bias", bias);
  fd.append("voice", voice);
  fd.append("audio_only", String(audio_only));
  return httpJson(`${API_BASE}/analyze_commentate`, { method: "POST", body: fd });
}

export default {
  API_BASE,
  staticUrl,
  getHealth,
  uploadFile,
  runCommentaryFromVideo,
  getVoiceOptions,
  previewVoice,
  analyzeCommentate,
};
