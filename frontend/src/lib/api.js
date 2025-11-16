// frontend/src/lib/api.js
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

// ---- One-shot pipeline ----
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
  return res.json();
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

export default {
  API_BASE,
  staticUrl,
  getHealth,
  runCommentaryFromVideo,
  getVoiceOptions,
  previewVoice,
};
