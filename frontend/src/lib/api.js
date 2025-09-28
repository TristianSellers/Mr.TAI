// frontend/src/lib/api.js
// Centralized API helpers for Mr. TAI frontend

// Base URL (no trailing slash)
const RAW_API_BASE = import.meta?.env?.VITE_API_BASE ?? "http://localhost:8000";
export const API_BASE = RAW_API_BASE.replace(/\/+$/, "");

// ---- Tone / Bias defaults from env ----
export const DEFAULT_TONE = import.meta?.env?.VITE_TONE_DEFAULT ?? "neutral";
export const DEFAULT_BIAS = import.meta?.env?.VITE_BIAS_DEFAULT ?? "neutral";

// ---- Utilities ----
function toUrl(path) {
  return `${API_BASE}${path.startsWith("/") ? path : `/${path}`}`;
}

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
  return httpJson(toUrl("/health"));
}

// ---- OCR (optional helpers for debugging/manual use) ----
export async function runOcrImage(file, { viz = false, debug = false, dx = 0, dy = 0 } = {}) {
  const fd = new FormData();
  fd.append("image", file);
  fd.append("viz", String(viz));
  fd.append("debug", String(debug));
  fd.append("dx", String(dx));
  fd.append("dy", String(dy));
  return httpJson(toUrl("/ocr/image"), { method: "POST", body: fd });
}

export async function runOcrVideo(file, { viz = false, dx = 0, dy = 0, t = 0.1 } = {}) {
  const fd = new FormData();
  fd.append("video", file);
  fd.append("viz", String(viz));
  fd.append("dx", String(dx));
  fd.append("dy", String(dy));
  fd.append("t", String(t));
  return httpJson(toUrl("/ocr/video"), { method: "POST", body: fd });
}

// ---- One-shot pipeline (recommended) ----
// Upload *only* a video; backend does OCR → LLM → TTS → (optional) mux.
export async function runCommentaryFromVideo(
  file,
  {
    tone = DEFAULT_TONE,
    bias = DEFAULT_BIAS,
    audioOnly = false,
    viz = false,
    dx = 0,
    dy = 0,
    t = 0.1,
  } = {}
) {
  const fd = new FormData();
  fd.append("video", file);
  fd.append("tone", tone);
  fd.append("bias", bias);
  fd.append("audio_only", String(audioOnly));
  fd.append("viz", String(viz));
  fd.append("dx", String(dx));
  fd.append("dy", String(dy));
  fd.append("t", String(t));
  return httpJson(toUrl("/pipeline/run-commentary-from-video"), { method: "POST", body: fd });
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
    tone = DEFAULT_TONE,
    bias = DEFAULT_BIAS,
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
  return httpJson(toUrl("/analyze_commentate"), { method: "POST", body: fd });
}

// ---- Helpers for building absolute static URLs in the UI ----
export function staticUrl(pathOrNull) {
  return pathOrNull ? toUrl(pathOrNull) : "";
}
