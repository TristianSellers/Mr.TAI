// frontend/src/lib/api.js
const RAW_API_BASE = import.meta?.env?.VITE_API_BASE ?? "http://localhost:8000";
// trim any trailing slashes
export const API_BASE = RAW_API_BASE.replace(/\/+$/, "");

export async function getHealth() {
  const res = await fetch(`${API_BASE}/health`);
  if (!res.ok) throw new Error(`Health check failed: ${res.status}`);
  return res.json();
}

export async function uploadFile(file) {
  const form = new FormData();
  form.append("file", file);

  const res = await fetch(`${API_BASE}/upload`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) throw new Error(`Upload failed: ${res.status}`);
  return res.json();
}

export async function analyzeCommentate({ file, tone = "play-by-play", audioOnly = false, extra = {} }) {
  const API_BASE = import.meta?.env?.VITE_API_BASE?.replace(/\/+$/, "") || "http://127.0.0.1:8000";
  const form = new FormData();
  if (file) form.append("file", file);
  form.append("tone", tone);
  form.append("audio_only", String(audioOnly));
  // optional extras (home_team, score, etc.)
  for (const [k, v] of Object.entries(extra)) {
    if (v != null && v !== "") form.append(k, v);
  }

  const res = await fetch(`${API_BASE}/analyze_commentate`, { method: "POST", body: form });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`analyze_commentate failed: ${res.status} ${text}`);
  }
  return res.json();
}
