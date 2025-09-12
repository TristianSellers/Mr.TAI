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
