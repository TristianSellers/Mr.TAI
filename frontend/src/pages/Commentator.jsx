import React, { useState } from "react";
import { API_BASE } from "../lib/api";

export default function Commentator() {
  const [file, setFile] = useState(null);
  const [form, setForm] = useState({
    home_team: "USC",
    away_team: "UCLA",
    score: "21-24",
    quarter: "Q4",
    clock: "0:42",
    tone: "hype",
    voice: "default",
  });
  const [loading, setLoading] = useState(false);
  const [resp, setResp] = useState(null);
  const [error, setError] = useState(null);

  const onSubmit = async () => {
    if (!file) return setError("Pick an .mp4 first.");
    setError(null);
    setLoading(true);
    setResp(null);

    const body = new FormData();
    body.append("file", file);
    Object.entries(form).forEach(([k, v]) => body.append(k, v));

    try {
      const r = await fetch(`${API_BASE}/analyze_commentate`, { method: "POST", body });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const j = await r.json();
      setResp(j);
    } catch (e) {
      setError(e?.message || "Request failed");
    } finally {
      setLoading(false);
    }
  };

  const staticUrl = (u) => (u ? `${API_BASE}${u}` : "");

  return (
    <div style={{ maxWidth: 720, margin: "40px auto", padding: 16 }}>
      <h1>Mr. TAI — Madden Commentator (MVP)</h1>

      <div style={{ display: "grid", gap: 12, marginTop: 16 }}>
        <input
          type="file"
          accept="video/mp4,video/*"
          onChange={(e) => setFile(e.target.files?.[0] || null)}
        />

        <div style={{ display: "grid", gap: 8, gridTemplateColumns: "1fr 1fr" }}>
          <input placeholder="Home" value={form.home_team}
                 onChange={(e) => setForm({ ...form, home_team: e.target.value })}/>
          <input placeholder="Away" value={form.away_team}
                 onChange={(e) => setForm({ ...form, away_team: e.target.value })}/>
          <input placeholder="Score (e.g., 21-24)" value={form.score}
                 onChange={(e) => setForm({ ...form, score: e.target.value })}/>
          <input placeholder="Quarter (Q1–Q4/OT)" value={form.quarter}
                 onChange={(e) => setForm({ ...form, quarter: e.target.value })}/>
          <input placeholder="Clock (e.g., 0:42)" value={form.clock}
                 onChange={(e) => setForm({ ...form, clock: e.target.value })}/>
          <select value={form.tone} onChange={(e) => setForm({ ...form, tone: e.target.value })}>
            <option value="hype">hype</option>
            <option value="neutral">neutral</option>
            <option value="radio">radio</option>
          </select>
        </div>

        <button onClick={onSubmit} disabled={loading}>
          {loading ? "Analyzing…" : "Analyze & Commentate"}
        </button>

        {error && <div style={{ color: "crimson" }}>{error}</div>}

        {resp && (
          <div style={{ marginTop: 16, display: "grid", gap: 12 }}>
            <div>
              <h3>Commentary Text</h3>
              <p style={{ whiteSpace: "pre-wrap" }}>{resp.text}</p>
            </div>

            {resp.audio_url && (
              <div>
                <h3>Audio</h3>
                <audio controls src={staticUrl(resp.audio_url)} />
                <div><a href={staticUrl(resp.audio_url)} download>Download audio</a></div>
              </div>
            )}

            {resp.video_url && (
              <div>
                <h3>Dubbed Video</h3>
                <video controls src={staticUrl(resp.video_url)} style={{ width: "100%", borderRadius: 8 }} />
                <div><a href={staticUrl(resp.video_url)} download>Download video</a></div>
              </div>
            )}

            <small>Latency: {resp.meta?.duration_s?.toFixed?.(3)}s</small>
            {resp.meta?.errors && <pre style={{ color: "crimson" }}>{JSON.stringify(resp.meta.errors, null, 2)}</pre>}
          </div>
        )}
      </div>
    </div>
  );
}
