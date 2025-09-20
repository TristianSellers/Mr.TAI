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
    bias: "neutral",
    voice: "default",
  });
  const [audioOnly, setAudioOnly] = useState(false);
  const [loading, setLoading] = useState(false);
  const [resp, setResp] = useState(null);
  const [error, setError] = useState(null);

  const staticUrl = (u) => (u ? `${API_BASE}${u}` : "");

  async function onSubmit() {
    if (!file && !audioOnly) {
      return setError("Select a clip or enable “Audio only”.");
    }
    setError(null);
    setLoading(true);
    setResp(null);

    const body = new FormData();
    if (file) body.append("file", file);
    Object.entries(form).forEach(([k, v]) => body.append(k, v));
    body.append("audio_only", String(audioOnly));

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
  }

  async function onDemoMode() {
    setError(null);
    setLoading(true);
    setResp(null);
    try {
      const r = await fetch(`${API_BASE}/demo/artifacts`);
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const j = await r.json();
      setResp({
        id: "demo",
        text: j.text || "Demo commentary",
        audio_url: j.audio_url || null,
        video_url: j.video_url || null,
        meta: { duration_s: 0, errors: null, usedManualContext: false },
      });
    } catch (e) {
      setError(e?.message || "Failed to load demo artifacts");
    } finally {
      setLoading(false);
    }
  }

  function onReset() {
    setFile(null);
    setResp(null);
    setError(null);
    setAudioOnly(false);
    setForm({
      home_team: "USC",
      away_team: "UCLA",
      score: "21-24",
      quarter: "Q4",
      clock: "0:42",
      tone: "hype",
      bias: "neutral",
      voice: "default",
    });
  }

  const canAnalyze = !!file || audioOnly;
  const labelStyle = { display: "grid", gap: 6, fontSize: 12, color: "#374151" };
  const inputStyle = { padding: 8, border: "1px solid #d1d5db", borderRadius: 6 };

  return (
    <div style={{ maxWidth: 720, margin: "40px auto", padding: 16 }}>
      <h1>Mr. TAI — Madden Commentator (MVP)</h1>

      <div style={{ display: "grid", gap: 12, marginTop: 16 }}>
        <div style={{ display: "flex", gap: 12, alignItems: "center" }}>
          <label style={{ ...labelStyle, margin: 0 }}>
            <span>Clip (video/audio)</span>
            <input
              type="file"
              accept="video/mp4,video/*,audio/*"
              onChange={(e) => setFile(e.target.files?.[0] || null)}
              disabled={loading}
            />
          </label>

          <label style={{ display: "flex", alignItems: "center", gap: 6 }}>
            <input
              type="checkbox"
              checked={audioOnly}
              onChange={(e) => setAudioOnly(e.target.checked)}
              disabled={loading}
            />
            <span>Audio only (no clip)</span>
          </label>
        </div>

        <div style={{ display: "grid", gap: 10, gridTemplateColumns: "1fr 1fr" }}>
          <label style={labelStyle}>
            <span>Home Team</span>
            <input
              style={inputStyle}
              value={form.home_team}
              onChange={(e) => setForm({ ...form, home_team: e.target.value })}
              disabled={loading}
            />
          </label>

          <label style={labelStyle}>
            <span>Away Team</span>
            <input
              style={inputStyle}
              value={form.away_team}
              onChange={(e) => setForm({ ...form, away_team: e.target.value })}
              disabled={loading}
            />
          </label>

          <label style={labelStyle}>
            <span>Score</span>
            <input
              style={inputStyle}
              placeholder="e.g., 21-24"
              value={form.score}
              onChange={(e) => setForm({ ...form, score: e.target.value })}
              disabled={loading}
            />
          </label>

          <label style={labelStyle}>
            <span>Quarter</span>
            <input
              style={inputStyle}
              placeholder="Q1–Q4/OT"
              value={form.quarter}
              onChange={(e) => setForm({ ...form, quarter: e.target.value })}
              disabled={loading}
            />
          </label>

          <label style={labelStyle}>
            <span>Clock</span>
            <input
              style={inputStyle}
              placeholder="e.g., 0:42"
              value={form.clock}
              onChange={(e) => setForm({ ...form, clock: e.target.value })}
              disabled={loading}
            />
          </label>

          <div style={{ display: "grid", gap: 10 }}>
            <label style={labelStyle}>
              <span>Tone</span>
              <select
                style={inputStyle}
                value={form.tone}
                onChange={(e) => setForm({ ...form, tone: e.target.value })}
                disabled={loading}
              >
                <option value="hype">hype</option>
                <option value="neutral">neutral</option>
                <option value="radio">radio</option>
              </select>
            </label>

            <label style={labelStyle}>
              <span>Bias (POV)</span>
              <select
                style={inputStyle}
                value={form.bias}
                onChange={(e) => setForm({ ...form, bias: e.target.value })}
                disabled={loading}
                title="Commentator POV bias"
              >
                <option value="neutral">neutral</option>
                <option value="home">home</option>
                <option value="away">away</option>
              </select>
            </label>
          </div>
        </div>

        <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
          <button onClick={onSubmit} disabled={loading || !canAnalyze}>
            {loading ? "Working…" : "Analyze & Commentate"}
          </button>
          <button onClick={onDemoMode} disabled={loading}>
            Demo Mode (prefilled backup)
          </button>
          <button onClick={onReset} disabled={loading}>
            Reset
          </button>
          {!canAnalyze && (
            <span style={{ marginLeft: 8, color: "#b45309", fontSize: 12 }}>
              Select a clip or enable “Audio only”.
            </span>
          )}
        </div>

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
                <video
                  controls
                  src={staticUrl(resp.video_url)}
                  style={{ width: "100%", borderRadius: 8 }}
                />
                <div><a href={staticUrl(resp.video_url)} download>Download video</a></div>
              </div>
            )}

            <small>
              Latency: {resp.meta?.duration_s != null ? resp.meta.duration_s.toFixed?.(3) : "—"}s
              {resp.meta?.usedManualContext ? " • manual context" : ""}
              {resp.meta?.audio_only ? " • audio-only" : ""}
              {resp.meta?.prompt_tone ? ` • tone: ${resp.meta.prompt_tone}` : ""}
              {resp.meta?.prompt_bias ? ` • bias: ${resp.meta.prompt_bias}` : ""}
            </small>
            {resp.meta?.errors && (
              <pre style={{ color: "crimson" }}>{JSON.stringify(resp.meta.errors, null, 2)}</pre>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
