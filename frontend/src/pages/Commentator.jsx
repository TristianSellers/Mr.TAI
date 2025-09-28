// frontend/src/pages/Commentator.jsx
import React, { useState } from "react";
import { API_BASE, runCommentaryFromVideo, DEFAULT_TONE, DEFAULT_BIAS } from "../lib/api";

export default function Commentator() {
  const [file, setFile] = useState(null);
  const [tone, setTone] = useState(String(DEFAULT_TONE || "neutral"));   // neutral | hype | radio
  const [bias, setBias] = useState(String(DEFAULT_BIAS || "neutral"));   // neutral | home | away
  const [loading, setLoading] = useState(false);
  const [resp, setResp] = useState(null);
  const [err, setErr] = useState("");

  const canRun = !!file;
  const staticUrl = (u) => (u ? `${API_BASE}${u}` : "");

  async function onRun() {
    if (!file) return;
    setErr(""); setResp(null); setLoading(true);
    try {
      const data = await runCommentaryFromVideo(file, { tone, bias, audioOnly: false });
      setResp(data);
    } catch (e) {
      setErr(String(e.message || e));
    } finally {
      setLoading(false);
    }
  }

  function onReset() {
    setFile(null); setResp(null); setErr("");
  }

  return (
    <div style={{ maxWidth: 840, margin: "40px auto", padding: 16 }}>
      <h1 style={{ fontSize: 22, fontWeight: 700, marginBottom: 12 }}>Mr. TAI • One-Shot Commentary</h1>

      <div style={{ display: "grid", gap: 10 }}>
        <input
          type="file"
          accept="video/*"
          onChange={(e) => setFile(e.target.files?.[0] || null)}
          disabled={loading}
        />

        <div style={{ display: "flex", gap: 12, flexWrap: "wrap", alignItems: "center" }}>
          <label>
            Tone:&nbsp;
            <select value={tone} onChange={(e) => setTone(e.target.value)} disabled={loading}>
              <option value="neutral">neutral (ElevenLabs)</option>
              <option value="hype">hype (TypeCast)</option>
              <option value="radio">radio (TypeCast)</option>
            </select>
          </label>
          <label>
            Bias:&nbsp;
            <select value={bias} onChange={(e) => setBias(e.target.value)} disabled={loading}>
              <option value="neutral">neutral</option>
              <option value="home">home</option>
              <option value="away">away</option>
            </select>
          </label>

          <button onClick={onRun} disabled={!canRun || loading} style={{ padding: "8px 12px" }}>
            {loading ? "Running…" : "Run Commentary"}
          </button>
          <button onClick={onReset} disabled={loading} style={{ padding: "8px 12px" }}>
            Reset
          </button>
        </div>
      </div>

      {!canRun && <div style={{ marginTop: 8, color: "#b45309", fontSize: 12 }}>Select a video to continue.</div>}
      {err && <div style={{ marginTop: 12, color: "crimson" }}>{err}</div>}

      {resp && (
        <div style={{ marginTop: 20, display: "grid", gap: 16 }}>
          <div style={{ padding: 12, border: "1px solid #ddd", borderRadius: 8 }}>
            <div style={{ fontWeight: 600, marginBottom: 6 }}>Commentary Text</div>
            <pre style={{ whiteSpace: "pre-wrap", margin: 0 }}>{resp.text}</pre>
          </div>

          <div style={{ display: "grid", gap: 16 }}>
            {resp.audio_url && (
              <div style={{ padding: 12, border: "1px solid #ddd", borderRadius: 8 }}>
                <div style={{ fontWeight: 600, marginBottom: 6 }}>Audio</div>
                <audio controls src={staticUrl(resp.audio_url)} />
              </div>
            )}
            {resp.video_url && (
              <div style={{ padding: 12, border: "1px solid #ddd", borderRadius: 8 }}>
                <div style={{ fontWeight: 600, marginBottom: 6 }}>Dubbed Video</div>
                <video controls width={640} src={staticUrl(resp.video_url)} />
              </div>
            )}
          </div>

          <div style={{ padding: 12, border: "1px solid #eee", borderRadius: 8 }}>
            <div style={{ fontWeight: 600, marginBottom: 6 }}>Meta</div>
            <pre style={{ margin: 0, fontSize: 12 }}>{JSON.stringify(resp.meta, null, 2)}</pre>
          </div>
        </div>
      )}
    </div>
  );
}
