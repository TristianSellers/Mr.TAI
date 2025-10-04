import React, { useEffect, useState } from "react";
import { staticUrl, runCommentaryFromVideo, getVoiceOptions, previewVoice } from "../lib/api";

// Fixed mapping from your backend spec:
// neutral → Dean (male), Caitlyn (female)
// radio   → Walter (male), Nia (female)
// hype    → Matthew (male), Rachel (female)
const MAPPED_BY_VOICE_ID = {
  // Dean / Caitlyn
  "tc_673eb45cdc1073aef51e6b90": { tone: "neutral", gender: "male" },
  "tc_6412c42d733f60ab8ad369a9": { tone: "neutral", gender: "female" },
  // Walter / Nia
  "tc_6837b58f80ceeb17115bb771": { tone: "radio", gender: "male" },
  "tc_684a5a7ba2ce934624b59c6e": { tone: "radio", gender: "female" },
  // Matthew / Rachel
  "tc_623145940c2c1ff30c30f3a9": { tone: "hype", gender: "male" },
  "tc_630494521f5003bebbfdafef": { tone: "hype", gender: "female" },
};

export default function Commentator() {
  const [file, setFile] = useState(null);
  const [tone, setTone] = useState("hype");
  const [bias, setBias] = useState("neutral");
  const [gender, setGender] = useState("male");
  const [voices, setVoices] = useState([]);
  const [loading, setLoading] = useState(false);
  const [resp, setResp] = useState(null);
  const [error, setError] = useState(null);
  const [previewUrl, setPreviewUrl] = useState("");

  const label = { display: "grid", gap: 6, fontSize: 12, color: "#374151", lineHeight: 1.25 };
  const input = { padding: 8, border: "1px solid #d1d5db", borderRadius: 6 };

  // refresh voice list when tone changes (we still show all voices; suggested_emotion adapts to tone)
  useEffect(() => {
    (async () => {
      try {
        const j = await getVoiceOptions({ tone, gender: undefined });
        setVoices(j.voices);
      } catch (e) {
        console.error(e);
      }
    })();
  }, [tone]);

  async function onAnalyze() {
    if (!file) return setError("Select a video first.");
    setError(null); setResp(null); setLoading(true);
    try {
      const data = await runCommentaryFromVideo({ file, tone, bias, gender });
      setResp(data);
    } catch (e) {
      setError(e?.message || "Request failed");
    } finally {
      setLoading(false);
    }
  }

  async function onPreviewSpecific(v) {
    setError(null); setPreviewUrl("");
    try {
      const data = await previewVoice({
        tone, bias, gender,
        voiceId: v.id,
        emotion: v.suggested_emotion,
        text: "Mic check. One-two.",
      });
      if (data.audio_url) setPreviewUrl(staticUrl(data.audio_url));
    } catch (e) {
      setError(e?.message || "Preview failed");
    }
  }

  return (
    <div style={{ maxWidth: 720, margin: "40px auto", padding: 16 }}>
      <h1>Mr. TAI — Madden Commentator (MVP)</h1>

      <div style={{ display: "grid", gap: 12, marginTop: 16 }}>
        <label style={label}>
          <span>Video</span>
          <input type="file" accept="video/*" onChange={(e)=>setFile(e.target.files?.[0]||null)} disabled={loading}/>
        </label>

        <div style={{ display: "grid", gap: 10, gridTemplateColumns: "1fr 1fr 1fr" }}>
          <label style={label}>
            <span>Tone</span>
            <select style={input} value={tone} onChange={(e)=>setTone(e.target.value)} disabled={loading}>
              <option value="hype">hype</option>
              <option value="neutral">neutral</option>
              <option value="radio">radio</option>
            </select>
          </label>
          <label style={label}>
            <span>Bias</span>
            <select style={input} value={bias} onChange={(e)=>setBias(e.target.value)} disabled={loading}>
              <option value="neutral">neutral</option>
              <option value="home">home</option>
              <option value="away">away</option>
            </select>
          </label>
          <label style={label}>
            <span>Gender</span>
            <select style={input} value={gender} onChange={(e)=>setGender(e.target.value)} disabled={loading}>
              <option value="male">male</option>
              <option value="female">female</option>
            </select>
          </label>
        </div>

        {/* Voice gallery: all six, with suggested emotion and mapped tone/gender */}
        <div style={{ display: "grid", gap: 8 }}>
          <div style={{ fontSize: 12, color: "#374151" }}>Voices (click Preview to audition):</div>
          <div style={{ display: "grid", gap: 8, gridTemplateColumns: "repeat(auto-fill, minmax(180px, 1fr))" }}>
            {voices.map(v => {
              const mapping = MAPPED_BY_VOICE_ID[v.id];
              return (
                <div key={v.id} style={{ border: "1px solid #e5e7eb", borderRadius: 8, padding: 10 }}>
                  <div style={{ fontWeight: 600 }}>{v.name}</div>
                  <div style={{ fontSize: 12, color: "#6b7280" }}>
                    {v.gender} • {v.emotions.join(", ")}
                  </div>
                  {mapping && (
                    <div style={{ fontSize: 12, color: "#065f46", marginTop: 4 }}>
                      Mapped to: <strong>{mapping.tone}</strong> • <strong>{mapping.gender}</strong>
                    </div>
                  )}
                  <div style={{ fontSize: 12, color: "#6b7280", marginTop: 4 }}>
                    suggested: {v.suggested_emotion}
                  </div>
                  <button style={{ marginTop: 8 }} onClick={() => onPreviewSpecific(v)} disabled={loading}>
                    Preview
                  </button>
                </div>
              );
            })}
          </div>
        </div>

        <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
          <button onClick={onAnalyze} disabled={loading || !file}>
            {loading ? "Working…" : "Analyze & Commentate"}
          </button>
        </div>

        {error && <div style={{ color: "crimson" }}>{error}</div>}

        {previewUrl && (
          <div>
            <h3>Voice Preview</h3>
            <audio controls src={previewUrl} />
          </div>
        )}

        {resp && (
          <div style={{ marginTop: 16, display: "grid", gap: 12 }}>
            <div>
              <h3>Commentary Text</h3>
              <p style={{ whiteSpace: "pre-wrap" }}>{resp.text}</p>
            </div>
            {resp.audio_url && (<div><h3>Audio</h3><audio controls src={staticUrl(resp.audio_url)} /></div>)}
            {resp.video_url && (<div><h3>Dubbed Video</h3><video controls src={staticUrl(resp.video_url)} style={{ width: "100%", borderRadius: 8 }} /></div>)}
            <small>
              {resp.meta?.prompt_tone ? `tone: ${resp.meta.prompt_tone}` : ""}
              {resp.meta?.prompt_bias ? ` • bias: ${resp.meta.prompt_bias}` : ""}
              {resp.meta?.gender ? ` • gender: ${resp.meta.gender}` : ""}
            </small>
          </div>
        )}
      </div>
    </div>
  );
}
