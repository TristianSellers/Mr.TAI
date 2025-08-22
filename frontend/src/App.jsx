import { useEffect, useState } from "react";
import { getHealth, uploadFile } from "./api";

export default function App() {
  const [backendMsg, setBackendMsg] = useState("Checking backend…");
  const [uploading, setUploading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    (async () => {
      try {
        const data = await getHealth();
        // data is something like { message: "Hello from Mr. TAI!" }
        setBackendMsg(data.message ?? JSON.stringify(data));
      } catch  {
        setBackendMsg("Failed to reach backend. Is FastAPI running on :8000?");
      }
    })();
  }, []);

  async function handleUpload(e) {
    setError("");
    setResult(null);
    const file = e.target.files?.[0];
    if (!file) return;

    try {
      setUploading(true);
      const resp = await uploadFile(file);
      setResult(resp);
    } catch {
      setError("Upload failed. Check backend logs and CORS.");
    } finally {
      setUploading(false);
    }
  }

  return (
    <div style={{ maxWidth: 640, margin: "2rem auto", fontFamily: "system-ui, sans-serif" }}>
      <h1>Mr. TAI — Frontend</h1>

      <section style={{ marginBottom: "2rem" }}>
        <h2>Backend status</h2>
        <p>{backendMsg}</p>
      </section>

      <section>
        <h2>Upload a file</h2>
        <input type="file" onChange={handleUpload} disabled={uploading} />
        {uploading && <p>Uploading…</p>}
        {error && <p style={{ color: "red" }}>{error}</p>}
        {result && (
          <pre
            style={{
              background: "#f5f5f5",
              padding: "0.75rem",
              borderRadius: 8,
              overflowX: "auto",
              marginTop: "1rem",
            }}
          >
            {JSON.stringify(result, null, 2)}
          </pre>
        )}
      </section>
    </div>
  );
}
