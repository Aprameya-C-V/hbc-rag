import { useEffect, useState } from "react";

type Health = {
  status: string;
  database: { ok: boolean; documents: number; latency_ms: number };
  object_storage: { ok: boolean; bucket: string; roundtrip_ms: number };
};

type Run = {
  id: string;
  model: string;
  provider: string;
  policy: string;
  dataset: string;
  created_at: string;
  metrics: Record<string, unknown>;
};

export default function App() {
  const [health, setHealth] = useState<Health | null>(null);
  const [runs, setRuns] = useState<Run[]>([]);
  const [error, setError] = useState<string>("");

  useEffect(() => {
    async function load() {
      try {
        const [h, r] = await Promise.all([fetch("/health"), fetch("/api/runs?limit=20")]);
        if (!h.ok || !r.ok) throw new Error(`api ${h.status} ${r.status}`);
        setHealth(await h.json());
        const payload = await r.json();
        setRuns(payload.runs ?? []);
      } catch (err) {
        setError(err instanceof Error ? err.message : String(err));
      }
    }
    load();
  }, []);

  return (
    <main>
      <header>
        <p className="eyebrow">Hop-bounded agentic RAG</p>
        <h1>HBC-RAG</h1>
        <p className="sub">Load and cache characterization on rented model APIs.</p>
      </header>

      {error && <div className="banner error">API unreachable: {error}</div>}

      <section className="grid">
        <article className="card">
          <h2>Database</h2>
          {health ? (
            <>
              <p className="metric">{health.database.documents}</p>
              <p className="label">documents</p>
              <p className="note">{health.database.latency_ms} ms round trip</p>
            </>
          ) : (
            <p className="note">waiting</p>
          )}
        </article>
        <article className="card">
          <h2>Object storage</h2>
          {health ? (
            <>
              <p className="metric">{health.object_storage.ok ? "OK" : "FAIL"}</p>
              <p className="label">{health.object_storage.bucket}</p>
              <p className="note">{health.object_storage.roundtrip_ms} ms put+get+delete</p>
            </>
          ) : (
            <p className="note">waiting</p>
          )}
        </article>
        <article className="card">
          <h2>Runs</h2>
          <p className="metric">{runs.length}</p>
          <p className="label">recorded</p>
          <p className="note">traces in Postgres</p>
        </article>
      </section>

      <section className="table-wrap">
        <h2>Recent runs</h2>
        {runs.length === 0 ? (
          <p className="note">No runs yet.</p>
        ) : (
          <table>
            <thead>
              <tr>
                <th>ID</th>
                <th>Model</th>
                <th>Policy</th>
                <th>Dataset</th>
                <th>Created</th>
                <th>Metrics</th>
              </tr>
            </thead>
            <tbody>
              {runs.map((run) => (
                <tr key={run.id}>
                  <td className="mono">{run.id}</td>
                  <td>{run.model}</td>
                  <td>{run.policy}</td>
                  <td>{run.dataset}</td>
                  <td className="mono">{new Date(run.created_at).toLocaleString()}</td>
                  <td className="mono">{JSON.stringify(run.metrics)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </section>
    </main>
  );
}
