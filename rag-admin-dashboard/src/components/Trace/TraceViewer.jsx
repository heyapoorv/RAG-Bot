export default function TraceViewer({ trace }) {
  return (
    <div className="trace">
      <h3>RAG Trace</h3>

      {trace?.chunks?.map((c, i) => (
        <div key={i} className="trace-item">
          <b>{c.source}</b>
          <p>{c.chunk_text}</p>
        </div>
      ))}
    </div>
  );
}