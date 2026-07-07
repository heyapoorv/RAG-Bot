export default function Message({ msg }) {
  return (
    <div className={`msg ${msg.role}`}>
      <div className="text-base leading-relaxed">{msg.text}</div>

      {msg.sources?.length > 0 && (
        <div className="mt-3 pt-3 border-t border-secondary">
          <div className="text-xs text-muted mb-2">Sources:</div>
          <div className="flex flex-wrap gap-2">
            {msg.sources.map((s, i) => (
              <span key={i} className="inline-flex items-center gap-1 text-xs bg-secondary px-2 py-1 rounded">
                📄 {s}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}