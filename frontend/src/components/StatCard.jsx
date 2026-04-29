export default function StatCard({ label, value, delta, icon, highlight }) {
  const deltaPositive = delta > 0;
  const showDelta = delta !== undefined && delta !== null;

  return (
    <div className={`stat-card ${highlight ? "highlight" : ""}`}>
      <div className="stat-icon">{icon}</div>
      <div className="stat-body">
        <span className="stat-label">{label}</span>
        <span className="stat-value">{value}</span>
        {showDelta && (
          <span className={`stat-delta ${deltaPositive ? "up" : "down"}`}>
            {deltaPositive ? "▲" : "▼"} {Math.abs(delta)}
          </span>
        )}
      </div>
    </div>
  );
}
