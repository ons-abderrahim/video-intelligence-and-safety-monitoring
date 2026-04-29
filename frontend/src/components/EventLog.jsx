import { useState } from "react";

const EVENT_ICONS = {
  violence_detected: "🔴",
  ppe_violation: "🟡",
  zone_intrusion: "🔵",
  normal: "⚪",
};

const SEVERITY_ORDER = ["violence_detected", "ppe_violation", "zone_intrusion", "normal"];

function EventRow({ event, onAcknowledge }) {
  const icon = EVENT_ICONS[event.event_type] ?? "⚪";
  const time = new Date(event.timestamp).toLocaleTimeString();
  const confPct = `${(event.confidence * 100).toFixed(0)}%`;

  return (
    <div className={`event-row ${event.acknowledged ? "ack" : ""} type-${event.event_type}`}>
      <span className="ev-icon">{icon}</span>
      <span className="ev-time">{time}</span>
      <span className="ev-camera">{event.camera_id}</span>
      <span className="ev-type">{event.event_type.replace(/_/g, " ")}</span>
      <span className="ev-conf">{confPct}</span>
      {!event.acknowledged && (
        <button
          className="ev-ack-btn"
          onClick={() => onAcknowledge(event.id)}
          title="Acknowledge"
        >
          ✓
        </button>
      )}
      {event.acknowledged && <span className="ev-acked">✓ ack</span>}
    </div>
  );
}

export default function EventLog({ events, loading }) {
  const [filter, setFilter] = useState("all");
  const [acknowledged, setAcknowledged] = useState(new Set());

  const filtered =
    filter === "all"
      ? events
      : events.filter((e) => e.event_type === filter);

  const handleAck = (id) => {
    setAcknowledged((prev) => new Set([...prev, id]));
    // In production: call PATCH /api/events/:id/acknowledge
  };

  return (
    <div className="event-log">
      <div className="log-header">
        <h2>Event Log</h2>
        <div className="log-filters">
          {["all", ...SEVERITY_ORDER.slice(0, 3)].map((f) => (
            <button
              key={f}
              className={`filter-btn ${filter === f ? "active" : ""}`}
              onClick={() => setFilter(f)}
            >
              {f === "all" ? "All" : f.replace(/_/g, " ")}
            </button>
          ))}
        </div>
      </div>

      <div className="log-body">
        {loading && (
          <div className="log-loading">Loading events…</div>
        )}

        {!loading && filtered.length === 0 && (
          <div className="log-empty">
            <span>✅</span>
            <p>No events matching this filter.</p>
          </div>
        )}

        {filtered.map((ev) => (
          <EventRow
            key={ev.id}
            event={{ ...ev, acknowledged: ev.acknowledged || acknowledged.has(ev.id) }}
            onAcknowledge={handleAck}
          />
        ))}
      </div>

      <div className="log-footer">
        Showing {filtered.length} of {events.length} events
      </div>
    </div>
  );
}
