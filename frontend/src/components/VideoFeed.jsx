import { useEffect, useRef, useState } from "react";
import { useStream } from "../hooks/useStream";

/**
 * VideoFeed
 * ---------
 * Connects to the backend WebSocket stream for a given camera
 * and renders frames onto an HTML canvas with event overlay boxes.
 */
export default function VideoFeed({ cameraId }) {
  const canvasRef = useRef(null);
  const { lastEvent, connected, error } = useStream(cameraId);
  const [overlayEvent, setOverlayEvent] = useState(null);

  // Flash overlay when a new event arrives
  useEffect(() => {
    if (!lastEvent) return;
    setOverlayEvent(lastEvent);
    const timer = setTimeout(() => setOverlayEvent(null), 3000);
    return () => clearTimeout(timer);
  }, [lastEvent]);

  const borderColor = overlayEvent
    ? overlayEvent.event_type === "violence_detected"
      ? "#ef4444"
      : overlayEvent.event_type === "ppe_violation"
      ? "#f59e0b"
      : "#3b82f6"
    : connected
    ? "#22c55e"
    : "#6b7280";

  return (
    <div className="video-feed" style={{ borderColor }}>
      {/* Status bar */}
      <div className="feed-status">
        <span
          className="conn-dot"
          style={{ background: connected ? "#22c55e" : "#ef4444" }}
        />
        <span>{cameraId}</span>
        {connected ? (
          <span className="conn-label">Connected</span>
        ) : (
          <span className="conn-label error">Disconnected</span>
        )}
      </div>

      {/* Canvas — placeholder for actual frame rendering */}
      <canvas
        ref={canvasRef}
        className="feed-canvas"
        width={640}
        height={360}
      />

      {/* Demo placeholder when no real stream */}
      <div className="feed-placeholder">
        <span className="camera-icon">📷</span>
        <span>{cameraId}</span>
        <span className="feed-hint">Awaiting stream…</span>
      </div>

      {/* Event flash overlay */}
      {overlayEvent && (
        <div className={`event-overlay event-${overlayEvent.event_type}`}>
          <span className="overlay-icon">
            {overlayEvent.event_type === "violence_detected"
              ? "⚠️ VIOLENCE"
              : overlayEvent.event_type === "ppe_violation"
              ? "🦺 PPE VIOLATION"
              : "🚧 ZONE BREACH"}
          </span>
          <span className="overlay-conf">
            {(overlayEvent.confidence * 100).toFixed(0)}% confidence
          </span>
        </div>
      )}

      {error && (
        <div className="feed-error">
          ⚠ {error}
        </div>
      )}
    </div>
  );
}
