import { useState, useEffect, useRef } from "react";
import VideoFeed from "../components/VideoFeed";
import EventLog from "../components/EventLog";
import StatCard from "../components/StatCard";
import HeatmapOverlay from "../components/HeatmapOverlay";
import { useEvents } from "../hooks/useEvents";

const CAMERA_IDS = ["cam-01", "cam-02", "cam-03", "cam-04"];

export default function Dashboard() {
  const [activeCamera, setActiveCamera] = useState("cam-01");
  const { events, stats, loading } = useEvents();

  const alertEvents = events.filter((e) => e.event_type !== "normal");
  const recentAlerts = alertEvents.slice(0, 5);

  return (
    <div className="dashboard">
      {/* Header */}
      <header className="dashboard-header">
        <div className="header-left">
          <span className="brand-dot" />
          <h1>VISP Monitor</h1>
          <span className="status-pill live">● LIVE</span>
        </div>
        <div className="header-right">
          <span className="timestamp" suppressHydrationWarning>
            {new Date().toLocaleTimeString()}
          </span>
        </div>
      </header>

      {/* KPI row */}
      <section className="kpi-row">
        <StatCard
          label="Events Today"
          value={stats.total}
          delta={stats.deltaTotal}
          icon="📋"
        />
        <StatCard
          label="Alerts"
          value={stats.alerts}
          delta={stats.deltaAlerts}
          icon="⚠️"
          highlight={stats.alerts > 0}
        />
        <StatCard
          label="Active Cameras"
          value={stats.activeCameras}
          icon="📷"
        />
        <StatCard
          label="Avg Confidence"
          value={`${(stats.avgConfidence * 100).toFixed(0)}%`}
          icon="🎯"
        />
      </section>

      {/* Main grid */}
      <main className="main-grid">
        {/* Camera selector + feed */}
        <section className="feed-panel">
          <div className="camera-tabs">
            {CAMERA_IDS.map((id) => (
              <button
                key={id}
                className={`cam-tab ${activeCamera === id ? "active" : ""}`}
                onClick={() => setActiveCamera(id)}
              >
                {id}
              </button>
            ))}
          </div>

          <div className="feed-container">
            <VideoFeed cameraId={activeCamera} />
            <HeatmapOverlay cameraId={activeCamera} events={events} />
          </div>

          {/* Recent alerts */}
          <div className="alert-strip">
            <h3>Recent Alerts</h3>
            {recentAlerts.length === 0 ? (
              <p className="no-alerts">✅ No recent alerts</p>
            ) : (
              recentAlerts.map((ev) => (
                <div key={ev.id} className={`alert-chip severity-${ev.event_type}`}>
                  <span className="alert-icon">
                    {ev.event_type === "violence_detected" ? "🔴" :
                     ev.event_type === "ppe_violation" ? "🟡" : "🔵"}
                  </span>
                  <span className="alert-label">{ev.event_type.replace(/_/g, " ")}</span>
                  <span className="alert-conf">{(ev.confidence * 100).toFixed(0)}%</span>
                  <span className="alert-cam">{ev.camera_id}</span>
                </div>
              ))
            )}
          </div>
        </section>

        {/* Event log */}
        <section className="log-panel">
          <EventLog events={events} loading={loading} />
        </section>
      </main>
    </div>
  );
}
