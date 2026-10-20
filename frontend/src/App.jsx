
import { useCallback, useEffect, useState } from "react";
import "./App.css";

const API = "http://127.0.0.1:8000";

function formatDate(value) {
  if (!value) return "—";
  return new Date(value.replace(" ", "T") + "Z").toLocaleString();
}

function StatCard({ label, value, note, icon }) {
  return (
    <article className="stat-card">
      <div className="stat-top">
        <span>{label}</span>
        <span className="stat-icon">{icon}</span>
      </div>
      <div className="stat-value">{value}</div>
      <div className="stat-note">{note}</div>
    </article>
  );
}

function App() {
  const [events, setEvents] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [updatingId, setUpdatingId] = useState(null);
  const [filter, setFilter] = useState("ALL");

  const loadEvents = useCallback(async () => {
    try {
      const response = await fetch(`${API}/events?limit=100`);
      if (!response.ok) throw new Error(`API error: ${response.status}`);
      const data = await response.json();
      setEvents(data);
      setError("");
    } catch {
      setError("Cannot reach the backend. Check that FastAPI is running at 127.0.0.1:8000.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadEvents();
    const timer = setInterval(loadEvents, 10000);
    return () => clearInterval(timer);
  }, [loadEvents]);

  async function reviewEvent(id, status) {
    setUpdatingId(id);
    try {
      const response = await fetch(`${API}/events/${id}/review`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ status }),
      });

      if (!response.ok) {
        throw new Error(`Review failed: ${response.status}`);
      }

      const updated = await response.json();
      setEvents((current) =>
        current.map((event) => (event.id === id ? updated : event))
      );
    } catch (err) {
      setError(err.message || "Could not update the event.");
    } finally {
      setUpdatingId(null);
    }
  }

  const pending = events.filter((event) => event.owner_status === "PENDING");
  const highPriority = events.filter((event) => event.severity === "HIGH");
  const filteredEvents =
    filter === "ALL"
      ? events
      : filter === "PENDING"
        ? pending
        : events.filter((event) => event.severity === filter);

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">🌾</div>
          <div>
            <h1>SmartFarm</h1>
            <span>GUARDIAN SYSTEM</span>
          </div>
        </div>

        <div className="side-label">WORKSPACE</div>
        <a className="nav-item active" href="#overview">▦ &nbsp; Overview</a>
        <a className="nav-item" href="#events">◉ &nbsp; Security events</a>
        <a className="nav-item" href="#review">✓ &nbsp; Owner review</a>

        <div className="sidebar-bottom">
          <div className="system-indicator">
            <span className="pulse-dot" />
            <div>
              <strong>API connection</strong>
              <small>{error ? "Connection issue" : "Checking status…"}</small>
            </div>
          </div>
          <p>AI-assisted farm security</p>
          <p className="version">Prototype · v1.0</p>
        </div>
      </aside>

      <main className="main-content">
        <header className="topbar">
          <div>
            <div className="eyebrow">FARM SECURITY / CONTROL CENTER</div>
            <h2>Security overview</h2>
            <p>Monitor events, review alerts, and track farm activity.</p>
          </div>
          <button className="refresh-button" onClick={loadEvents}>
            ↻ &nbsp; Refresh
          </button>
        </header>

        {error && <div className="error-banner">⚠ {error}</div>}

        <section id="overview" className="stats-grid">
          <StatCard label="Total events" value={events.length} note="Stored event records" icon="◷" />
          <StatCard label="High priority" value={highPriority.length} note="Events marked HIGH" icon="⚠" />
          <StatCard label="Pending review" value={pending.length} note="Awaiting owner action" icon="◎" />
          <StatCard label="Reviewed" value={events.filter((e) => e.owner_status !== "PENDING").length} note="Approved or dismissed" icon="✓" />
        </section>

        <section className="content-panel" id="events">
          <div className="panel-heading">
            <div>
              <div className="eyebrow">ACTIVITY MONITOR</div>
              <h3>Security event history</h3>
              <p>Events loaded from your FastAPI + SQLite backend.</p>
            </div>
            <span className="live-badge"><span className="pulse-dot" /> AUTO-REFRESH 10s</span>
          </div>

          <div className="filter-row">
            {["ALL", "PENDING", "HIGH", "MEDIUM", "LOW"].map((item) => (
              <button
                key={item}
                className={`filter-chip ${filter === item ? "selected" : ""}`}
                onClick={() => setFilter(item)}
              >
                {item === "ALL" ? "All events" : item === "PENDING" ? "Pending review" : `${item} severity`}
              </button>
            ))}
          </div>

          {loading ? (
            <div className="empty-state">Loading security events…</div>
          ) : filteredEvents.length === 0 ? (
            <div className="empty-state">No events match this filter.</div>
          ) : (
            <div className="event-list">
              {filteredEvents.map((event) => (
                <article className="event-card" key={event.id}>
                  <div className={`severity-marker ${event.severity.toLowerCase()}`} />
                  <div className="event-main">
                    <div className="event-title-row">
                      <h4>{event.zone.replaceAll("_", " ")}</h4>
                      <span className={`severity-pill ${event.severity.toLowerCase()}`}>
                        {event.severity}
                      </span>
                    </div>
                    <p className="event-message">{event.message}</p>
                    <p className="event-reasons">{event.reasons || "No additional reasons recorded."}</p>
                    <div className="event-meta">
                      <span>EVENT #{event.id}</span>
                      <span>{formatDate(event.created_at)}</span>
                      <span>Worker: {event.worker_status}</span>
                    </div>
                  </div>
                  <div className="event-actions" id="review">
                    <span className={`status-label ${event.owner_status.toLowerCase()}`}>
                      {event.owner_status}
                    </span>
                    {event.owner_status === "PENDING" && (
                      <div className="action-buttons">
                        <button
                          className="approve-button"
                          disabled={updatingId === event.id}
                          onClick={() => reviewEvent(event.id, "APPROVED")}
                        >
                          Approve
                        </button>
                        <button
                          className="dismiss-button"
                          disabled={updatingId === event.id}
                          onClick={() => reviewEvent(event.id, "DISMISSED")}
                        >
                          Dismiss
                        </button>
                      </div>
                    )}
                  </div>
                </article>
              ))}
            </div>
          )}
        </section>

        <footer className="footer">
          <span>SMARTFARM GUARDIAN</span>
          <span>AI-assisted monitoring · Owner-controlled review</span>
        </footer>
      </main>
    </div>
  );
}

export default App;
