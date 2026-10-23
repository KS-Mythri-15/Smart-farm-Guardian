
import { useCallback, useEffect, useState } from "react";
import "./App.css";

const API = (import.meta.env.VITE_API_URL || "https://smart-farm-guardian.onrender.com").replace(/\/+$/, "");

const NAV = [
  { id: "overview", icon: "◫", label: "Overview", group: "WORKSPACE" },
  { id: "monitoring", icon: "◎", label: "Monitoring", group: "WORKSPACE" },
  { id: "events", icon: "≋", label: "Security events", group: "SECURITY" },
  { id: "review", icon: "✓", label: "Owner review", group: "SECURITY" },
  { id: "rules", icon: "⚙", label: "Farm rules", group: "CONFIGURATION" },
  { id: "evidence", icon: "▤", label: "Demo evidence", group: "CONFIGURATION" },
];

const DEFAULT_FARM_RULES = {
  work_start_hour: 6,
  work_end_hour: 19,
  restricted_zones: ["chemical_storage", "equipment_shed", "main_gate_restricted"],
  monitored_zones: ["field", "main_gate"],
  approved_workers: [],
  alert_threshold: 5,
};

const TESTS = [
  "Participant-defined demonstration",
  "Real-time person detection",
  "Worker recognition",
  "Intelligent activity monitoring",
  "Unauthorized activity alerts",
  "Owner-controlled security",
  "Security dashboard",
  "Incident tracking",
  "Input, agent steps and verifiable outcome",
];

function formatDate(value) {
  if (!value) return "—";
  const parsed = new Date(value.replace(" ", "T") + "Z");
  return Number.isNaN(parsed.getTime()) ? value : parsed.toLocaleString();
}

function severityClass(value) {
  return (value || "LOW").toLowerCase();
}

function StatCard({ label, value, note, icon, tone = "" }) {
  return (
    <article className="stat-card">
      <div className="stat-heading">
        <span>{label}</span>
        <span className={`stat-icon ${tone}`}>{icon}</span>
      </div>
      <div className="stat-value">{value}</div>
      <div className="stat-note">{note}</div>
    </article>
  );
}

function SectionHeading({ eyebrow, title, description, action }) {
  return (
    <div className="section-heading">
      <div>
        <div className="eyebrow">{eyebrow}</div>
        <h3>{title}</h3>
        {description && <p>{description}</p>}
      </div>
      {action}
    </div>
  );
}

function EventRow({ event, updatingId, onReview, compact = false }) {
  const severity = severityClass(event.severity);
  const status = (event.owner_status || "PENDING").toLowerCase();

  return (
    <article className={`incident-row ${compact ? "compact" : ""}`}>
      <div className={`incident-priority ${severity}`} />
      <div className="incident-content">
        <div className="incident-title-line">
          <h4>{(event.zone || "Farm").replaceAll("_", " ")}</h4>
          <span className={`severity-badge ${severity}`}>
            {event.severity || "UNKNOWN"}
          </span>
        </div>
        <p className="incident-message">
          {event.message || "Security event recorded."}
        </p>
        <p className="incident-reasons">
          {event.reasons || "No additional reasons recorded."}
        </p>
        <div className="incident-metadata">
          <span>#{event.id}</span>
          <span>{formatDate(event.created_at)}</span>
          <span>Worker: {event.worker_status || "unknown"}</span>
        </div>
      </div>
      <div className="incident-controls">
        <span className={`review-badge ${status}`}>
          {event.owner_status || "PENDING"}
        </span>
        {event.owner_status === "PENDING" && (
          <div className="review-actions">
            <button
              className="button button-approve"
              disabled={updatingId === event.id}
              onClick={() => onReview(event.id, "APPROVED")}
            >
              Approve
            </button>
            <button
              className="button button-subtle"
              disabled={updatingId === event.id}
              onClick={() => onReview(event.id, "DISMISSED")}
            >
              Dismiss
            </button>
          </div>
        )}
      </div>
    </article>
  );
}

function App() {
  const [page, setPage] = useState("overview");
  const [events, setEvents] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [apiOnline, setApiOnline] = useState(false);
  const [updatingId, setUpdatingId] = useState(null);
  const [filter, setFilter] = useState("ALL");

  const [farmRules, setFarmRules] = useState(DEFAULT_FARM_RULES);
  const [restrictedZonesText, setRestrictedZonesText] = useState(DEFAULT_FARM_RULES.restricted_zones.join(", "));
  const [monitoredZonesText, setMonitoredZonesText] = useState(DEFAULT_FARM_RULES.monitored_zones.join(", "));
  const [approvedWorkersText, setApprovedWorkersText] = useState("");
  const [rulesLoading, setRulesLoading] = useState(false);
  const [rulesSaving, setRulesSaving] = useState(false);
  const [rulesError, setRulesError] = useState("");
  const [rulesNotice, setRulesNotice] = useState("");

  const [media, setMedia] = useState(null);
  const [mediaType, setMediaType] = useState("photo");
  const [previewUrl, setPreviewUrl] = useState("");
  const [analysis, setAnalysis] = useState(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [analysisError, setAnalysisError] = useState("");

  const loadEvents = useCallback(async () => {
    try {
      const response = await fetch(`${API}/events?limit=100`);
      if (!response.ok) throw new Error(`API returned ${response.status}`);
      const data = await response.json();
      setEvents(data);
      setApiOnline(true);
      setError("");
    } catch {
      setApiOnline(false);
     setError(`Can't reach FastAPI at ${API}. Check the backend service.`);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadEvents();
    const timer = setInterval(loadEvents, 10000);
    return () => clearInterval(timer);
  }, [loadEvents]);

  const loadFarmRules = useCallback(async () => {
    setRulesLoading(true);
    setRulesError("");
    try {
      const response = await fetch(`${API}/farm-rules`);
      const data = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(data.detail || `Could not load farm rules (${response.status}).`);
      }
      const loadedRules = {
        ...DEFAULT_FARM_RULES,
        ...data,
        restricted_zones: Array.isArray(data.restricted_zones) ? data.restricted_zones : DEFAULT_FARM_RULES.restricted_zones,
        monitored_zones: Array.isArray(data.monitored_zones) ? data.monitored_zones : DEFAULT_FARM_RULES.monitored_zones,
        approved_workers: Array.isArray(data.approved_workers) ? data.approved_workers : DEFAULT_FARM_RULES.approved_workers,
      };
      setFarmRules(loadedRules);
      setRestrictedZonesText(loadedRules.restricted_zones.join(", "));
      setMonitoredZonesText(loadedRules.monitored_zones.join(", "));
      setApprovedWorkersText(loadedRules.approved_workers.join(", "));
    } catch (err) {
      setRulesError(err.message || "Could not load saved farm rules.");
    } finally {
      setRulesLoading(false);
    }
  }, []);

  useEffect(() => {
    loadFarmRules();
  }, [loadFarmRules]);

  function updateFarmRule(field, value) {
    setFarmRules((current) => ({ ...current, [field]: value }));
    setRulesNotice("");
    setRulesError("");
  }

  function parseCommaList(value) {
    return value.split(",").map((item) => item.trim()).filter(Boolean);
  }

  async function saveFarmRules() {
    setRulesSaving(true);
    setRulesError("");
    setRulesNotice("");
    try {
      const response = await fetch(`${API}/farm-rules`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          ...farmRules,
          work_start_hour: Number(farmRules.work_start_hour),
          work_end_hour: Number(farmRules.work_end_hour),
          restricted_zones: parseCommaList(restrictedZonesText),
          monitored_zones: parseCommaList(monitoredZonesText),
          approved_workers: parseCommaList(approvedWorkersText),
          alert_threshold: Number(farmRules.alert_threshold),
        }),
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(
          typeof data.detail === "string"
            ? data.detail
            : `Could not save farm rules (${response.status}).`
        );
      }
      setFarmRules({ ...DEFAULT_FARM_RULES, ...data });
      setRestrictedZonesText((data.restricted_zones || []).join(", "));
      setMonitoredZonesText((data.monitored_zones || []).join(", "));
      setApprovedWorkersText((data.approved_workers || []).join(", "));
      setRulesNotice("Farm rules saved successfully and will be used for event evaluation.");
    } catch (err) {
      setRulesError(err.message || "Could not save farm rules.");
    } finally {
      setRulesSaving(false);
    }
  }

  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  async function reviewEvent(id, status) {
    setUpdatingId(id);
    setError("");
    try {
      const response = await fetch(`${API}/events/${id}/review`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ status }),
      });
      if (!response.ok) throw new Error(`Review failed: ${response.status}`);
      const updated = await response.json();
      setEvents((current) =>
        current.map((item) => (item.id === id ? updated : item))
      );
    } catch (err) {
      setError(err.message || "Could not update the review.");
    } finally {
      setUpdatingId(null);
    }
  }

  function chooseMedia(e) {
    const file = e.target.files?.[0];
    if (!file) return;

    const photo = file.type.startsWith("image/");
    const video = file.type.startsWith("video/");
    if (!photo && !video) {
      setAnalysisError("Choose an image or video file.");
      return;
    }
    if (file.size > 100 * 1024 * 1024) {
      setAnalysisError("Choose a file smaller than 100 MB.");
      return;
    }

    setMedia(file);
    setMediaType(photo ? "photo" : "video");
    setPreviewUrl(URL.createObjectURL(file));
    setAnalysis(null);
    setAnalysisError("");
  }

  async function analyzeMedia() {
    if (!media) return setAnalysisError("Choose a photo or video first.");

    setAnalyzing(true);
    setAnalysis(null);
    setAnalysisError("");

    try {
      const form = new FormData();
      form.append("file", media);

      const endpoint =
        mediaType === "photo" ? "/analyze/photo" : "/analyze/video";
      const response = await fetch(`${API}${endpoint}`, {
        method: "POST",
        body: form,
      });
      const result = await response.json().catch(() => ({}));

      if (!response.ok) {
        throw new Error(result.detail || `Analysis failed (${response.status}).`);
      }
      setAnalysis(result);
    } catch (err) {
      setAnalysisError(err.message || "Analysis failed. Check the backend route.");
    } finally {
      setAnalyzing(false);
    }
  }

  function clearMedia() {
    setMedia(null);
    setPreviewUrl("");
    setAnalysis(null);
    setAnalysisError("");
  }

  const pending = events.filter((e) => e.owner_status === "PENDING");
  const high = events.filter((e) => e.severity === "HIGH");
  const reviewed = events.filter((e) => e.owner_status !== "PENDING");
  const filtered =
    filter === "ALL" ? events :
    filter === "PENDING" ? pending :
    events.filter((e) => e.severity === filter);

  const titles = {
    overview: ["Overview", "Your farm security at a glance."],
    monitoring: ["Monitoring", "Inspect farm footage and review AI results."],
    events: ["Security events", "A searchable view of recorded incidents."],
    review: ["Owner review", "Your decisions. Your farm. Your control."],
    rules: ["Farm rules", "Context that helps the agent evaluate activity."],
    evidence: ["Demo evidence", "Track the required hackathon demonstration runs."],
  };

  const [title, subtitle] = titles[page];

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">🌾</div>
          <div className="brand-copy">
            <h1>SmartFarm</h1>
            <span>GUARDIAN</span>
          </div>
        </div>

        <div className="farm-switcher">
          <div className="farm-avatar">SF</div>
          <div><strong>My Farm</strong><span>Security workspace</span></div>
          <span className="switch-chevron">⌄</span>
        </div>

        {["WORKSPACE", "SECURITY", "CONFIGURATION"].map((group) => (
          <div className="nav-group" key={group}>
            <div className="side-label">{group}</div>
            {NAV.filter((item) => item.group === group).map((item) => (
              <button
                key={item.id}
                className={`nav-item ${page === item.id ? "active" : ""}`}
                onClick={() => setPage(item.id)}
              >
                <span className="nav-icon">{item.icon}</span>
                <span>{item.label}</span>
                {item.id === "review" && pending.length > 0 && (
                  <span className="nav-count">{pending.length}</span>
                )}
              </button>
            ))}
          </div>
        ))}

        <div className="sidebar-bottom">
          <div className={`connection-card ${apiOnline ? "online" : "offline"}`}>
            <span className="connection-dot" />
            <div>
              <strong>{apiOnline ? "Backend connected" : "Backend offline"}</strong>
              <small>{apiOnline ? "Event API responding" : "Check local API"}</small>
            </div>
          </div>
          <div className="profile-row">
            <div className="profile-avatar">SG</div>
            <div><strong>Farm owner</strong><span>Owner access</span></div>
            <span className="profile-menu">···</span>
          </div>
          <p className="sidebar-version">SMARTFARM GUARDIAN · MVP</p>
        </div>
      </aside>

      <main className="main-content">
        <header className="topbar">
          <div>
            <div className="breadcrumb">SMARTFARM <span>/</span> {title.toUpperCase()}</div>
            <h2>{title}</h2>
            <p>{subtitle}</p>
          </div>
          <div className="topbar-actions">
            <span className={`system-pill ${apiOnline ? "good" : "bad"}`}>
              <span className="connection-dot" />
              {apiOnline ? "System connected" : "Connection issue"}
            </span>
            <button className="button button-refresh" onClick={loadEvents}>
              ↻ <span>Refresh</span>
            </button>
          </div>
        </header>

        {error && <div className="error-banner">⚠ {error}</div>}

        {page === "overview" && (
          <>
            <section className="hero-panel">
              <div className="hero-copy">
                <span className="hero-kicker"><span className="hero-kicker-dot" /> OWNER-CONTROLLED SECURITY</span>
                <h3>Keep an eye on<br /><em>what matters.</em></h3>
                <p>
                  Review farm activity, understand unusual events, and make
                  informed access decisions—even when you're away.
                </p>
                <button className="button button-primary" onClick={() => setPage("monitoring")}>
                  Open monitoring <span>↗</span>
                </button>
              </div>
              <div className="hero-art" aria-hidden="true">
                <div className="hero-orbit orbit-one" />
                <div className="hero-orbit orbit-two" />
                <div className="hero-plant">♧</div>
                <div className="hero-art-caption"><span className="connection-dot" /> FARM SECURITY ACTIVE</div>
              </div>
              <div className="hero-index">01 — CONTROL CENTER</div>
            </section>

            <div className="section-title-row">
              <div><div className="eyebrow">AT A GLANCE</div><h3>Security overview</h3></div>
              <span className="muted-note">From stored event records</span>
            </div>

            <section className="stats-grid">
              <StatCard label="Total incidents" value={events.length} note="Recorded by the system" icon="≋" />
              <StatCard label="High priority" value={high.length} note="Needs careful attention" icon="!" tone="icon-red" />
              <StatCard label="Awaiting review" value={pending.length} note="Needs an owner decision" icon="◷" tone="icon-amber" />
              <StatCard label="Reviewed" value={reviewed.length} note="Decision recorded" icon="✓" tone="icon-green" />
            </section>

            <section className="content-panel">
              <SectionHeading
                eyebrow="LATEST ACTIVITY"
                title="Recent incidents"
                description="The newest records from your security event history."
                action={<button className="text-link" onClick={() => setPage("events")}>View all events ↗</button>}
              />
              {loading ? <div className="empty-state">Loading incidents…</div> :
                events.length === 0 ? <div className="empty-state">No recorded incidents yet. Run a monitoring test to get started.</div> :
                <div className="incident-list">
                  {events.slice(0, 5).map((event) => (
                    <EventRow key={event.id} event={event} compact updatingId={updatingId} onReview={reviewEvent} />
                  ))}
                </div>}
            </section>
          </>
        )}

        {page === "monitoring" && (
          <>
            <section className="monitoring-banner">
              <div className="monitoring-symbol">◎</div>
              <div><div className="eyebrow">ANALYSIS WORKSPACE</div><h3>Inspect farm footage</h3>
                <p>Choose a photo or video. The configured backend processes the upload and returns detection results.</p></div>
              <span className="mode-tag">MANUAL INPUT</span>
            </section>

            <section className="content-panel upload-panel">
              <SectionHeading eyebrow="01 / INPUT SOURCE" title="Upload footage" description="Select a farm image or video for analysis." />
              <label className="upload-zone">
                <span className="upload-symbol">↑</span>
                <strong>{media ? media.name : "Drop your footage here"}</strong>
                <span>{media ? `${(media.size / 1048576).toFixed(2)} MB · ${mediaType}` : "or browse your device · Images and videos"}</span>
                <span className="browse-button">Browse files</span>
                <input type="file" accept="image/*,video/*" onChange={chooseMedia} />
              </label>

              {analysisError && <div className="error-banner inner-error">{analysisError}</div>}

              {previewUrl && (
                <div className="selected-media">
                  <div className="selected-media-header">
                    <div><div className="eyebrow">SELECTED INPUT</div><strong>{media?.name}</strong></div>
                    <button className="text-link" onClick={clearMedia}>Remove file ×</button>
                  </div>
                  {mediaType === "photo"
                    ? <img className="media-preview" src={previewUrl} alt="Selected farm input" />
                    : <video className="media-preview" src={previewUrl} controls />}
                  <button className="button button-primary analyze-button" disabled={analyzing} onClick={analyzeMedia}>
                    {analyzing ? "Analyzing footage…" : "Run AI analysis →"}
                  </button>
                </div>
              )}
            </section>

            {analysis && (
              <section className="content-panel results-panel">
                <SectionHeading eyebrow="02 / AI OUTPUT" title="Detection results" description="Output returned by the analysis endpoint." />
                <div className="result-grid">
                  <StatCard label="People detected" value={analysis.people_detected ?? analysis.people?.length ?? "—"} note="Detection output" icon="◎" />
                  <StatCard label="Potential owner matches" value={analysis.owner_matches ?? "—"} note="Not an access grant" icon="✓" tone="icon-green" />
                  <StatCard label="Unverified people" value={analysis.unverified_people ?? "—"} note="Needs context or review" icon="!" tone="icon-amber" />
                </div>
                {analysis.annotated_image && <div className="output-preview"><h4>Annotated image</h4><img className="media-preview" src={`data:image/jpeg;base64,${analysis.annotated_image}`} alt="Annotated AI result" /></div>}
                {analysis.annotated_video_url && <div className="output-preview"><h4>Processed video</h4><video controls className="media-preview" src={analysis.annotated_video_url.startsWith("http") ? analysis.annotated_video_url : `${API}${analysis.annotated_video_url}`} /></div>}
                {Array.isArray(analysis.people) && (
                  <div className="person-table">
                    <h4>Person-level results</h4>
                    {analysis.people.length === 0 ? <p className="empty-state">No people detected.</p> : analysis.people.map((person, i) => {
                      const match = ["OWNER_MATCH", "POTENTIAL_OWNER_MATCH"].includes(person.status);
                      return <div className="person-line" key={`${i}-${person.status}`}>
                        <span className={`person-indicator ${match ? "matched" : ""}`} />
                        <strong>Person {i + 1}</strong>
                        <span className={`severity-badge ${match ? "low" : "medium"}`}>{person.status || "UNVERIFIED"}</span>
                        <span className="person-score">{person.similarity == null ? "Score —" : `Similarity ${(Number(person.similarity) * 100).toFixed(1)}%`}</span>
                      </div>;
                    })}
                  </div>
                )}
                <p className="disclaimer">A recognition match is not the same as access authorization. Unknown identity alone does not prove trespassing.</p>
              </section>
            )}
          </>
        )}

        {(page === "events" || page === "review") && (
          <section className="content-panel">
            <SectionHeading
              eyebrow={page === "review" ? "OWNER DECISIONS" : "INCIDENT REGISTER"}
              title={page === "review" ? "Awaiting your review" : "Security event history"}
              description={page === "review" ? "Inspect the recorded context before deciding." : "All available incidents returned by the backend."}
              action={<span className="record-count">{page === "review" ? pending.length : events.length} records</span>}
            />
            {page === "events" && <div className="filter-bar">
              {["ALL", "PENDING", "HIGH", "MEDIUM", "LOW"].map((item) => <button key={item} className={`filter-chip ${filter === item ? "selected" : ""}`} onClick={() => setFilter(item)}>{item === "ALL" ? "All events" : item === "PENDING" ? "Pending review" : `${item} severity`}</button>)}
            </div>}
            {loading ? <div className="empty-state">Loading incidents…</div> :
              (page === "review" ? pending : filtered).length === 0
                ? <div className="empty-state">{page === "review" ? "You're all caught up. No incidents are awaiting review." : "No incidents match this filter."}</div>
                : <div className="incident-list">{(page === "review" ? pending : filtered).map((event) => <EventRow key={event.id} event={event} updatingId={updatingId} onReview={reviewEvent} />)}</div>}
          </section>
        )}

        {page === "rules" && (
          <section className="content-panel rules-panel">
            <SectionHeading
              eyebrow="CONTEXTUAL DECISION-MAKING"
              title="Farm security rules"
              description="Edit the rules used by the backend agent when evaluating manually created and unknown-person events."
              action={<button className="button button-refresh" onClick={loadFarmRules} disabled={rulesLoading || rulesSaving}>{rulesLoading ? "Loading…" : "↻ Reload saved rules"}</button>}
            />

            {rulesError && <div className="error-banner rules-message">{rulesError}</div>}
            {rulesNotice && <div className="rules-success rules-message" role="status">{rulesNotice}</div>}
            {rulesLoading ? (
              <div className="empty-state">Loading saved farm rules…</div>
            ) : (
              <div className="rules-form">
                <section className="rule-form-section">
                  <div className="rule-form-heading">
                    <span className="rule-icon">◷</span>
                    <div><strong>Working hours</strong><p>Activity outside this window can trigger review.</p></div>
                  </div>
                  <div className="form-grid two-columns">
                    <label className="field-label">Start hour (24-hour time)
                      <select value={farmRules.work_start_hour} onChange={(e) => updateFarmRule("work_start_hour", Number(e.target.value))}>
                        {Array.from({ length: 24 }, (_, hour) => <option key={hour} value={hour}>{String(hour).padStart(2, "0")}:00</option>)}
                      </select>
                    </label>
                    <label className="field-label">End hour (24-hour time)
                      <select value={farmRules.work_end_hour} onChange={(e) => updateFarmRule("work_end_hour", Number(e.target.value))}>
                        {Array.from({ length: 24 }, (_, hour) => <option key={hour} value={hour}>{String(hour).padStart(2, "0")}:00</option>)}
                      </select>
                    </label>
                  </div>
                  {Number(farmRules.work_start_hour) === Number(farmRules.work_end_hour) && <p className="field-help field-error">Start and end hours must be different.</p>}
                </section>

                <section className="rule-form-section">
                  <div className="rule-form-heading">
                    <span className="rule-icon">⌖</span>
                    <div><strong>Farm zones</strong><p>Enter zone names separated by commas. Use the same names that your events send to the backend.</p></div>
                  </div>
                  <label className="field-label">Restricted zones
                    <input value={restrictedZonesText} onChange={(e) => { setRestrictedZonesText(e.target.value); setRulesNotice(""); }} placeholder="chemical_storage, equipment_shed" />
                  </label>
                  <p className="field-help">Activity in these zones is treated as high priority.</p>
                  <label className="field-label">Monitored zones
                    <input value={monitoredZonesText} onChange={(e) => { setMonitoredZonesText(e.target.value); setRulesNotice(""); }} placeholder="field, main_gate" />
                  </label>
                  <p className="field-help">Events in these zones are logged for review.</p>
                </section>

                <section className="rule-form-section">
                  <div className="rule-form-heading">
                    <span className="rule-icon">⚑</span>
                    <div><strong>Alert threshold</strong><p>Repeated detections at or above this number trigger review.</p></div>
                  </div>
                  <label className="field-label">Repeated detections
                    <input type="number" min="1" step="1" value={farmRules.alert_threshold} onChange={(e) => updateFarmRule("alert_threshold", e.target.value === "" ? "" : Number(e.target.value))} />
                  </label>
                </section>

                <section className="rule-form-section">
                  <div className="rule-form-heading">
                    <span className="rule-icon">♙</span>
                    <div><strong>Approved worker labels</strong><p>Stored as configuration only; this list does not currently grant access or change face-recognition matching.</p></div>
                  </div>
                  <label className="field-label">Worker names or labels
                    <input value={approvedWorkersText} onChange={(e) => { setApprovedWorkersText(e.target.value); setRulesNotice(""); }} placeholder="Enter labels separated by commas" />
                  </label>
                  <p className="field-help">Do not rely on this list as an access-control mechanism until identity matching is explicitly integrated.</p>
                </section>

                <div className="rules-actions">
                  <button className="button button-primary" onClick={saveFarmRules} disabled={rulesSaving || rulesLoading || Number(farmRules.work_start_hour) === Number(farmRules.work_end_hour) || !Number.isInteger(Number(farmRules.alert_threshold)) || Number(farmRules.alert_threshold) < 1}>
                    {rulesSaving ? "Saving rules…" : "Save farm rules"}
                  </button>
                  <span className="field-help">Saved settings persist in SQLite and are read by the backend event evaluator.</span>
                </div>
              </div>
            )}
          </section>
        )}

        {page === "evidence" && (
          <section className="content-panel">
            <SectionHeading eyebrow="HACKATHON VALIDATION" title="Demo test register" description="Track each required demonstration honestly. Record the actual result and evidence after running it." />
            <div className="evidence-note"><strong>Evidence standard</strong><p>Do not mark a scenario as passed until it has been executed and its result has been captured.</p></div>
            <div className="test-list">{TESTS.map((test, i) => <div className="test-row" key={test}><span className="test-number">{String(i + 1).padStart(2, "0")}</span><div><strong>{test}</strong><span>Input, expected result, actual result and evidence must be recorded.</span></div><span className="status-tag">Not run</span></div>)}</div>
          </section>
        )}

        <footer className="app-footer"><span>🌾 SMARTFARM GUARDIAN</span><span>AI-assisted monitoring <i /> Owner-controlled decisions</span></footer>
      </main>
    </div>
  );
}

export default App;
