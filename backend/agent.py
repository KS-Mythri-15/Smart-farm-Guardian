
from datetime import datetime
from database import initialize_database, save_event

DEFAULT_FARM_RULES = {
    "work_start_hour": 6,
    "work_end_hour": 19,
    "restricted_zones": [
        "chemical_storage",
        "equipment_shed",
        "main_gate_restricted",
    ],
    "monitored_zones": ["field", "main_gate"],
    "approved_workers": [],
    "alert_threshold": 5,
}


def evaluate_event(
    zone="field",
    hour=None,
    worker_status="unknown",
    repeated_detections=1,
    rules=None,
):
    """Evaluate farm activity using configured security rules."""
    settings = {**DEFAULT_FARM_RULES, **(rules or {})}

    if hour is None:
        hour = datetime.now().hour

    zone = str(zone).strip().lower() or "field"
    worker_status = str(worker_status).strip().lower()
    repeated_detections = max(1, int(repeated_detections))

    start = int(settings["work_start_hour"])
    end = int(settings["work_end_hour"])
    threshold = max(1, int(settings["alert_threshold"]))

    restricted = {
        str(z).strip().lower()
        for z in settings["restricted_zones"]
    }
    monitored = {
        str(z).strip().lower()
        for z in settings["monitored_zones"]
    }

    # Supports normal and overnight working-hour windows.
    if start < end:
        outside_hours = not (start <= hour < end)
    else:
        outside_hours = not (hour >= start or hour < end)

    reasons = []
    severity = "LOW"

    if outside_hours:
        reasons.append("Activity outside permitted working hours")
        severity = "MEDIUM"

    if zone in restricted:
        reasons.append(f"Person detected in restricted zone: {zone}")
        severity = "HIGH"

    if worker_status == "unknown":
        reasons.append("Worker identity has not been verified")

    if repeated_detections >= threshold:
        reasons.append("Repeated activity requires review")
        if severity == "LOW":
            severity = "MEDIUM"

    if severity == "HIGH":
        decision = "ALERT_OWNER"
        message = "High-priority event: owner review required"
    elif outside_hours or zone in restricted:
        decision = "REVIEW_REQUIRED"
        message = "Potential security concern: review this event"
    elif repeated_detections >= threshold:
        decision = "REVIEW_REQUIRED"
        message = "Repeated activity: review this event"
    elif zone in monitored or worker_status == "unknown":
        decision = "REVIEW_REQUIRED"
        message = "Monitored activity: owner review recommended"
    else:
        decision = "LOG_EVENT"
        message = "No configured security rule was triggered"

    return {
        "zone": zone,
        "hour": hour,
        "worker_status": worker_status,
        "repeated_detections": repeated_detections,
        "severity": severity,
        "decision": decision,
        "message": message,
        "reasons": reasons,
    }


if __name__ == "__main__":
    initialize_database()

    test_events = [
        {
            "zone": "field",
            "hour": 10,
            "worker_status": "approved",
            "repeated_detections": 1,
        },
        {
            "zone": "equipment_shed",
            "hour": 22,
            "worker_status": "unknown",
            "repeated_detections": 2,
        },
        {
            "zone": "field",
            "hour": 21,
            "worker_status": "unknown",
            "repeated_detections": 1,
        },
        {
            "zone": "field",
            "hour": 14,
            "worker_status": "unknown",
            "repeated_detections": 6,
        },
    ]

    for index, event in enumerate(test_events, start=1):
        result = evaluate_event(**event)
        event_id = save_event(result)

        print(f"\nEvent {index} | Database ID: {event_id}")
        print("Decision:", result["decision"])
        print("Severity:", result["severity"])
        print("Message:", result["message"])
        print("Reasons:", result["reasons"])
