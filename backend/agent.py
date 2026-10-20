
from datetime import datetime
from database import initialize_database, save_event
# Demo farm configuration
WORK_START_HOUR = 6
WORK_END_HOUR = 19

RESTRICTED_ZONES = {
    "chemical_storage",
    "equipment_shed",
    "main_gate_restricted"
}


def evaluate_event(
    zone="field",
    hour=None,
    worker_status="unknown",
    repeated_detections=1
):
    """
    Evaluate a detected-person event.
    This is a rule-based prototype, not a trained AI classifier.
    """

    if hour is None:
        hour = datetime.now().hour

    reasons = []
    severity = "LOW"

    # Rule 1: Activity outside working hours
    outside_working_hours = not (
        WORK_START_HOUR <= hour < WORK_END_HOUR
    )

    if outside_working_hours:
        reasons.append("Activity outside permitted working hours")
        severity = "MEDIUM"

    # Rule 2: Restricted zone
    if zone in RESTRICTED_ZONES:
        reasons.append(f"Person detected in restricted zone: {zone}")
        severity = "HIGH"

    # Rule 3: Worker status
    if worker_status == "unknown":
        reasons.append("Worker identity has not been verified")

    # Rule 4: Repeated detections
    if repeated_detections >= 5:
        reasons.append("Repeated activity requires review")

        if severity == "LOW":
            severity = "MEDIUM"

    # Produce the agent's decision
    if severity == "HIGH":
        decision = "ALERT_OWNER"
        message = "High-priority event: owner review required"

    elif outside_working_hours or zone in RESTRICTED_ZONES:
        decision = "REVIEW_REQUIRED"
        message = "Potential security concern: review this event"

    elif repeated_detections >= 5:
        decision = "REVIEW_REQUIRED"
        message = "Repeated activity: review this event"

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
        "reasons": reasons
    }


if __name__ == "__main__":
    initialize_database()
    print("=== SMARTFARM GUARDIAN AGENT ===\n")
    test_events = [
        {
            "zone": "field",
            "hour": 10,
            "worker_status": "approved",
            "repeated_detections": 1
        },
        {
            "zone": "equipment_shed",
            "hour": 22,
            "worker_status": "unknown",
            "repeated_detections": 2
        },
        {
            "zone": "field",
            "hour": 21,
            "worker_status": "unknown",
            "repeated_detections": 1
        },
        {
            "zone": "field",
            "hour": 14,
            "worker_status": "unknown",
            "repeated_detections": 6
        }
    ]

    for index, event in enumerate(test_events, start=1):
        result = evaluate_event(**event)
        event_id = save_event(result)

        print("Saved to database with event ID:", event_id)

        print(f"Event {index}")
        print("Decision:", result["decision"])
        print("Severity:", result["severity"])
        print("Message:", result["message"])

        if result["reasons"]:
            print("Reasons:")
            for reason in result["reasons"]:
                print(" -", reason)
        else:
            print("Reasons: No security rule triggered")

        print("-" * 45)
