from typing import Literal, Optional

from fastapi import FastAPI, HTTPException, Query, UploadFile, File, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.agent import evaluate_event, DEFAULT_FARM_RULES
from backend.database import initialize_database, save_event, get_connection
import base64
import json

import cv2
import numpy as np

from fastapi import File, UploadFile
from pathlib import Path
from datetime import datetime
import tempfile
import shutil
import os

from fastapi.staticfiles import StaticFiles

from backend.detector import (
    load_person_model,
    load_owner_models,
    detect_people,
    evaluate_frame,
    draw_person_result,
    draw_unmatched_face,
)

app = FastAPI(
    title="SmartFarm Guardian API",
    description="AI-assisted farm security and owner-controlled access review",
    version="1.0.0"
)

# Allow local development and the deployed frontend.
frontend_origins = [
    origin.strip()
    for origin in os.getenv("FRONTEND_ORIGINS", "").split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        *frontend_origins,
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

initialize_database()

# Farm rules are stored separately so existing security events and reviews remain intact.
with get_connection() as connection:
    connection.execute("""
        CREATE TABLE IF NOT EXISTS farm_rules (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            settings_json TEXT NOT NULL
        )
    """)
    connection.execute(
        """
        INSERT OR IGNORE INTO farm_rules (id, settings_json)
        VALUES (1, ?)
        """,
        (json.dumps(DEFAULT_FARM_RULES),),
    )


def load_farm_rules():
    """Load the currently saved farm rules."""
    with get_connection() as connection:
        row = connection.execute(
            "SELECT settings_json FROM farm_rules WHERE id = 1"
        ).fetchone()

    if row is None:
        return DEFAULT_FARM_RULES.copy()

    try:
        return json.loads(row["settings_json"])
    except (TypeError, json.JSONDecodeError):
        return DEFAULT_FARM_RULES.copy()


@app.get("/farm-rules")
def get_farm_rules():
    """Return saved farm rules."""
    return load_farm_rules()


@app.put("/farm-rules")
def update_farm_rules(payload: dict = Body(...)):
    """Validate and persist the owner's farm rules."""
    required_fields = {
        "work_start_hour",
        "work_end_hour",
        "restricted_zones",
        "monitored_zones",
        "approved_workers",
        "alert_threshold",
    }
    missing = required_fields - payload.keys()
    if missing:
        raise HTTPException(
            status_code=422,
            detail="Missing fields: " + ", ".join(sorted(missing)),
        )

    start = payload["work_start_hour"]
    end = payload["work_end_hour"]
    threshold = payload["alert_threshold"]

    if (
        not isinstance(start, int) or isinstance(start, bool)
        or not 0 <= start <= 23
        or not isinstance(end, int) or isinstance(end, bool)
        or not 0 <= end <= 23
        or start == end
    ):
        raise HTTPException(
            status_code=422,
            detail="Working hours must be different integers from 0 to 23.",
        )

    if (
        not isinstance(threshold, int)
        or isinstance(threshold, bool)
        or threshold < 1
    ):
        raise HTTPException(
            status_code=422,
            detail="Alert threshold must be a positive integer.",
        )

    settings = {
        "work_start_hour": start,
        "work_end_hour": end,
        "alert_threshold": threshold,
    }

    for field in ("restricted_zones", "monitored_zones", "approved_workers"):
        value = payload[field]
        if not isinstance(value, list) or not all(
            isinstance(item, str) and item.strip() for item in value
        ):
            raise HTTPException(
                status_code=422,
                detail=f"{field} must be a list of non-empty strings.",
            )

        if field == "approved_workers":
            settings[field] = [item.strip() for item in value]
        else:
            settings[field] = [item.strip().lower() for item in value]

    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO farm_rules (id, settings_json)
            VALUES (1, ?)
            ON CONFLICT(id)
            DO UPDATE SET settings_json = excluded.settings_json
            """,
            (json.dumps(settings),),
        )

    return settings


class EventCreate(BaseModel):
    zone: str = Field(default="field", min_length=1, max_length=100)
    hour: Optional[int] = Field(default=None, ge=0, le=23)
    worker_status: Literal["approved", "unknown"] = "unknown"
    repeated_detections: int = Field(default=1, ge=1)


class ReviewUpdate(BaseModel):
    status: Literal["APPROVED", "DISMISSED"]


def get_event_by_id(event_id: int):
    with get_connection() as connection:
        row = connection.execute(
            "SELECT * FROM security_events WHERE id = ?",
            (event_id,)
        ).fetchone()

    return dict(row) if row else None


@app.get("/")
def home():
    return {
        "project": "SmartFarm Guardian",
        "status": "running",
        "docs": "/docs"
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/events")
def list_events(
    limit: int = Query(default=100, ge=1, le=500)
):
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT * FROM security_events
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,)
        ).fetchall()

    return [dict(row) for row in rows]


@app.get("/events/{event_id}")
def get_event(event_id: int):
    event = get_event_by_id(event_id)

    if event is None:
        raise HTTPException(
            status_code=404,
            detail="Security event not found"
        )

    return event


@app.post("/events", status_code=201)
def create_event(payload: EventCreate):
    event = evaluate_event(**payload.model_dump(), rules=load_farm_rules())
    event_id = save_event(event)

    return get_event_by_id(event_id)


@app.patch("/events/{event_id}/review")
def review_event(event_id: int, payload: ReviewUpdate):
    existing_event = get_event_by_id(event_id)

    if existing_event is None:
        raise HTTPException(
            status_code=404,
            detail="Security event not found"
        )

    with get_connection() as connection:
        connection.execute(
            """
            UPDATE security_events
            SET owner_status = ?
            WHERE id = ?
            """,
            (payload.status, event_id)
        )

    return get_event_by_id(event_id)

# ---------- YOLO + OWNER RECOGNITION API ----------

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "runs" / "smartfarm" / "api_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

app.mount(
    "/outputs",
    StaticFiles(directory=str(OUTPUT_DIR)),
    name="outputs"
)


def analyze_frame(frame):
    """Run YOLO person detection and enrolled-owner recognition."""
    person_model = load_person_model()
    face_detector, face_recognizer, owner_embeddings = load_owner_models()

    detections = detect_people(frame, person_model)

    people, unmatched_faces = evaluate_frame(
        frame,
        detections,
        face_detector,
        face_recognizer,
        owner_embeddings,
    )

    for person in people:
        draw_person_result(frame, person)

    for face in unmatched_faces:
        draw_unmatched_face(
            frame,
            face,
            face_detector,
            face_recognizer,
            owner_embeddings,
        )

    return people, unmatched_faces


def format_people(people):
    """Return JSON-safe detection results for React."""
    output = []

    for person in people:
        score = person.get("score")
        output.append({
            "status": person.get("status", "UNKNOWN"),
            "similarity": float(score) if score is not None else None,
            "confidence": float(person.get("confidence", 0)),
            "box": [int(v) for v in person.get("box", [])],
        })

    return output


def record_unknown_activity(people):
    """Send suspicious detections through the existing AI agent."""
    unknown_count = sum(
        p.get("status") in ("UNKNOWN", "NO_FACE")
        for p in people
    )

    if unknown_count:
        event = evaluate_event(
            zone="field",
            hour=datetime.now().hour,
            worker_status="unknown",
            repeated_detections=unknown_count,
            rules=load_farm_rules(),
        )
        event_id = save_event(event)
        return get_event_by_id(event_id)

    return None


@app.post("/analyze/photo")
async def analyze_photo(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload an image file.")

    contents = await file.read()

    if not contents:
        raise HTTPException(status_code=400, detail="The uploaded image is empty.")

    frame = cv2.imdecode(
        np.frombuffer(contents, dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )

    if frame is None:
        raise HTTPException(status_code=400, detail="Could not read this image.")

    try:
        people, unmatched_faces = analyze_frame(frame)

        # Save an annotated copy for the frontend.
        ok, encoded = cv2.imencode(".jpg", frame)
        if not ok:
            raise HTTPException(
                status_code=500,
                detail="Could not encode the annotated image.",
            )

        record_unknown_activity(people)

        return {
            "people_detected": len(people),
            "owner_matches": sum(
                p["status"] == "POTENTIAL_OWNER_MATCH" for p in people
            ),
            "unverified_people": sum(
                p["status"] in ("UNKNOWN", "NO_FACE") for p in people
            ) + len(unmatched_faces),
            "people": format_people(people),
            "annotated_image": base64.b64encode(
                encoded.tobytes()
            ).decode("utf-8"),
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Photo analysis failed: {exc}",
        ) from exc


@app.post("/analyze/video")
async def analyze_video(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="Upload a video file.")

    suffix = Path(file.filename or "farm_video.mp4").suffix or ".mp4"
    temp_path = None
    cap = None
    writer = None

    try:
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=suffix
        ) as temp_file:
            temp_path = Path(temp_file.name)
            shutil.copyfileobj(file.file, temp_file)

        cap = cv2.VideoCapture(str(temp_path))

        if not cap.isOpened():
            raise HTTPException(
                status_code=400,
                detail="Could not open the uploaded video.",
            )

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if width <= 0 or height <= 0:
            raise HTTPException(
                status_code=400,
                detail="The video has invalid dimensions.",
            )

        output_name = f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.mp4"
        output_path = OUTPUT_DIR / output_name

        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )

        if not writer.isOpened():
            raise HTTPException(
                status_code=500,
                detail="Could not create the annotated video.",
            )

        frame_number = 0
        all_people = []
        unknown_event_recorded = False

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame_number += 1

            # Run expensive recognition on every fifth frame.
            if frame_number % 5 == 0:
                people, unmatched_faces = analyze_frame(frame)
                all_people = people

                if not unknown_event_recorded and any(
                    p["status"] in ("UNKNOWN", "NO_FACE")
                    for p in people
                ):
                    record_unknown_activity(people)
                    unknown_event_recorded = True

                cv2.putText(
                    frame,
                    f"People detected: {len(people)}",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                )

            writer.write(frame)

        writer.release()
        writer = None
        cap.release()
        cap = None

        return {
            "people_detected": len(all_people),
            "owner_matches": sum(
                p["status"] == "POTENTIAL_OWNER_MATCH"
                for p in all_people
            ),
            "unverified_people": sum(
                p["status"] in ("UNKNOWN", "NO_FACE")
                for p in all_people
            ),
            "people": format_people(all_people),
            "annotated_video_url": f"/outputs/{output_name}",
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Video analysis failed: {exc}",
        ) from exc
    finally:
        if writer is not None:
            writer.release()
        if cap is not None:
            cap.release()
        if temp_path and temp_path.exists():
            temp_path.unlink(missing_ok=True)