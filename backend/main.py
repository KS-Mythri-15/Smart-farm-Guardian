
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from agent import evaluate_event
from database import initialize_database, save_event, get_connection


app = FastAPI(
    title="SmartFarm Guardian API",
    description="AI-assisted farm security and owner-controlled access review",
    version="1.0.0"
)

# Allow the local development frontend to communicate with this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

initialize_database()


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
    event = evaluate_event(**payload.model_dump())
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
