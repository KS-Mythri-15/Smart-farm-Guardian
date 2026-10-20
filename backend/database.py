
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "smartfarm.db"


def get_connection():
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database():
    with get_connection() as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS security_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                zone TEXT NOT NULL,
                event_hour INTEGER NOT NULL,
                worker_status TEXT NOT NULL,
                repeated_detections INTEGER NOT NULL,
                severity TEXT NOT NULL,
                decision TEXT NOT NULL,
                message TEXT NOT NULL,
                reasons TEXT NOT NULL,
                owner_status TEXT NOT NULL DEFAULT 'PENDING',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)


def save_event(event):
    reasons = " | ".join(event["reasons"])

    with get_connection() as connection:
        cursor = connection.execute("""
            INSERT INTO security_events (
                zone, event_hour, worker_status,
                repeated_detections, severity, decision,
                message, reasons
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            event["zone"],
            event["hour"],
            event["worker_status"],
            event["repeated_detections"],
            event["severity"],
            event["decision"],
            event["message"],
            reasons
        ))

        return cursor.lastrowid


if __name__ == "__main__":
    initialize_database()
    print("SmartFarm Guardian database initialized.")
    print(f"Database location: {DB_PATH}")
