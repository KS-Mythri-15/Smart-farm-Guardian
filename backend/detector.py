
from pathlib import Path
import argparse
import time

import cv2
import numpy as np
import requests
from ultralytics import YOLO


# Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_DIR / "models"
OWNER_PROFILE = (
    PROJECT_DIR / "backend" / "known_faces" / "owner_profile.npz"
)

# Update this to your actual test video path.
DEFAULT_VIDEO = PROJECT_DIR / "videos" / "farm.mp4"

API_URL = "http://127.0.0.1:8000/events"

# Provisional threshold only. Calibrate using genuine and non-owner photos.
OWNER_SIMILARITY_THRESHOLD = 0.40

# Process one frame every N frames to reduce CPU usage.
FRAME_SKIP = 5

# Avoid sending a new event for every processed frame.
EVENT_COOLDOWN_SECONDS = 15


def load_owner_models():
    """Load face detector, face recognizer, and enrolled owner profile."""
    detector_path = MODEL_DIR / "face_detection_yunet_2023mar.onnx"
    recognizer_path = MODEL_DIR / "face_recognition_sface_2021dec.onnx"

    for path in (detector_path, recognizer_path, OWNER_PROFILE):
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    face_detector = cv2.FaceDetectorYN.create(
        str(detector_path),
        "",
        (320, 320),
        0.8,
        0.3,
        5000,
    )

    face_recognizer = cv2.FaceRecognizerSF.create(
        str(recognizer_path), ""
    )

    profile = np.load(OWNER_PROFILE)
    owner_embeddings = profile["embeddings"].astype(np.float32)

    if owner_embeddings.ndim == 1:
        owner_embeddings = owner_embeddings.reshape(1, -1)

    owner_embeddings /= (
        np.linalg.norm(owner_embeddings, axis=1, keepdims=True) + 1e-8
    )

    return face_detector, face_recognizer, owner_embeddings


def identify_potential_owner(frame, face_detector, face_recognizer,
                             owner_embeddings):
    """
    Return a potential match, not an authenticated identity.
    Unknown or unclear faces remain unverified.
    """
    height, width = frame.shape[:2]
    face_detector.setInputSize((width, height))

    _, faces = face_detector.detect(frame)

    if faces is None or len(faces) == 0:
        return "NO_FACE", None

    # Use the largest detected face for this prototype.
    face = max(faces, key=lambda item: item[2] * item[3])

    aligned = face_recognizer.alignCrop(frame, face)
    embedding = face_recognizer.feature(aligned).reshape(-1).astype(
        np.float32
    )
    embedding /= np.linalg.norm(embedding) + 1e-8

    scores = owner_embeddings @ embedding
    best_score = float(np.max(scores))

    if best_score >= OWNER_SIMILARITY_THRESHOLD:
        return "POTENTIAL_OWNER_MATCH", best_score

    return "UNKNOWN", best_score


def send_event(zone, worker_status, repeated_detections=1):
    """Record a security event using the existing FastAPI service."""
    payload = {
        "zone": zone,
        "worker_status": worker_status,
        "repeated_detections": repeated_detections,
    }

    try:
        response = requests.post(API_URL, json=payload, timeout=3)
        response.raise_for_status()
        result = response.json()
        print(
            f"Event #{result.get('id', '?')}: "
            f"{result.get('severity', 'UNKNOWN')} | "
            f"{result.get('decision', 'RECORDED')}"
        )
    except requests.RequestException as exc:
        print(f"Could not send event to API: {exc}")


def process_video(video_path):
    if not video_path.exists():
        raise FileNotFoundError(
            f"Video not found: {video_path}\n"
            "Change DEFAULT_VIDEO to the correct video location."
        )

    person_model = YOLO("yolo11n.pt")
    face_detector, face_recognizer, owner_embeddings = load_owner_models()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    output_dir = PROJECT_DIR / "runs" / "smartfarm"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "farm_security_annotated.mp4"

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 25.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    frame_number = 0
    last_event_time = 0.0

    print("Processing video:", video_path)
    print("Press Q in the preview window to stop.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame_number += 1

            if frame_number % FRAME_SKIP != 0:
                writer.write(frame)
                continue

            results = person_model(
                frame,
                classes=[0],
                conf=0.35,
                verbose=False,
            )

            person_count = len(results[0].boxes)
            labels = []

            for box in results[0].boxes:
                x1, y1, x2, y2 = map(
                    int, box.xyxy[0].cpu().tolist()
                )

                # Face recognition needs a usable face in the frame.
                status, score = identify_potential_owner(
                    frame,
                    face_detector,
                    face_recognizer,
                    owner_embeddings,
                )

                if status == "POTENTIAL_OWNER_MATCH":
                    label = f"Potential owner ({score:.2f})"
                    color = (0, 200, 0)
                elif status == "UNKNOWN":
                    label = "Unverified person"
                    color = (0, 165, 255)
                else:
                    label = "Person - face not visible"
                    color = (0, 165, 255)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame,
                    label,
                    (x1, max(25, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2,
                )
                labels.append(status)

            cv2.putText(
                frame,
                f"People detected: {person_count}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

            # Prototype policy: unknown or unverified people may require review.
            # This does NOT prove intrusion or grant access.
            now = time.time()
            if person_count > 0 and now - last_event_time >= EVENT_COOLDOWN_SECONDS:
                if any(item == "UNKNOWN" for item in labels):
                    send_event(
                        zone="field",
                        worker_status="unknown",
                        repeated_detections=1,
                    )
                    last_event_time = now

            writer.write(frame)
            cv2.imshow("SmartFarm Guardian - Video Test", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        writer.release()
        cv2.destroyAllWindows()

    print("Annotated video saved to:", output_path)



def process_photo(photo_path):
    """Detect people and check for a potential owner match in one photo."""
    if not photo_path.exists():
        raise FileNotFoundError(f"Photo not found: {photo_path}")

    frame = cv2.imread(str(photo_path))
    if frame is None:
        raise ValueError(f"Could not read photo: {photo_path}")

    person_model = YOLO("yolo11n.pt")
    face_detector, face_recognizer, owner_embeddings = load_owner_models()

    results = person_model(frame, classes=[0], conf=0.35, verbose=False)
    person_count = len(results[0].boxes)

    print(f"People detected: {person_count}")

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().tolist())

        # This first version checks the largest face in the full image.
        status, score = identify_potential_owner(
            frame,
            face_detector,
            face_recognizer,
            owner_embeddings,
        )

        if status == "POTENTIAL_OWNER_MATCH":
            label = f"Potential owner ({score:.2f})"
            color = (0, 200, 0)
        elif status == "UNKNOWN":
            label = f"Unverified person ({score:.2f})"
            color = (0, 165, 255)
        else:
            label = "Face not visible"
            color = (0, 165, 255)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            label,
            (x1, max(25, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
        )

        print(f"Detection result: {label}")

    output_dir = PROJECT_DIR / "runs" / "smartfarm"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "owner_photo_test.jpeg"

    cv2.imwrite(str(output_path), frame)
    print(f"Annotated photo saved to: {output_path}")

    cv2.imshow("SmartFarm Guardian - Photo Test", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SmartFarm Guardian photo and video detector"
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--photo", type=Path, help="Path to a test photo")
    mode.add_argument("--video", type=Path, help="Path to a test video")

    args = parser.parse_args()

    if args.photo:
        process_photo(args.photo)
    else:
        process_video(args.video)

