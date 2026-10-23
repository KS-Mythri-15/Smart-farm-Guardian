import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import requests
from ultralytics import YOLO
import os
import base64

# --------------------------------------------------
# PROJECT CONFIGURATION
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_DIR / "models"
KNOWN_FACES_DIR = PROJECT_DIR / "backend" / "known_faces"
OWNER_PROFILE = KNOWN_FACES_DIR / "owner_profile.npz"
def ensure_owner_profile():
    if OWNER_PROFILE.exists():
        return

    encoded = os.getenv("OWNER_PROFILE_B64")
    if encoded:
        KNOWN_FACES_DIR.mkdir(parents=True, exist_ok=True)
        OWNER_PROFILE.write_bytes(base64.b64decode(encoded))
# YOLO11s is more capable than YOLO11n for small/partly occluded people.
# Ultralytics downloads yolo11s.pt automatically if it is not available locally.
YOLO_MODEL_NAME = "yolo11s.pt"

PERSON_CONFIDENCE = 0.10
PERSON_IMAGE_SIZE = 1280
PERSON_IOU_THRESHOLD = 0.35

# Keep this calibrated using your enrolled-owner photos and non-owner photos.
# This is a similarity threshold, NOT a probability.
OWNER_SIMILARITY_THRESHOLD = 0.52

FRAME_SKIP = 3
EVENT_COOLDOWN_SECONDS = 10
API_URL = "http://127.0.0.1:8000/events"


# --------------------------------------------------
# LOAD MODELS AND OWNER PROFILE
# --------------------------------------------------

def load_owner_models():
    ensure_owner_profile()
    """Load YuNet, SFace, and the enrolled owner embeddings."""
    detector_path = MODEL_DIR / "face_detection_yunet_2023mar.onnx"
    recognizer_path = MODEL_DIR / "face_recognition_sface_2021dec.onnx"

    for path in (detector_path, recognizer_path, OWNER_PROFILE):
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    face_detector = cv2.FaceDetectorYN.create(
        str(detector_path), "", (320, 320), 0.75, 0.3, 5000
    )
    face_recognizer = cv2.FaceRecognizerSF.create(
        str(recognizer_path), ""
    )

    with np.load(OWNER_PROFILE) as profile:
        owner_embeddings = profile["embeddings"].astype(np.float32)

    if owner_embeddings.ndim == 1:
        owner_embeddings = owner_embeddings.reshape(1, -1)

    norms = np.linalg.norm(owner_embeddings, axis=1, keepdims=True)
    owner_embeddings = owner_embeddings / np.maximum(norms, 1e-8)

    return face_detector, face_recognizer, owner_embeddings


def load_person_model():
    """Load the stronger YOLO model; Ultralytics downloads it if needed."""
    return YOLO(YOLO_MODEL_NAME)


# --------------------------------------------------
# FACE DETECTION AND RECOGNITION
# --------------------------------------------------

def detect_faces(frame, face_detector):
    """Return YuNet face detections for this frame."""
    height, width = frame.shape[:2]
    face_detector.setInputSize((width, height))
    _, faces = face_detector.detect(frame)
    return [] if faces is None else list(faces)


def identify_face(frame, face, face_recognizer, owner_embeddings):
    """Compare one face with the enrolled owner profile."""
    try:
        aligned_face = face_recognizer.alignCrop(frame, face)
        embedding = face_recognizer.feature(aligned_face).reshape(-1).astype(
            np.float32
        )
        norm = float(np.linalg.norm(embedding))
        if norm < 1e-8:
            return "UNKNOWN", None

        embedding /= norm
        best_score = float(np.max(owner_embeddings @ embedding))

        if best_score >= OWNER_SIMILARITY_THRESHOLD:
            return "POTENTIAL_OWNER_MATCH", best_score
        return "UNKNOWN", best_score
    except cv2.error:
        return "UNKNOWN", None


# --------------------------------------------------
# PERSON DETECTION
# --------------------------------------------------

def detect_people(frame, person_model):
    """Detect people, returning clipped boxes and confidence scores."""
    results = person_model.predict(
        source=frame,
        classes=[0],
        conf=PERSON_CONFIDENCE,
        imgsz=PERSON_IMAGE_SIZE,
        iou=PERSON_IOU_THRESHOLD,
        augment=True,
        max_det=100,
        verbose=False,
    )

    height, width = frame.shape[:2]
    detections = []

    for detection in results[0].boxes:
        x1, y1, x2, y2 = map(int, detection.xyxy[0].cpu().tolist())
        confidence = float(detection.conf[0].cpu())

        x1 = max(0, min(x1, width - 1))
        x2 = max(0, min(x2, width))
        y1 = max(0, min(y1, height - 1))
        y2 = max(0, min(y2, height))

        if x2 > x1 and y2 > y1:
            detections.append(((x1, y1, x2, y2), confidence))

    # Keep the higher-confidence box first.
    detections.sort(key=lambda item: item[1], reverse=True)
    return detections


def face_center(face):
    return (
        float(face[0] + face[2] / 2),
        float(face[1] + face[3] / 2),
    )




def associate_faces_with_people(person_boxes, faces):
    """
    Associate detected faces with person boxes.
    Each face and each person can be assigned at most once.
    Returns face indices, as expected by evaluate_frame().
    """
    assignments = {i: [] for i in range(len(person_boxes))}
    candidates = []

    for face_index, face in enumerate(faces):
        fx, fy = face_center(face)

        for person_index, (x1, y1, x2, y2) in enumerate(person_boxes):
            bw = max(1, x2 - x1)
            bh = max(1, y2 - y1)

            # Allow the face to sit near the top of the person box.
            # Use a wider vertical range for imperfect body detections.
            if not (x1 <= fx <= x2):
                continue
            if not (y1 - 0.12 * bh <= fy <= y1 + 0.85 * bh):
                continue

            expected_x = (x1 + x2) / 2
            expected_y = y1 + 0.18 * bh

            dx = (fx - expected_x) / bw
            dy = (fy - expected_y) / bh

            # Prefer tighter person boxes when detections overlap.
            width_penalty = 0.15 * min(bw / bh, 3.0)
            cost = dx * dx + dy * dy + width_penalty

            candidates.append((cost, face_index, person_index))
            
    candidates.sort(key=lambda item: item[0])

    used_faces = set()
    used_people = set()

    for cost, face_index, person_index in candidates:
        if face_index in used_faces or person_index in used_people:
            continue

        assignments[person_index].append(face_index)
        used_faces.add(face_index)
        used_people.add(person_index)

    return assignments, used_faces





def evaluate_frame(frame, person_detections, face_detector,
                   face_recognizer, owner_embeddings):
    """
    Recognize each detected face and associate it with a person box.
    Keeps the existing person-detection results and return structure.
    """
    person_boxes = [box for box, _ in person_detections]
    faces = detect_faces(frame, face_detector)

    # Step 1: Recognize each face independently.
    face_results = []

    for face_index, face in enumerate(faces):
        status, score = identify_face(
            frame, face, face_recognizer, owner_embeddings
        )

        face_results.append({
            "face": face,
            "status": status,
            "score": score,
            "index": face_index,
        })

        print(
            f"Face {face_index + 1}: "
            f"status={status}, similarity={score}"
        )

    # Step 2: Find plausible face-to-person associations.
    # This changes association only, not person detection.
    candidates = []

    for person_index, box in enumerate(person_boxes):
        x1, y1, x2, y2 = box
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)

        for face_index, result in enumerate(face_results):
            face = result["face"]
            fx, fy = face_center(face)
            fw = float(face[2])
            fh = float(face[3])

            # Allow a face slightly outside the top/side of a
            # person box because detected body boxes can be imperfect.
            horizontal_margin = 0.20 * bw
            if fx < x1 - horizontal_margin or fx > x2 + horizontal_margin:
                continue

            if fy < y1 - 0.50 * bh or fy > y1 + 0.85 * bh:
                continue

            expected_x = (x1 + x2) / 2
            expected_y = y1 + 0.15 * bh

            dx = (fx - expected_x) / bw
            dy = (fy - expected_y) / bh

            # Prefer boxes that contain the face horizontally and
            # place it near the expected head location.
            horizontal_outside = max(
                0.0, x1 - fx, fx - x2
            ) / bw

            distance = (
                dx * dx
                + 1.5 * dy * dy
                + 2.0 * horizontal_outside
            )

            # Prefer a recognized owner face over unknown faces.
            owner_priority = (
                0 if result["status"] == "POTENTIAL_OWNER_MATCH" else 1
            )

            candidates.append(
                (owner_priority, distance, person_index, face_index)
            )

    # Step 3: Assign one face per person and one person per face.
    # Owner recognition gets first choice among plausible candidates.
    candidates.sort(key=lambda item: (item[0], item[1]))

    person_to_face = {}
    used_faces = set()

    for _, distance, person_index, face_index in candidates:
        if person_index in person_to_face:
            continue
        if face_index in used_faces:
            continue

        person_to_face[person_index] = face_index
        used_faces.add(face_index)

    # Step 4: Build results using the original return format.
    evaluated_people = []

    for person_index, (box, confidence) in enumerate(person_detections):
        face_index = person_to_face.get(person_index)

        if face_index is None:
            status, score = "NO_FACE", None
        else:
            face_result = face_results[face_index]
            status = face_result["status"]
            score = face_result["score"]

        evaluated_people.append({
            "box": box,
            "status": status,
            "score": score,
            "confidence": confidence,
        })

    # Return faces that were not associated with a person box.
    unmatched_faces = [
        result["face"]
        for index, result in enumerate(face_results)
        if index not in used_faces
    ]

    return evaluated_people, unmatched_faces


# --------------------------------------------------
# DRAW RESULTS
# --------------------------------------------------

def draw_person_result(frame, result):
    x1, y1, x2, y2 = result["box"]
    status = result["status"]
    score = result["score"]

    if status == "POTENTIAL_OWNER_MATCH":
        color = (0, 200, 0)
        label = f"OWNER MATCH ({score:.2f})" if score is not None else "OWNER MATCH"
    elif status == "UNKNOWN":
        color = (0, 165, 255)
        label = f"Unverified ({score:.2f})" if score is not None else "Unverified person"
    else:
        color = (0, 165, 255)
        label = "Unverified - face not visible"

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    text_y = max(20, y1 - 8)
    cv2.putText(
        frame, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX,
        0.5, color, 2, cv2.LINE_AA
    )
    return label


def draw_unmatched_face(frame, face, face_detector, face_recognizer,
                        owner_embeddings):
    """Draw a face-level fallback when no body box was associated."""
    x, y, width, height = map(int, face[:4])
    status, score = identify_face(
        frame, face, face_recognizer, owner_embeddings
    )

    if status == "POTENTIAL_OWNER_MATCH":
        color = (0, 200, 0)
        label = f"OWNER FACE ({score:.2f})" if score is not None else "OWNER FACE"
    elif status == "UNKNOWN":
        color = (0, 165, 255)
        label = f"Unverified face ({score:.2f})" if score is not None else "Unverified face"
    else:
        color = (0, 165, 255)
        label = "Unverified face"

    cv2.rectangle(frame, (x, y), (x + width, y + height), color, 2)
    cv2.putText(
        frame, label, (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX,
        0.5, color, 2, cv2.LINE_AA
    )


# --------------------------------------------------
# EVENT REPORTING
# --------------------------------------------------

def send_event(zone, worker_status, repeated_detections=1):
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


# --------------------------------------------------
# PHOTO PROCESSING
# --------------------------------------------------

def process_photo(photo_path):
    if not photo_path.exists():
        raise FileNotFoundError(f"Photo not found: {photo_path}")

    frame = cv2.imread(str(photo_path))
    if frame is None:
        raise ValueError(f"Could not read photo: {photo_path}")

    person_model = load_person_model()
    face_detector, face_recognizer, owner_embeddings = load_owner_models()

    person_detections = detect_people(frame, person_model)
    results, unmatched_faces = evaluate_frame(
        frame, person_detections, face_detector,
        face_recognizer, owner_embeddings
    )

    print(f"Person boxes detected: {len(results)}")
    print(f"Unmatched visible faces: {len(unmatched_faces)}")

    for result in results:
        print(draw_person_result(frame, result))

    for face in unmatched_faces:
        draw_unmatched_face(
            frame, face, face_detector, face_recognizer, owner_embeddings
        )

    output_dir = PROJECT_DIR / "runs" / "smartfarm"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "owner_photo_test.jpeg"

    if not cv2.imwrite(str(output_path), frame):
        raise RuntimeError(f"Could not save annotated photo: {output_path}")

    print(f"Annotated photo saved to: {output_path}")
    cv2.imshow("SmartFarm Guardian - Photo Test", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# --------------------------------------------------
# VIDEO PROCESSING
# --------------------------------------------------

def process_video(video_path):
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    person_model = load_person_model()
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
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not create video: {output_path}")

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

            person_detections = detect_people(frame, person_model)
            results, unmatched_faces = evaluate_frame(
                frame, person_detections, face_detector,
                face_recognizer, owner_embeddings
            )

            for result in results:
                draw_person_result(frame, result)

            for face in unmatched_faces:
                draw_unmatched_face(
                    frame, face, face_detector, face_recognizer,
                    owner_embeddings
                )

            cv2.putText(
                frame, f"People detected: {len(results)}",
                (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (255, 255, 255), 2, cv2.LINE_AA
            )

            now = time.time()
            if (
                results
                and now - last_event_time >= EVENT_COOLDOWN_SECONDS
                and any(item["status"] == "UNKNOWN" for item in results)
            ):
                send_event("field", "unknown", repeated_detections=1)
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


# --------------------------------------------------
# COMMAND-LINE ENTRY POINT
# --------------------------------------------------

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
