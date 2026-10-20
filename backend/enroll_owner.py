
from pathlib import Path
import cv2
import numpy as np

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
PHOTO_DIR = PROJECT_DIR / "backend" / "known_faces" / "owner"
MODEL_DIR = PROJECT_DIR / "models"
OUTPUT_FILE = PROJECT_DIR / "backend" / "known_faces" / "owner_profile.npz"

FACE_DETECTOR_PATH = MODEL_DIR / "face_detection_yunet_2023mar.onnx"
FACE_RECOGNIZER_PATH = MODEL_DIR / "face_recognition_sface_2021dec.onnx"


def main():
    # Verify required files exist
    for path in [FACE_DETECTOR_PATH, FACE_RECOGNIZER_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

    if not PHOTO_DIR.exists():
        raise FileNotFoundError(f"Photo folder not found: {PHOTO_DIR}")

    photos = [
        p for p in PHOTO_DIR.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    ]

    if not photos:
        raise RuntimeError(f"No photos found in {PHOTO_DIR}")

    detector = cv2.FaceDetectorYN.create(
        str(FACE_DETECTOR_PATH),
        "",
        (320, 320),
        0.8,    # Face detection confidence threshold
        0.3,    # Non-maximum suppression threshold
        5000
    )

    recognizer = cv2.FaceRecognizerSF.create(
        str(FACE_RECOGNIZER_PATH), ""
    )

    embeddings = []
    successful_photos = 0

    for photo_path in photos:
        image = cv2.imread(str(photo_path))

        if image is None:
            print(f"SKIPPED: Could not read {photo_path.name}")
            continue

        height, width = image.shape[:2]
        detector.setInputSize((width, height))

        _, faces = detector.detect(image)

        if faces is None or len(faces) == 0:
            print(f"NO FACE FOUND: {photo_path.name}")
            continue

        # Choose the largest detected face in this enrollment photo.
        face = max(
            faces,
            key=lambda f: max(0, f[2]) * max(0, f[3])
        )

        aligned_face = recognizer.alignCrop(image, face)
        feature = recognizer.feature(aligned_face).flatten()

        # Store a normalized face representation.
        feature = feature / (np.linalg.norm(feature) + 1e-10)
        embeddings.append(feature.astype(np.float32))

        successful_photos += 1
        print(f"ENROLLED: {photo_path.name}")

    if successful_photos < 3:
        raise RuntimeError(
            f"Only {successful_photos} usable photos found. "
            "Use at least 3 clear photos with a detectable face."
        )

    # Save the face representations locally, not the original photos.
    np.savez_compressed(
        OUTPUT_FILE,
        embeddings=np.stack(embeddings),
        identity=np.array("OWNER")
    )

    print("\nOWNER ENROLLMENT COMPLETE")
    print("Usable photos:", successful_photos)
    print("Owner profile:", OUTPUT_FILE)
    print("Original photos remain on your computer.")


if __name__ == "__main__":
    main()
