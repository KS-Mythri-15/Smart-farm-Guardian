
from pathlib import Path
import cv2
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_DIR / "models"
PROFILE_PATH = (
    PROJECT_DIR / "backend" / "known_faces" / "owner_profile.npz"
)
TEST_IMAGE = (
    PROJECT_DIR / "backend" / "known_faces" / "owner_test.jpeg"
)

detector = cv2.FaceDetectorYN.create(
    str(MODEL_DIR / "face_detection_yunet_2023mar.onnx"),
    "",
    (320, 320),
    0.8,
    0.3,
    5000
)

recognizer = cv2.FaceRecognizerSF.create(
    str(MODEL_DIR / "face_recognition_sface_2021dec.onnx"),
    ""
)


def main():
    if not PROFILE_PATH.exists():
        raise FileNotFoundError("Owner profile is missing. Enroll first.")

    if not TEST_IMAGE.exists():
        raise FileNotFoundError(f"Test photo not found: {TEST_IMAGE}")

    profile = np.load(PROFILE_PATH)
    known_embeddings = profile["embeddings"]

    image = cv2.imread(str(TEST_IMAGE))
    if image is None:
        raise RuntimeError("Could not read the test image.")

    height, width = image.shape[:2]
    detector.setInputSize((width, height))
    _, faces = detector.detect(image)

    if faces is None or len(faces) == 0:
        print("NO FACE DETECTED — try a clearer photo.")
        return

    if len(faces) > 1:
        print("MULTIPLE FACES DETECTED — use a photo containing only you.")
        return

    aligned_face = recognizer.alignCrop(image, faces[0])
    feature = recognizer.feature(aligned_face).flatten()
    feature /= np.linalg.norm(feature) + 1e-10

    # SFace uses cosine similarity to compare face representations.
    similarities = known_embeddings @ feature
    best_score = float(np.max(similarities))

    print(f"Best similarity score: {best_score:.4f}")
    print("This score is a comparison result, not a probability.")

    # Provisional threshold for testing only; calibrate before use.
    if best_score >= 0.40:
        print("POTENTIAL OWNER MATCH — further verification required.")
    else:
        print("UNKNOWN — owner match not established.")


if __name__ == "__main__":
    main()
