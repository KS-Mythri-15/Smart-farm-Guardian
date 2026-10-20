
from ultralytics import YOLO
from pathlib import Path

video_path = Path("videos/people.mp4")

if not video_path.exists():
    raise FileNotFoundError(
        "Video not found! Put your MP4 at videos/people.mp4"
    )

model = YOLO("yolo11n.pt")

results = model.predict(
    source=str(video_path),
    classes=[0],       # Person class only
    conf=0.4,
    save=True,
    stream=True,
    verbose=False
)

total_frames = 0
frames_with_people = 0
total_person_detections = 0

for result in results:
    total_frames += 1
    person_count = len(result.boxes)

    if person_count > 0:
        frames_with_people += 1
        total_person_detections += person_count

    if total_frames % 30 == 0:
        print(
            f"Frames processed: {total_frames} | "
            f"People in current frame: {person_count}"
        )

print("\n--- VIDEO DETECTION SUMMARY ---")
print("Total frames processed:", total_frames)
print("Frames containing people:", frames_with_people)
print("Total person detections across frames:", total_person_detections)
print("Annotated video saved under runs/detect/")
