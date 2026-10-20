
from ultralytics import YOLO

# Load a small pretrained person-detection model
model = YOLO("yolo11n.pt")

# Test on a sample image available online
results = model.predict(
    source="https://ultralytics.com/images/bus.jpg",
    classes=[0],  # COCO class 0 = person
    conf=0.4,
    save=True,
    verbose=True
)

print("\nDetection test completed.")
print("Detected people:", len(results[0].boxes))
print("Annotated image saved by YOLO.")
