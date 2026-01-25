"""
run_video_with_tracking.py

Usage:
    python run_video_with_tracking.py test_files/input_video.mp4

What this script does:
- Runs YOLOv8 + ByteTrack
- Assigns unique IDs to FOD objects
- Counts UNIQUE debris (no double counting)
- Saves annotated output video
"""

import sys
import os
import cv2
from ultralytics import YOLO

# -----------------------------
# 1. INPUT VALIDATION
# -----------------------------

if len(sys.argv) < 2:
    print("Usage: python run_video_with_tracking.py <video_path>")
    sys.exit(1)

VIDEO_PATH = sys.argv[1]
assert os.path.exists(VIDEO_PATH), "Video file not found"

# -----------------------------
# 2. LOAD MODEL
# -----------------------------

MODEL_PATH = "runs/detect/runs_fod/yolov8n_fod_v1/weights/best.pt"
model = YOLO(MODEL_PATH)

# -----------------------------
# 3. OPEN VIDEO
# -----------------------------

cap = cv2.VideoCapture(VIDEO_PATH)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# -----------------------------
# 4. OUTPUT VIDEO SETUP
# -----------------------------

os.makedirs("output", exist_ok=True)
output_path = "output/fod_tracked_output.mp4"

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# -----------------------------
# 5. TRACKING STATE
# -----------------------------

# This set stores UNIQUE tracked object IDs
unique_fod_ids = set()

frame_idx = 0

print("▶ Processing video with tracking...")

# -----------------------------
# 6. PROCESS FRAMES
# -----------------------------

for result in model.track(
    source=VIDEO_PATH,
    conf=0.4,
    tracker="bytetrack.yaml",
    stream=True,
    verbose=False
):
    frame_idx += 1

    frame = result.orig_img
    boxes = result.boxes

    if boxes.id is not None:
        for track_id in boxes.id.tolist():
            unique_fod_ids.add(int(track_id))

    # Draw boxes + IDs
    annotated_frame = result.plot()

    # Overlay UNIQUE count
    cv2.putText(
        annotated_frame,
        f"Unique FOD Count: {len(unique_fod_ids)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 255),
        2
    )

    out.write(annotated_frame)

    if frame_idx % 50 == 0:
        print(f"Processed {frame_idx} frames | Unique FOD: {len(unique_fod_ids)}")

# -----------------------------
# 7. CLEANUP
# -----------------------------

out.release()
cap.release()

print("✅ Tracking complete")
print(f"📁 Output video: {output_path}")
print(f"📊 TOTAL UNIQUE FOD OBJECTS: {len(unique_fod_ids)}")
