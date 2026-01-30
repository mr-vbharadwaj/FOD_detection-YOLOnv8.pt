import streamlit as st
import os
import cv2
import time
from ultralytics import YOLO

# -------------------------------------------------
# STREAMLIT PAGE CONFIG
# -------------------------------------------------

st.set_page_config(
    page_title="FOD Detection & Tracking",
    layout="centered"
)

st.title("🛫 FOD Detection & Tracking")
st.write(
    "Upload a runway video. "
    "The system will detect, track, and count **unique** FOD objects "
    "and generate an annotated output video."
)

# -------------------------------------------------
# CONSTANTS
# -------------------------------------------------

MODEL_PATH = "runs/detect/runs_fod/yolov8n_fod_v1/weights/best.pt"

CONF_THRESHOLD = 0.4
IMG_SIZE = 416
DRAW_BOXES = True

TEMP_DIR = "temp"
os.makedirs(TEMP_DIR, exist_ok=True)

# -------------------------------------------------
# LOAD MODEL (CACHED)
# -------------------------------------------------

@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

# -------------------------------------------------
# FILE UPLOAD
# -------------------------------------------------

uploaded_video = st.file_uploader(
    "Upload a video file",
    type=["mp4", "avi", "mov"]
)

if uploaded_video is not None:
    input_path = os.path.join(TEMP_DIR, uploaded_video.name)

    with open(input_path, "wb") as f:
        f.write(uploaded_video.read())

    st.success("✅ Video uploaded successfully")

    if st.button("▶ Run FOD Detection"):
        model = load_model()

        # Open video ONLY to read metadata
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            st.error("❌ Failed to open video")
            st.stop()

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        # Output writer
        output_path = os.path.join(TEMP_DIR, "fod_output.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        unique_fod_ids = set()

        progress_bar = st.progress(0)
        status_text = st.empty()

        processed_frames = 0
        start_time = time.time()

        st.info("🚀 Processing video…")

        # -------------------------------------------------
        # YOLO TRACKING LOOP (NO FRAME SKIPPING)
        # -------------------------------------------------

        for result in model.track(
            source=input_path,
            conf=CONF_THRESHOLD,
            imgsz=IMG_SIZE,
            tracker="bytetrack.yaml",
            stream=True,
            verbose=False
        ):
            processed_frames += 1

            boxes = result.boxes
            if boxes.id is not None:
                for tid in boxes.id.tolist():
                    unique_fod_ids.add(int(tid))

            frame = result.plot() if DRAW_BOXES else result.orig_img

            cv2.putText(
                frame,
                f"Unique FOD Count: {len(unique_fod_ids)}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2
            )

            out.write(frame)

            if processed_frames % 10 == 0:
                progress_bar.progress(
                    min(processed_frames / total_frames, 1.0)
                )
                status_text.text(
                    f"Frame {processed_frames}/{total_frames} | "
                    f"Unique FOD: {len(unique_fod_ids)}"
                )

        out.release()

        elapsed = time.time() - start_time
        avg_fps = processed_frames / elapsed if elapsed > 0 else 0

        st.success("✅ Processing complete")

        st.markdown("### 📊 Results Summary")
        st.write(f"**Unique FOD detected:** {len(unique_fod_ids)}")
        st.write(f"**Total frames processed:** {processed_frames}")
        st.write(f"**Average inference FPS:** {avg_fps:.2f}")

        st.video(output_path)

        with open(output_path, "rb") as f:
            st.download_button(
                "⬇ Download Output Video",
                f,
                "fod_detection_output.mp4",
                "video/mp4"
            )
