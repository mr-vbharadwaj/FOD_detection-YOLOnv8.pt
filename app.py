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
# CONSTANTS (TUNABLE PERFORMANCE KNOBS)
# -------------------------------------------------

MODEL_PATH = "runs/detect/runs_fod/yolov8n_fod_v1/weights/best.pt"

CONF_THRESHOLD = 0.4      # Detection confidence threshold
IMG_SIZE = 416            # Lower resolution → faster inference
FRAME_SKIP = 3            # Process every Nth frame (performance boost)
DRAW_BOXES = True         # Set False for max speed (backend mode)

TEMP_DIR = "temp"
os.makedirs(TEMP_DIR, exist_ok=True)

# -------------------------------------------------
# LOAD MODEL (CACHED — VERY IMPORTANT)
# -------------------------------------------------

@st.cache_resource
def load_model():
    """
    Load YOLO model ONCE.
    Streamlit reruns scripts frequently — caching prevents reloading
    the model every time, which is expensive.
    """
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

    # Save uploaded video to disk
    with open(input_path, "wb") as f:
        f.write(uploaded_video.read())

    st.success("✅ Video uploaded successfully")

    # -------------------------------------------------
    # RUN DETECTION BUTTON
    # -------------------------------------------------

    if st.button("▶ Run FOD Detection"):
        model = load_model()

        # Open input video
        cap = cv2.VideoCapture(input_path)

        if not cap.isOpened():
            st.error("❌ Failed to open video file")
            st.stop()

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Setup output video
        output_path = os.path.join(TEMP_DIR, "fod_output.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Tracking state
        unique_fod_ids = set()

        # UI elements
        progress_bar = st.progress(0)
        status_text = st.empty()

        processed_frames = 0
        start_time = time.time()

        st.info("🚀 Processing video… This may take a few minutes.")

        # -------------------------------------------------
        # YOLOv8 + BYTE TRACKING LOOP
        # -------------------------------------------------

        for idx, result in enumerate(
            model.track(
                source=input_path,
                conf=CONF_THRESHOLD,
                imgsz=IMG_SIZE,
                tracker="bytetrack.yaml",
                stream=True,
                verbose=False
            )
        ):
            # Skip frames for performance
            if idx % FRAME_SKIP != 0:
                continue

            processed_frames += 1

            # Collect UNIQUE object IDs
            boxes = result.boxes
            if boxes.id is not None:
                for track_id in boxes.id.tolist():
                    unique_fod_ids.add(int(track_id))

            # Draw output frame
            if DRAW_BOXES:
                frame = result.plot()
            else:
                frame = result.orig_img

            # Overlay unique count
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

            # UI updates (throttled to avoid lag)
            if processed_frames % 10 == 0:
                progress_bar.progress(min(processed_frames / frame_count, 1.0))
                status_text.text(
                    f"Processed {processed_frames} frames | "
                    f"Unique FOD: {len(unique_fod_ids)}"
                )

        # -------------------------------------------------
        # CLEANUP & RESULTS
        # -------------------------------------------------

        cap.release()
        out.release()

        elapsed = time.time() - start_time
        avg_fps = processed_frames / elapsed if elapsed > 0 else 0

        st.success("✅ Processing complete")

        st.markdown("### 📊 Results Summary")
        st.write(f"**Unique FOD detected:** {len(unique_fod_ids)}")
        st.write(f"**Processed frames:** {processed_frames}")
        st.write(f"**Average FPS:** {avg_fps:.2f}")

        # Show output video
        st.video(output_path)

        # Download button
        with open(output_path, "rb") as f:
            st.download_button(
                label="⬇ Download Output Video",
                data=f,
                file_name="fod_detection_output.mp4",
                mime="video/mp4"
            )
