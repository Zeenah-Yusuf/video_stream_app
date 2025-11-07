import streamlit as st
import cv2
import tempfile
import os
import pandas as pd
import numpy as np
from ultralytics import YOLO
from sort import Sort
from PIL import Image

# --- Page Config ---
st.set_page_config(page_title="YOLOv5 Detector", layout="wide")

# --- Styling ---
st.markdown("""
    <style>
    .stApp { background-color: #121212; color: #00ffe1; font-family: 'Segoe UI'; }
    .css-1v0mbdj, .css-1d391kg {
        background-color: #1f1f1f; border: 1px solid #00ffe1; color: #00ffe1;
    }
    .css-1v0mbdj:hover { background-color: #00ffe1; color: #121212; }
    </style>
""", unsafe_allow_html=True)

# --- Sidebar ---
st.sidebar.title("⚙ Input Settings")
input_mode = st.sidebar.radio("Choose Input Mode", ["Video Upload", "Camera Snapshot"])
confidence = st.sidebar.slider(
    "Detection Confidence", 0.2, 0.95, 0.2,
    help="Lower values detect more objects but may include false positives"
)
frame_skip = st.sidebar.slider("Frame Skip (for speed)", 1, 5, 2)

# --- Main Title ---
st.title("YOLOv5 Object Detection with Tracking & Alerts")

# --- Load Model (cached) ---
@st.cache_resource
def load_model():
    return YOLO('yolov5nu')

model = load_model()
tracker = Sort()
alert_classes = ["knife", "gun", "vehicle", "person"]

# --- Video Upload Mode ---
if input_mode == "Video Upload":
    video_file = st.sidebar.file_uploader("Upload a video", type=["mp4", "mov", "avi"])
    run_detection = st.sidebar.button("Run Detection")

    if video_file and run_detection:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
            temp_video.write(video_file.read())
            temp_video_path = temp_video.name

        cap = cv2.VideoCapture(temp_video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        output_path = os.path.join(tempfile.gettempdir(), "annotated_output.mp4")
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        detection_log = []
        frame_count = 0
        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % frame_skip != 0:
                continue

            timestamp = round(frame_count / fps, 2)
            results = model(frame, conf=confidence)
            boxes = results[0].boxes

            detections = []
            class_map = {}

            if boxes is not None and boxes.cls is not None:
                for cls_id, conf_score, box in zip(boxes.cls.tolist(), boxes.conf.tolist(), boxes.xyxy.tolist()):
                    if conf_score >= confidence:
                        x1, y1, x2, y2 = map(int, box)
                        detections.append([x1, y1, x2, y2, conf_score])
                        class_map[(x1, y1, x2, y2)] = model.names[int(cls_id)]

            track_results = tracker.update(np.array(detections))
            annotated_frame = results[0].plot()
            out.write(annotated_frame)
            stframe.image(annotated_frame, channels="BGR", use_container_width=True)

            for track in track_results:
                x1, y1, x2, y2, track_id = map(int, track)
                key = (x1, y1, x2, y2)
                class_name = class_map.get(key, "Unknown")
                conf_score = next((d[4] for d in detections if tuple(map(int, d[:4])) == key), 0)
                alert_triggered = "Yes" if class_name in alert_classes else "No"

                if alert_triggered == "Yes":
                    st.warning(f"⚠️ Alert: {class_name} detected at {timestamp}s (Frame {frame_count})")

                detection_log.append({
                    "Frame": frame_count,
                    "Timestamp (s)": timestamp,
                    "Object ID": int(track_id),
                    "Class": class_name,
                    "Confidence (%)": round(conf_score * 100, 2),
                    "X1": x1,
                    "Y1": y1,
                    "X2": x2,
                    "Y2": y2,
                    "Alert Triggered": alert_triggered
                })

        cap.release()
        out.release()

        # --- Download Annotated Video ---
        with open(output_path, "rb") as f:
            st.download_button("📥 Download Annotated Video", f, file_name="annotated_output.mp4")

        # --- Download Excel Log ---
        if detection_log:
            df_log = pd.DataFrame(detection_log)
            excel_path = os.path.join(tempfile.gettempdir(), "detection_log.xlsx")
            df_log.to_excel(excel_path, index=False)

            with open(excel_path, "rb") as ef:
                st.download_button("📊 Download Detection Log (Excel)", ef, file_name="detection_log.xlsx")
        # --- Dashboard Summary ---
        if detection_log:
            st.subheader("📈 Detection Summary")
            df_summary = pd.DataFrame(detection_log)
        
            total_frames = df_summary["Frame"].nunique() if "Frame" in df_summary else 1
            total_detections = len(df_summary)
            total_alerts = df_summary[df_summary["Alert Triggered"] == "Yes"].shape[0]
            class_counts = df_summary["Class"].value_counts()
        
            col1, col2, col3 = st.columns(3)
            col1.metric("Frames Processed", total_frames)
            col2.metric("Total Detections", total_detections)
            col3.metric("Alerts Triggered", total_alerts)
        
            st.markdown("#### 🔍 Detected Object Breakdown")
            st.dataframe(class_counts.rename("Count"))
            
# --- Camera Snapshot Mode ---
elif input_mode == "Camera Snapshot":
    camera_image = st.camera_input("📸 Take a snapshot")
    if camera_image:
        image = Image.open(camera_image)
        frame = np.array(image)
        results = model(frame, conf=confidence)
        boxes = results[0].boxes

        detection_log = []
        st.image(results[0].plot(), channels="BGR", use_container_width=True)

        if boxes is not None and boxes.cls is not None:
            for cls_id, conf_score, box in zip(boxes.cls.tolist(), boxes.conf.tolist(), boxes.xyxy.tolist()):
                if conf_score >= confidence:
                    x1, y1, x2, y2 = map(int, box)
                    class_name = model.names[int(cls_id)]
                    alert_triggered = "Yes" if class_name in alert_classes else "No"
                    if alert_triggered == "Yes":
                        st.warning(f"⚠️ Alert: {class_name} detected in snapshot!")

                    detection_log.append({
                        "Class": class_name,
                        "Confidence (%)": round(conf_score * 100, 2),
                        "X1": x1,
                        "Y1": y1,
                        "X2": x2,
                        "Y2": y2,
                        "Alert Triggered": alert_triggered
                    })

        # --- Download Excel Log for Snapshot ---
        if detection_log:
            df_log = pd.DataFrame(detection_log)
            excel_path = os.path.join(tempfile.gettempdir(), "snapshot_log.xlsx")
            df_log.to_excel(excel_path, index=False)

            with open(excel_path, "rb") as ef:
                st.download_button(" Download Snapshot Log (Excel)", ef, file_name="snapshot_log.xlsx")
        # --- Dashboard Summary ---
        if detection_log:
            st.subheader(" Detection Summary")
            df_summary = pd.DataFrame(detection_log)
        
            total_frames = df_summary["Frame"].nunique() if "Frame" in df_summary else 1
            total_detections = len(df_summary)
            total_alerts = df_summary[df_summary["Alert Triggered"] == "Yes"].shape[0]
            class_counts = df_summary["Class"].value_counts()
        
            col1, col2, col3 = st.columns(3)
            col1.metric("Frames Processed", total_frames)
            col2.metric("Total Detections", total_detections)
            col3.metric("Alerts Triggered", total_alerts)
        
            st.markdown("####  Detected Object Breakdown")
            st.dataframe(class_counts.rename("Count"))

