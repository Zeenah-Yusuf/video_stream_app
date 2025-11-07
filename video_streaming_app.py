import streamlit as st
import cv2
import tempfile
import os
import pandas as pd
import numpy as np
import time
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

# --- Sidebar Settings ---
st.sidebar.title("⚙ Input Settings")
input_mode = st.sidebar.radio("Choose Input Mode", ["Video Upload", "Camera Snapshot"])
confidence = st.sidebar.slider("Detection Confidence", 0.2, 0.95, 0.2)
frame_skip = st.sidebar.slider("Frame Skip (for speed)", 1, 5, 2)

# --- Load Model ---
@st.cache_resource
def load_model():
    return YOLO('yolov5nu')

model = load_model()
tracker = Sort()
available_classes = list(model.names.values())
alert_classes = st.sidebar.multiselect("Alert Classes (trigger warnings)", options=available_classes, default=["knife", "gun", "person"])

# --- Detection Logger ---
def log_detection(cls_id, conf_score, box, model_names, alert_classes):
    x1, y1, x2, y2 = map(int, box)
    class_name = model_names[int(cls_id)]
    alert_triggered = "Yes" if class_name in alert_classes else "No"
    return {
        "Class": class_name,
        "Confidence (%)": round(conf_score * 100, 2),
        "X1": x1, "Y1": y1, "X2": x2, "Y2": y2,
        "Alert Triggered": alert_triggered
    }

# --- Excel Export ---
@st.cache_data
def export_excel(df, filename):
    path = os.path.join(tempfile.gettempdir(), filename)
    df.to_excel(path, index=False)
    return path

# --- Summary Dashboard ---
def show_summary(log, mode="Video"):
    df = pd.DataFrame(log)
    st.subheader("📈 Detection Summary")
    total_frames = df["Frame"].nunique() if mode == "Video" else 1
    total_detections = len(df)
    total_alerts = df[df["Alert Triggered"] == "Yes"].shape[0]
    class_counts = df["Class"].value_counts()
    col1, col2, col3 = st.columns(3)
    col1.metric("Frames Processed", total_frames)
    col2.metric("Total Detections", total_detections)
    col3.metric("Alerts Triggered", total_alerts)
    with st.expander("🔍 Detected Object Breakdown"):
        st.dataframe(class_counts.rename("Count"))

# --- Video Detection ---
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    width, height = int(cap.get(3)), int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    output_path = os.path.join(tempfile.gettempdir(), "annotated_output.mp4")
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    detection_log = []
    frame_count = 0
    stframe = st.empty()
    progress = st.progress(0)
    status = st.empty()
    st.info("🔍 Detection in progress...")
    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % frame_skip != 0:
            continue
        frame = cv2.resize(frame, (640, 360))
        timestamp = round(frame_count / fps, 2)
        results = model.predict(frame, conf=confidence)
        boxes = results[0].boxes
        detections = []
        class_map = {}

        if boxes is not None and boxes.cls is not None:
            for cls_id, conf_score, box in zip(boxes.cls.tolist(), boxes.conf.tolist(), boxes.xyxy.tolist()):
                if conf_score >= confidence:
                    x1, y1, x2, y2 = map(int, box)
                    key = (x1, y1, x2, y2, round(conf_score, 2))
                    detections.append([x1, y1, x2, y2, conf_score])
                    class_map[key] = model.names[int(cls_id)]

        track_results = tracker.update(np.array(detections))
        annotated_frame = results[0].plot()
        out.write(annotated_frame)
        stframe.image(annotated_frame, channels="BGR", use_container_width=True)
        elapsed = time.time() - start_time
        fps_display = round(frame_count / elapsed, 2)
        progress.progress(frame_count / total_frames)
        status.text(f"Frame {frame_count}/{total_frames} | FPS: {fps_display}")

        for track in track_results:
            x1, y1, x2, y2, track_id = map(int, track)
            key = (x1, y1, x2, y2, round(next((d[4] for d in detections if tuple(map(int, d[:4])) == (x1, y1, x2, y2)), 0), 2))
            class_name = class_map.get(key, "Unknown")
            conf_score = next((d[4] for d in detections if tuple(map(int, d[:4])) == (x1, y1, x2, y2)), 0)
            alert_triggered = "Yes" if class_name in alert_classes else "No"
            if alert_triggered == "Yes":
                st.warning(f"⚠ Alert: {class_name} detected at {timestamp}s (Frame {frame_count})")
            detection_log.append({
                "Frame": frame_count,
                "Timestamp (s)": timestamp,
                "Object ID": int(track_id),
                "Class": class_name,
                "Confidence (%)": round(conf_score * 100, 2),
                "X1": x1, "Y1": y1, "X2": x2, "Y2": y2,
                "Alert Triggered": alert_triggered
            })

    cap.release()
    out.release()
    st.success("✅ Detection complete!")
    return detection_log, output_path

# --- Snapshot Detection ---
def process_snapshot(image):
    frame = np.array(image)
    st.info("🔍 Detection in progress...")
    results = model.predict(frame, conf=confidence)
    boxes = results[0].boxes
    detection_log = []
    st.image(results[0].plot(), channels="BGR", use_container_width=True)

    if boxes is not None and boxes.cls is not None:
        for cls_id, conf_score, box in zip(boxes.cls.tolist(), boxes.conf.tolist(), boxes.xyxy.tolist()):
            if conf_score >= confidence:
                entry = log_detection(cls_id, conf_score, box, model.names, alert_classes)
                detection_log.append(entry)
                if entry["Alert Triggered"] == "Yes":
                    st.warning(f"⚠ Alert: {entry['Class']} detected in snapshot!")

    st.success("✅ Detection complete!")
    return detection_log

# --- Main Logic ---
st.title("YOLOv5 Object Detection with Tracking & Alerts")

if input_mode == "Video Upload":
    video_file = st.sidebar.file_uploader("Upload a video", type=["mp4", "mov", "avi"])
    run_detection = st.sidebar.button("Run Detection")
    if video_file and run_detection:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
            temp_video.write(video_file.read())
            temp_video_path = temp_video.name
        log, output_path = process_video(temp_video_path)
        with open(output_path, "rb") as f:
            st.download_button("📥 Download Annotated Video", f, file_name="annotated_output.mp4")
        if log:
            df_log = pd.DataFrame(log)
            excel_path = export_excel(df_log, "detection_log.xlsx")
            with open(excel_path, "rb") as ef:
                st.download_button("📊 Download Detection Log (Excel)", ef, file_name="detection_log.xlsx")
            show_summary(log)

elif input_mode == "Camera Snapshot":
    camera_image = st.camera_input("📸 Take a snapshot")
    if camera_image:
        image = Image.open(camera_image)
        log = process_snapshot(image)
        if log:
            df_log = pd.DataFrame(log)
            excel_path = export_excel(df_log, "
