import streamlit as st
import cv2
import tempfile
import torch
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from Ultralytics import YOLO

# Load YOLOv5 model
model = YOLO('yolov5nu')

# Page config
st.set_page_config(page_title="Object Detection Dashboard", layout="wide")

# Custom CSS
st.markdown("""
    <style>
    .stApp {
        background-color: #121212;
        color: #00ffe1;
        font-family: 'Segoe UI';
    }
    .css-1v0mbdj, .css-1d391kg {
        background-color: #1f1f1f;
        border: 1px solid #00ffe1;
        color: #00ffe1;
    }
    .css-1v0mbdj:hover {
        background-color: #00ffe1;
        color: #121212;
    }
    </style>
""", unsafe_allow_html=True)

st.title("Object Detection Dashboard")
st.subheader("Upload a video, run detection, and explore the data")

uploaded_file = st.file_uploader("🎥 Upload your video", type=["mp4", "avi"])
save_dir = st.text_input("📁 Enter save directory", "C:/Users/zeena/Desktop/advanced ai")

if uploaded_file and save_dir:
    with tempfile.NamedTemporaryFile(delete=False) as temp_input:
        temp_input.write(uploaded_file.read())
        input_path = temp_input.name

    output_path = os.path.join(save_dir, "annotated_output.mp4")

    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_bar = st.progress(0)
    preview_frame = st.empty()

    # Data collection
    detection_data = []

    st.text("🔍 Processing frames...")

    current_frame = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        results = model(frame)[0]
        annotated_frame = results.plot()
        out.write(annotated_frame)

        # Collect detection stats
        labels = results.names
        detected = results.boxes.cls.tolist()
        counts = pd.Series(detected).value_counts().to_dict()
        frame_stats = {labels[int(k)]: v for k, v in counts.items()}
        frame_stats["frame"] = current_frame
        detection_data.append(frame_stats)

        current_frame += 1
        preview_frame = st.empty()
        if current_frame % 30 == 0:
            img_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            preview_frame.image(img_rgb, caption=f"Frame {current_frame}", use_container_width=True)

        progress_bar.progress(min(current_frame / frame_count, 1.0))

    cap.release()
    out.release()

    st.success("✅ Detection complete!")
    st.video(output_path)

    with open(output_path, "rb") as file:
        st.download_button("📥 Download Annotated Video", file.read(), "annotated_output.mp4", mime="video/mp4")

    # Dashboard
    st.header("📊 Detection Dashboard")

    df = pd.DataFrame(detection_data).fillna(0)
    st.dataframe(df.style.background_gradient(cmap="cool"), use_container_width=True)

    st.subheader("🔥 Detection Heatmap")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(df.drop(columns=["frame"]).T, cmap="mako", cbar=True, ax=ax)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Object Class")
    st.pyplot(fig)
