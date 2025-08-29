import streamlit as st
import cv2
import tempfile
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from ultralytics import YOLO

# --- Page Config ---
st.set_page_config(page_title="YOLOv5 Video Detector", layout="wide")

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

# --- Sidebar ---
st.sidebar.title("⚙ Settings")
video_file = st.sidebar.file_uploader("Upload a video", type=["mp4", "mov", "avi"])
confidence = st.sidebar.slider("Detection Confidence", 0.2, 0.9, 0.5)
run_detection = st.sidebar.button("Run Detection")

# --- Main Title ---
st.title("YOLOv5 Object Detection")

# Load YOLOv5 model
model = YOLO('yolov5nu')

# --- Detection Logic ---
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

    object_counts = []
    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=confidence)
        annotated_frame = results[0].plot()
        out.write(annotated_frame)

        labels = results[0].boxes.cls.tolist()
        object_counts.extend(labels)

        stframe.image(annotated_frame, channels="BGR", use_column_width=True)

    cap.release()
    out.release()

    # --- Download Annotated Video ---
    with open(output_path, "rb") as f:
        st.download_button("📥 Download Annotated Video", f, file_name="annotated_output.mp4")

    # --- Correlation Heatmap ---
    st.subheader("📊 Object Detection Correlation Heatmap")
    df = pd.DataFrame(object_counts, columns=["class_id"])
    df["class_name"] = df["class_id"].apply(lambda x: model.names[int(x)])
    count_df = df["class_name"].value_counts().reset_index()
    count_df.columns = ["Object", "Count"]

    corr_matrix = pd.DataFrame(index=model.names.values(), columns=model.names.values()).fillna(0)
    for obj in count_df["Object"]:
        corr_matrix.loc[obj, obj] = count_df[count_df["Object"] == obj]["Count"].values[0]

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr_matrix, annot=True, fmt=".0f", cmap="coolwarm", ax=ax)
    st.pyplot(fig)
