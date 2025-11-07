# Real-Time Object Detection & Tracking App

This project is a Streamlit-powered web application that performs real-time object detection and tracking using YOLOv5 and SORT.
Designed with security and surveillance in mind, it allows users to upload videos or capture snapshots via webcam,
then analyzes the content for high-confidence detections and persistent object tracking.                                                                                                                                                                                                                                                                              

## Features

- **Dual Input Modes**: Choose between uploading a video or capturing a live snapshot from your webcam.
- **YOLOv5 Detection**: Uses a custom-trained YOLOv5 model for accurate object recognition.
- **SORT Tracking**: Assigns unique IDs to objects and tracks them across video frames.
- **High Confidence Threshold**: Default detection confidence set to 85% for reliable results.
- **Real-Time Alerts**: Flags critical objects like weapons, vehicles, or intruders with instant warnings.
- **Timestamped Logs**: Each detection includes frame number and time in seconds.
- **Downloadable Outputs**:
  - Annotated video with bounding boxes and tracked IDs
  - Excel log of all detections with coordinates, confidence, and alerts

## Use Cases

- Security monitoring and intrusion detection
- Smart city surveillance
- Retail analytics and crowd tracking
- Investor demos and AI capability showcases


## Installation

To run locally or deploy on Streamlit Cloud, make sure your environment includes:

### requirements.txt
streamlit
torch
torchvision
opencv-python-headless
tqdm
numpy
seaborn
pandas
ultralytics
pillow
sort-tracker
openpyxl

### packages.txt
libgl1
libglib2.0-0
ffmpeg

### .gitattributes (for Git LFS)
*.mp4 filter=lfs diff=lfs merge=lfs -text
*.avi filter=lfs diff=lfs merge=lfs -text
*.mov filter=lfs diff=lfs merge=lfs -text
*.pt  filter=lfs diff=lfs merge=lfs -text
*.onnx filter=lfs diff=lfs merge=lfs -text
*.xlsx filter=lfs diff=lfs merge=lfs -text

## How to Use

1. Launch the app via Streamlit or locally.
2. Select your input mode: video upload or camera snapshot.
3. Adjust the detection confidence if needed.
4. Run detection and view results in real time.
5. Download annotated video and Excel logs for analysis.


## File Structure

video_stream_app/
├── video_streaming_app.py      # Main Streamlit app
├── requirements.txt
├── packages.txt
├── .gitattributes
├── .gitignore
├── models/
│   └── yolov5nu.pt             # YOLOv5 model weights


## Contributing

Pull requests are welcome! If you have ideas for new features like heatmaps, REST API integration, or dashboard analytics 
feel free to open an issue or submit a PR.


## Contact

For questions, feedback, or demo requests, reach out via GitHub or email.

## License

This project is open-source and available under the MIT License.
---

Let me know if you’d like a version tailored for investor presentations or a public-facing landing page. I can also help you write a `streamlit_app.py` launcher or a `Dockerfile` if you plan to containerize it.
