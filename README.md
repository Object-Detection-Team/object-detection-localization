# YOLOv8 Real-Time Object Detection and Localization
Browser-based, Streamlit-powered YOLOv8n app for live object detection and localization.

## Overview
- Real-time detection and localization using pretrained YOLOv8n for fast inference.
- Built with Python, Streamlit, Ultralytics YOLOv8, OpenCV, and PyTorch.
- Designed as an “Introduction to Artificial Intelligence” course project.

## Features
- Live webcam detection with bounding boxes, labels, and confidence scores.
- Two-page UI (Overview and Detection) for quick orientation and streaming.
- Auto-selects the best available device (CPU or MPS) for inference.
- In-browser WebRTC streaming; audio is disabled for privacy.

## Demo
- **Overview tab:** Project summary, method snapshot, dataset and tool highlights, team, and a “Start Live Detection” call-to-action.
- **Detection tab:** Session controls (device info and start instructions), centered live stream widget, and guidance on expected results and best practices.
- Start the stream, grant camera access, and watch detections update in real time.

## Tech Stack
- Python
- Streamlit
- Ultralytics YOLOv8
- OpenCV
- PyTorch

## Project Structure
- `main.py` — Streamlit app (Overview + Detection tabs) and YOLOv8n inference pipeline.
- `requirements.txt` — Python dependencies.
- `yolov8n.pt` — Pretrained YOLOv8n weights.
- `sampled-unlabeled-data/`, `unlabeled-data/` — Dataset placeholders for future work.
- `notebook.ipynb` — Supporting experiments (optional).

## Installation
```bash
git clone <repo-url>
cd object-detection-localization

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
