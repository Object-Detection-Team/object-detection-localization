# 🎯 Multi-Model YOLO Vision System

A state-of-the-art biological object detection system comparing **YOLOv5**, **YOLOv8**, and **YOLOv11** architectures. This project implements real-time localization and classification of fruit species using a modern, optimized Streamlit web interface.

![Project Status](https://img.shields.io/badge/Status-Active-success)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![YOLO](https://img.shields.io/badge/YOLO-v5%20%7C%20v8%20%7C%20v11-orange)

---

## 📋 Project Overview

This system provides a comprehensive platform to compare different YOLO (You Only Look Once) architectures. It features a **responsive dark-mode UI**, **real-time WebRTC inference**, and an **analytics dashboard** to visualize performance metrics across different models.

### ✨ Key Features

*   **⚡ Multi-Model Architecture**: Seamlessly switch between YOLOv5 (Speed), YOLOv8 (Balanced), and YOLOv11 (Accuracy) in real-time.
*   **🎥 Real-Time Detection**: WebRTC integration for low-latency live camera feed inference (~30 FPS).
*   **🎨 Premium UI/UX**: Fully responsive, gold-accented dark theme tailored for professional presentation.
*   **📊 Analytics Dashboard**: Built-in performance tracking, confidence distribution, and class statistics.
*   **🧪 Comparative Analysis**: Detailed breakdown of accuracy (mAP), recall, and inference speed for each model.

---

## 🛠️ Technology Stack

*   **Core**: Python 3.8+, PyTorch
*   **Models**: Ultralytics YOLO (v5, v8, v11)
*   **Interface**: Streamlit, Streamlit-WebRTC
*   **Computer Vision**: OpenCV, PyAV
*   **Data Processing**: NumPy, Pandas

---

## 📁 Project Structure

```bash
object-detection-localization/
├── main.py                 # 🚀 Main entry point (Streamlit App)
├── requirements.txt        # Dependency list
├── notebooks/              # 📓 Training notebooks
│   ├── train_yolov5.ipynb
│   ├── train_yolov8.ipynb
│   └── train_yolov11.ipynb
├── training_results/       # 📈 Model artifacts & metrics
│   ├── yolov5_fruits/      # YOLOv5 weights & graphs
│   ├── yolov8_fruits/      # YOLOv8 weights & graphs
│   └── yolo11_fruits/      # YOLOv11 weights & graphs
└── labeled-datasets/       # 📂 Fruit classification datasets
```

---

## 🚀 Installation & Setup

### 1. Clone Repository
```bash
git clone https://github.com/Object-Detection-Team/object-detection-localization.git
cd object-detection-localization
```

### 2. Create Virtual Environment
```bash
# macOS / Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run Application
```bash
streamlit run main.py
```
> The app will open automatically at `http://localhost:8501`

---

## 🏆 Model Benchmarks

We rigorously trained and tested three architectures on the Fruit Classification dataset (9 classes).

| Architecture | Role | mAP50 | Recall | Inference Speed |
| :--- | :--- | :---: | :---: | :---: |
| **YOLOv11 Medium** | 🎯 **High Accuracy** | **77.8%** | **75.5%** | 12.8 ms |
| **YOLOv8 Medium** | ⚖️ **Balanced** | 76.5% | 74.8% | 11.6 ms |
| **YOLOv5 Medium** | ⚡ **Fastest** | 76.7% | 72.1% | **10.4 ms** |

---

## 👥 Development Team

Built with ❤️ by **Team 6** (Ataturk University - Computer Vision Lab):

*   **👨‍💻 Abdelrahman MOHAMED** - YOLOv5 Specialist & Web Dev.
*   **👨‍🔬 Ramazan YILDIZ** - Project Planning & AI Research
*   **👩‍💼 Beyza GULER** - YOLOv11 Specialist & Reporting

---

## 📝 License

This project is an academic research initiative.
*   ✅ **Allowed**: Academic use, Personal learning.
*   ❌ **Restricted**: Commercial use without explicit permission.

© 2025 Object Detection Team. All Rights Reserved.
