🎯 YOLOv8 AI Vision System
📋 Project Overview
YOLOv8 Real-Time Object Detection System is a cutting-edge computer vision application that implements state-of-the-art object detection using the YOLOv8 architecture. This system provides real-time object localization and classification through an intuitive web interface, delivering approximately 30 FPS performance on modern hardware.

✨ Key Features
🔍 Real-Time Detection
80+ Object Classes: Detect people, vehicles, animals, furniture, electronics, and everyday objects

High Performance: ~30 FPS on modern hardware with optimized inference

Live Camera Feed: WebRTC-based streaming with minimal latency

Bounding Boxes: Color-coded visualizations with confidence scores

🖥️ Web Interface
Dark Theme UI: Professional dark interface with gold accents for reduced eye strain

Three Main Pages:

📊 Project Overview: Complete project documentation and methodology

🎥 Live Detection: Real-time camera feed with object detection

📈 Analytics Dashboard: Performance metrics and detection statistics

Responsive Design: Works seamlessly on desktop and mobile browsers

⚡ Technical Excellence
Optimized Model: YOLOv8n with fused layers for faster inference

Multi-Device Support: Automatic hardware detection (Apple Silicon MPS, NVIDIA CUDA, CPU)

Efficient Streaming: WebRTC implementation for real-time video processing

Performance Metrics: Built-in analytics and system monitoring

🛠️ Technology Stack
Core Framework
Python 3.8+: Primary programming language

YOLOv8 (Ultralytics): State-of-the-art object detection model

PyTorch: Deep learning framework backend

Web Interface
Streamlit: Interactive web application framework

WebRTC: Real-time communication for camera streaming

OpenCV: Image processing and computer vision operations

Supporting Libraries
Pandas & NumPy: Data processing and analytics

PyAV: Video frame processing

Streamlit-WebRTC: WebRTC integration for Streamlit

🚀 Installation & Setup
Prerequisites
bash
Python 3.8 or higher
pip (Python package manager)
Webcam or camera device
Modern web browser (Chrome, Edge recommended)
Step 1: Clone the Repository
bash
git clone https://github.com/team6-ataturk/yolov8-ai-vision.git
cd yolov8-ai-vision
Step 2: Create Virtual Environment
bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
Step 3: Install Dependencies
bash
pip install -r requirements.txt
Step 4: Download YOLOv8 Model
The application automatically downloads the YOLOv8n model (6.2 MB) on first run. Manual download:

bash
# Optional: Manual model download
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
Step 5: Run the Application
bash
streamlit run main.py
Step 6: Access the Application
Open your browser and navigate to:

text
http://localhost:8501
📁 Project Structure
text
yolov8-ai-vision/
├── main.py                 # Main application file
├── requirements.txt        # Python dependencies
├── README.md              # Project documentation
├── assets/                # Images and resources
│   ├── screenshots/
│   └── diagrams/
└── models/                # Model storage (auto-created)
    └── yolov8n.pt
🎮 How to Use
1. Project Overview Page
Learn about the project methodology and technology stack

View development team information

Check project roadmap and progress

Navigate to live detection with one click

2. Live Detection Page
Grant Camera Permissions when prompted by your browser

Click "START" to activate the camera feed

Position Objects 1-2 meters from the camera

View Real-Time Detection with bounding boxes and labels

Check Tips Section for optimal results

3. Analytics Dashboard
Monitor detection performance metrics

View frequency of detected object classes

Track confidence scores and processing times

Access historical statistics (simulated data)

⚙️ System Requirements
Minimum Requirements
Processor: Intel i5 or equivalent (4 cores)

RAM: 8 GB minimum, 16 GB recommended

Storage: 2 GB free space

Webcam: 720p resolution or higher

OS: Windows 10+, macOS 10.14+, or Linux Ubuntu 18.04+

Optimal Performance
Processor: Apple Silicon M1/M2/M3 or Intel i7/Ryzen 7

RAM: 16 GB or higher

GPU: NVIDIA CUDA-capable GPU (for CUDA acceleration)

Camera: 1080p with good lighting conditions

🔧 Troubleshooting
Common Issues & Solutions
Camera Not Working
text
Issue: "Camera Error" or "No video feed"
Solution:
1. Check browser permissions for camera access
2. Ensure no other application is using the camera
3. Try refreshing the page
4. Use Chrome or Edge browser for best compatibility
Model Loading Slow
text
Issue: Long loading time on first run
Solution:
1. Wait for initial model download (6.2 MB)
2. Ensure stable internet connection
3. Check if antivirus is blocking downloads
Low FPS Performance
text
Issue: Detection appears slow or choppy
Solution:
1. Reduce camera resolution to 720p
2. Close other applications using the camera
3. Ensure good lighting conditions
4. Use hardware-accelerated browser
WebRTC Connection Issues
text
Issue: "WebRTC not supported" error
Solution:
1. Update browser to latest version
2. Enable WebRTC in browser settings
3. Check firewall/network restrictions
4. Try different browser
📊 Methodology
Dataset
Source: Frute-262 dataset

Selected Classes: 12 fruit classes

Samples: 100 images per class (1,200 total)

Model Architecture
Base Model: YOLOv8 Nano (YOLOv8n)

Pre-trained Weights: COCO dataset (80+ classes)

Model Size: 6.2 MB

Inference Speed: ~30 FPS on modern hardware

Deployment Pipeline
Data Sampling: Selected 1,200 images from Frute-262

Model Selection: YOLOv8n for optimal speed/accuracy balance

Interface Development: Streamlit web application

Real-time Integration: WebRTC for camera streaming

Performance Optimization: Model fusion and hardware acceleration

🗺️ Project Roadmap
✅ Phase 1: Completed
Dataset sampling (1,200 images, 12 classes)

Prototype system deployment

Real-time detection implementation

Streamlit web interface development

Performance optimization (30+ FPS)

Analytics dashboard integration

🔄 Phase 2: In Progress
Image annotation and labeling

Custom model fine-tuning on fruit dataset

Quantitative evaluation (mAP, IoU metrics)

Advanced analytics and reporting

Multi-camera support

📅 Phase 3: Future Development
Mobile application version

Cloud deployment options

API for third-party integration

Additional object classes (custom training)

Real-time video file processing

👥 Development Team
🎯 Project Leads
<span style="color:#F5C453; font-weight:bold;">Team 6 - Computer Vision Lab</span>

Ataturk University, Department of Computer Engineering

Member	Role	Emoji	Contribution
Abdelrahman MOHAMED	Lead Developer	👨‍💻	System architecture, backend development, model integration
Ramazan YILDIZ	Computer Vision Engineer	👨‍🔬	YOLOv8 implementation, performance optimization, algorithm design
Beyza GULER	Data Specialist	👩‍💼	Dataset preparation, analytics, documentation, UI/UX design
🤝 Acknowledgments
Ataturk University for research facilities and support

Ultralytics for the YOLOv8 framework

Streamlit for the amazing web application framework

OpenCV & PyTorch communities for continuous improvements

📝 License
This project is developed for academic research purposes at Ataturk University. All rights reserved by the development team.

Usage Rights
✅ Academic Use: Free for educational and research purposes

✅ Personal Projects: Can be used as reference for learning

❌ Commercial Use: Requires permission from the development team

❌ Redistribution: Not allowed without proper attribution

📞 Contact & Support
Technical Support
For technical issues or questions:

Email: abdoessammo@gmail.com

Repository: https://github.com/Object-Detection-Team/object-detection-localization/tree/main

Academic Inquiries
For research collaborations or academic inquiries:

Department: Computer Engineering, Ataturk University

Location: Erzurum, Turkey

Lab: Computer Vision Research Laboratory

🌟 Star History
If you find this project useful, please consider giving it a star on GitHub! ⭐

<div align="center">
🎯 Built with passion by Team 6 | Ataturk University

"Advancing Computer Vision Research"

📅 Last Updated: December 2025

</div>
