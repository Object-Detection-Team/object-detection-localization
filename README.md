<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
</head>

<body>

<h1>🎯 YOLOv8 AI Vision System</h1>

<h2>📋 Project Overview</h2>
<p>
The <b>YOLOv8 Real-Time Object Detection System</b> implements state-of-the-art object detection using the YOLOv8 architecture.
It delivers real-time localization and classification with an intuitive web interface, reaching <b>~30 FPS</b> on modern hardware.
</p>

<hr>

<h2>✨ Key Features</h2>

<h3>🔍 Real-Time Detection</h3>
<ul>
    <li>80+ object classes</li>
    <li>~30 FPS performance</li>
    <li>WebRTC live camera feed</li>
    <li>Color-coded bounding boxes with confidence scores</li>
</ul>

<h3>🖥️ Web Interface</h3>
<ul>
    <li>Dark theme UI with gold accents</li>
    <li>Three pages: Project Overview, Live Detection, Analytics Dashboard</li>
    <li>Fully responsive layout</li>
</ul>

<h3>⚡ Technical Excellence</h3>
<ul>
    <li>YOLOv8n optimized model (6.2 MB)</li>
    <li>Automatic hardware detection (CUDA, MPS, CPU)</li>
    <li>WebRTC low-latency video</li>
    <li>Built-in performance analytics</li>
</ul>

<hr>

<h2>🛠️ Technology Stack</h2>

<h3>Core</h3>
<ul>
    <li>Python 3.8+</li>
    <li>YOLOv8 (Ultralytics)</li>
    <li>PyTorch</li>
</ul>

<h3>Web Interface</h3>
<ul>
    <li>Streamlit</li>
    <li>Streamlit-WebRTC</li>
    <li>OpenCV</li>
</ul>

<h3>Supporting Libraries</h3>
<ul>
    <li>NumPy, Pandas</li>
    <li>PyAV</li>
</ul>

<hr>

<h2>🚀 Installation & Setup</h2>

<h3>Step 1 Clone repository</h3>
<pre><code>git clone https://github.com/Object-Detection-Team/object-detection-localization.git
cd object-detection-localization
</code></pre>

<h3>Step 2 Create a virtual environment</h3>
<pre><code># Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
</code></pre>

<h3>Step 3 Install dependencies</h3>
<pre><code>pip install -r requirements.txt</code></pre>

<h3>Step 4 Download YOLOv8 model</h3>
<pre><code>python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')" </code></pre>

<h3>Step 5 Run the application</h3>
<pre><code>streamlit run main.py</code></pre>

<h3>Step 6 Open the app</h3>
<pre><code>http://localhost:8501</code></pre>

<hr>

<h2>📁 Project Structure</h2>
<pre><code>object-detection-localization/
├── main.py
├── requirements.txt
├── README.md
├── assets/
│   ├── screenshots/
│   └── diagrams/
└── models/
    └── yolov8n.pt
</code></pre>

<hr>

<h2>🎮 How to Use</h2>

<h3>Project Overview page</h3>
<ul>
    <li>View methodology</li>
    <li>Check stack and roadmap</li>
</ul>

<h3>Live Detection page</h3>
<ul>
    <li>Allow browser camera access</li>
    <li>Click START</li>
    <li>Best distance 1–2 meters</li>
</ul>

<h3>Analytics Dashboard</h3>
<ul>
    <li>FPS tracking</li>
    <li>Detection frequency</li>
    <li>Confidence distribution</li>
</ul>

<hr>

<h2>⚙️ System Requirements</h2>

<h3>Minimum</h3>
<ul>
    <li>Intel i5 / 8GB RAM</li>
    <li>720p webcam</li>
    <li>Windows 10+, macOS 10.14+, Ubuntu 18.04+</li>
</ul>

<h3>Optimal</h3>
<ul>
    <li>Apple Silicon or Intel i7 / Ryzen 7</li>
    <li>16GB RAM</li>
    <li>CUDA GPU</li>
    <li>1080p camera</li>
</ul>

<hr>

<h2>🔧 Troubleshooting</h2>

<h3>Camera Not Working</h3>
<ul>
    <li>Enable permissions</li>
    <li>Close other camera apps</li>
    <li>Refresh page</li>
    <li>Use Chrome or Edge</li>
</ul>

<h3>Low FPS</h3>
<ul>
    <li>Use 720p resolution</li>
    <li>Improve lighting</li>
    <li>Close heavy applications</li>
</ul>

<hr>

<h2>📊 Methodology</h2>

<h3>Dataset</h3>
<ul>
    <li>Frute-262 dataset</li>
    <li>12 classes</li>
    <li>1,200 images</li>
</ul>

<h3>Model Architecture</h3>
<ul>
    <li>YOLOv8 Nano (YOLOv8n)</li>
    <li>COCO pre-trained weights</li>
    <li>6.2 MB size</li>
    <li>~30 FPS inference</li>
</ul>

<h3>Pipeline</h3>
<ul>
    <li>Dataset sampling</li>
    <li>Model selection</li>
    <li>Streamlit UI</li>
    <li>WebRTC integration</li>
    <li>Performance optimization</li>
</ul>

<hr>

<h2>🗺️ Project Roadmap</h2>

<h3>Phase 1 Completed</h3>
<ul>
    <li>Dataset sampling</li>
    <li>Real-time detection prototype</li>
    <li>Streamlit UI</li>
    <li>Performance optimization</li>
    <li>Analytics dashboard</li>
</ul>

<h3>Phase 2 In Progress</h3>
<ul>
    <li>Annotation & fine-tuning</li>
    <li>mAP & IoU metrics</li>
    <li>Advanced analytics</li>
    <li>Multi-camera support</li>
</ul>

<h3>Phase 3 Future</h3>
<ul>
    <li>Mobile app</li>
    <li>Cloud deployment</li>
    <li>API development</li>
    <li>Custom object classes</li>
</ul>

<hr>

<h2>👥 Development Team</h2>

<table>
<tr>
    <th>Member</th>
    <th>Role</th>
    <th>Emoji</th>
    <th>Contribution</th>
</tr>
<tr>
    <td><b>Abdelrahman MOHAMED</b></td>
    <td>Lead Developer</td>
    <td>👨‍💻</td>
    <td>System architecture, backend development, model integration</td>
</tr>
<tr>
    <td><b>Ramazan YILDIZ</b></td>
    <td>Computer Vision Engineer</td>
    <td>👨‍🔬</td>
    <td>YOLOv8 implementation, optimization</td>
</tr>
<tr>
    <td><b>Beyza GULER</b></td>
    <td>Data Specialist</td>
    <td>👩‍💼</td>
    <td>Dataset, analytics, documentation, UI/UX</td>
</tr>
</table>

<hr>

<h2>📝 License</h2>
<p>
Academic research project for Ataturk University.<br>
All rights reserved.
</p>

<ul>
    <li>✔ Academic use allowed</li>
    <li>✔ Personal learning allowed</li>
    <li>✘ Commercial use requires permission</li>
    <li>✘ Redistribution not allowed</li>
</ul>

<hr>

<h2>📞 Contact & Support</h2>

<p>
<b>Email:</b> <a href="mailto:abdoessammo@gmail.com">abdoessammo@gmail.com</a><br>
<b>Repository:</b> <a href="https://github.com/Object-Detection-Team/object-detection-localization">GitHub Project</a>
</p>

<hr>

<div class="center">
🎯 Built with passion by <b>Team 6</b><br>
Ataturk University – Computer Vision Lab<br>
“Advancing Computer Vision Research”<br>
📅 Last Updated: December 2025
</div>

</body>
</html>
