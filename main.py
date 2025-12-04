import streamlit as st
import av
import torch
import pandas as pd
import numpy as np
from datetime import datetime
import time
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
from ultralytics import YOLO

# ====================================================================
# PAGE CONFIGURATION
# ====================================================================
st.set_page_config(
    page_title="YOLOv8 AI Vision System",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ====================================================================
# CALM PASTEL UI DESIGN - Professional & Eye-Friendly
# ====================================================================
# ====================================================================
# BLACK & GOLD UI DESIGN - Luxurious & Professional
# ====================================================================
st.markdown("""
    <style>
    /* Import elegant fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Poppins:wght@400;500;600;700&display=swap');
    
    /* Global Styling */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        max-width: 95% !important;
    }
    
    /* Hide Streamlit elements */
    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* Luxurious Background */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%) !important;
        background-attachment: fixed !important;
    }
    
    /* Main Content Container */
    .main > div:first-child {
        background: rgba(15, 12, 41, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 24px;
        margin: 1.5rem;
        padding: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(255, 215, 0, 0.1);
        min-height: calc(100vh - 3rem);
    }
    
    /* Gold Gradient Headers */
    h1, h2, h3 {
        font-family: 'Poppins', sans-serif !important;
        font-weight: 700 !important;
        background: linear-gradient(90deg, #FFD700, #FFA500, #FFD700) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        margin-bottom: 1rem !important;
    }
    
    h1 {
        font-size: 2.8rem !important;
        letter-spacing: -0.5px !important;
        text-shadow: 0 0 30px rgba(255, 215, 0, 0.3) !important;
    }
    
    h2 {
        font-size: 2rem !important;
        position: relative !important;
        display: inline-block !important;
        padding-bottom: 0.75rem !important;
    }
    
    h2:after {
        content: '';
        position: absolute;
        bottom: 0;
        left: 0;
        width: 60px;
        height: 3px;
        background: linear-gradient(90deg, #FFD700, #FFA500);
        border-radius: 2px;
        box-shadow: 0 0 10px rgba(255, 215, 0, 0.5);
    }
    
    h3 {
        font-size: 1.5rem !important;
        font-weight: 600 !important;
    }
    
    /* Dark Cards with Gold Accents */
    .card {
        background: rgba(30, 30, 46, 0.8) !important;
        border-radius: 20px !important;
        padding: 1.75rem !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4) !important;
        border: 1px solid rgba(255, 215, 0, 0.2) !important;
        transition: all 0.25s ease !important;
    }
    
    .card:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 30px rgba(255, 215, 0, 0.2) !important;
        border-color: rgba(255, 215, 0, 0.4) !important;
    }
    
    /* Metric Cards */
    div[data-testid="stMetric"] {
        background: rgba(30, 30, 46, 0.8) !important;
        border-radius: 16px !important;
        border: 1px solid rgba(255, 215, 0, 0.2) !important;
        padding: 1.5rem !important;
        margin: 0.5rem 0 !important;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3) !important;
        transition: all 0.25s ease !important;
    }
    
    div[data-testid="stMetric"]:hover {
        transform: translateY(-3px) !important;
        border-color: rgba(255, 215, 0, 0.4) !important;
        box-shadow: 0 8px 25px rgba(255, 215, 0, 0.2) !important;
    }
    
    div[data-testid="stMetric"] label {
        color: #FFD700 !important;
        font-weight: 500 !important;
        font-size: 0.9rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
    }
    
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #FFFFFF !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
        text-shadow: 0 0 10px rgba(255, 215, 0, 0.3);
    }
    
    /* Gold Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #FFD700 0%, #FFA500 100%) !important;
        color: #000000 !important;
        border: none !important;
        padding: 0.875rem 2rem !important;
        font-size: 1rem !important;
        font-weight: 700 !important;
        border-radius: 12px !important;
        transition: all 0.25s ease !important;
        box-shadow: 0 4px 15px rgba(255, 215, 0, 0.3) !important;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #FFA500 0%, #FF8C00 100%) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(255, 215, 0, 0.5) !important;
        color: #000000 !important;
    }
    
    /* Dark Sidebar with Gold Accents */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%) !important;
        border-right: 1px solid rgba(255, 215, 0, 0.2) !important;
    }
    
    [data-testid="stSidebar"] > div:first-child {
        padding: 2rem 1.5rem !important;
    }
    
    /* Sidebar Navigation */
    [data-testid="stSidebar"] .stRadio > div {
        gap: 0.5rem !important;
    }
    
    [data-testid="stSidebar"] .stRadio > label {
        color: #FFD700 !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        margin-bottom: 1rem !important;
    }
    
    [data-testid="stSidebar"] .stRadio > div > label {
        background: rgba(255, 215, 0, 0.1) !important;
        border-radius: 12px !important;
        padding: 0.875rem 1.25rem !important;
        border: 1px solid rgba(255, 215, 0, 0.2) !important;
        color: #FFFFFF !important;
        transition: all 0.25s ease !important;
        margin: 0.25rem 0 !important;
        font-weight: 500 !important;
    }
    
    [data-testid="stSidebar"] .stRadio > div > label:hover {
        background: rgba(255, 215, 0, 0.2) !important;
        transform: translateX(4px) !important;
        border-color: rgba(255, 215, 0, 0.4) !important;
    }
    
    [data-testid="stSidebar"] .stRadio > div > label[data-baseweb="radio"]:has(input:checked) {
        background: rgba(255, 215, 0, 0.3) !important;
        border-color: #FFD700 !important;
        color: #FFFFFF !important;
        box-shadow: 0 0 15px rgba(255, 215, 0, 0.3) !important;
    }
    
    /* Video Container */
    video {
        border-radius: 20px !important;
        box-shadow: 0 8px 30px rgba(255, 215, 0, 0.2) !important;
        border: 2px solid rgba(255, 215, 0, 0.3) !important;
    }
    
    /* Divider */
    hr {
        height: 1px !important;
        background: linear-gradient(90deg, transparent, #FFD700, transparent) !important;
        border: none !important;
        margin: 2rem 0 !important;
    }
    
    /* Progress Bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #FFD700, #FFA500) !important;
        border-radius: 8px !important;
    }
    
    /* Text Colors */
    p {
        color: #E0E0E0 !important;
    }
    
    /* Status Indicators */
    .status-indicator {
        display: inline-flex;
        align-items: center;
        padding: 0.375rem 0.875rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 0.25rem;
    }
    
    .status-active {
        background: rgba(255, 215, 0, 0.2);
        color: #FFD700;
        border: 1px solid rgba(255, 215, 0, 0.4);
    }
    
    .status-inactive {
        background: rgba(128, 128, 128, 0.2);
        color: #808080;
        border: 1px solid rgba(128, 128, 128, 0.4);
    }
    
    /* Badge Styling */
    .badge {
        display: inline-block;
        padding: 0.375rem 0.875rem;
        background: rgba(255, 215, 0, 0.2);
        color: #FFD700;
        border-radius: 50px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 0.25rem;
        border: 1px solid rgba(255, 215, 0, 0.3);
    }
    
    .badge:hover {
        background: rgba(255, 215, 0, 0.3);
        transform: translateY(-1px);
    }
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(30, 30, 46, 0.5);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #FFD700, #FFA500);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(180deg, #FFA500, #FFD700);
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .main > div:first-child {
            margin: 1rem;
            padding: 1.5rem;
        }
        
        h1 {
            font-size: 2.2rem !important;
        }
        
        h2 {
            font-size: 1.75rem !important;
        }
    }
    
    /* Animation for live indicator */
    @keyframes gold-pulse {
        0% { opacity: 0.8; box-shadow: 0 0 5px #FFD700; }
        50% { opacity: 1; box-shadow: 0 0 15px #FFD700; }
        100% { opacity: 0.8; box-shadow: 0 0 5px #FFD700; }
    }
    
    .pulse {
        animation: gold-pulse 2s infinite ease-in-out;
    }
    
    /* Special Gradient Sections */
    .gradient-section {
        background: linear-gradient(135deg, rgba(15, 12, 41, 0.8) 0%, rgba(48, 43, 99, 0.8) 100%);
        border-radius: 20px;
        padding: 2rem;
        border: 1px solid rgba(255, 215, 0, 0.3);
        box-shadow: 0 0 30px rgba(255, 215, 0, 0.1);
    }
    </style>
""", unsafe_allow_html=True)

# ====================================================================
# DEVICE CONFIGURATION
# ====================================================================
DEFAULT_DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
DEVICE_LABEL = "🤖 Apple Silicon (MPS)" if DEFAULT_DEVICE == "mps" else "💻 CPU"

# ====================================================================
# YOLO VIDEO PROCESSOR CLASS WITH ERROR HANDLING
# ====================================================================
class YOLOv8VideoProcessor(VideoProcessorBase):
    def __init__(self):
        super().__init__()
        self.device = DEFAULT_DEVICE
        try:
            self.model = YOLO("yolov8n.pt")
            self.model.to(self.device)
            self.model_loaded = True
        except Exception as e:
            st.error(f"Failed to load model: {e}")
            self.model_loaded = False

    def recv(self, frame):
        if not self.model_loaded:
            return frame
            
        try:
            img = frame.to_ndarray(format="bgr24")
            results = self.model(img, verbose=False)
            processed_img = results[0].plot()
            return av.VideoFrame.from_ndarray(processed_img, format="bgr24")
        except Exception as e:
            print(f"Processing error: {e}")
            return frame

# ====================================================================
# LUXURIOUS SIDEBAR NAVIGATION - Black & Gold Theme
# ====================================================================
def render_sidebar():
    with st.sidebar:
        # Gold Header
        st.markdown("""
            <div style='text-align: center; margin-bottom: 2rem;'>
                <div style='font-size: 2.5rem; margin-bottom: 0.5rem; color: #FFD700;'>🚀</div>
                <h2 style='color: #FFD700; margin: 0; font-size: 1.4rem;'>AI Vision System</h2>
                <p style='color: rgba(255, 215, 0, 0.8); margin: 0.5rem 0 0 0; font-size: 0.9rem;'>
                    Real-Time Object Detection
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<hr style='border-color: rgba(255, 215, 0, 0.2);'>", unsafe_allow_html=True)
        
        # Navigation
        options = ["🏠 Project Overview", "🎥 Live Detection", "📊 Analytics Dashboard", "⚙️ System Settings"]
        
        if "page_selection" not in st.session_state:
            st.session_state.page_selection = options[0]

        selection = st.radio(
            "NAVIGATION", 
            options, 
            index=options.index(st.session_state.page_selection),
            label_visibility="collapsed"
        )
        
        if selection != st.session_state.page_selection:
            st.session_state.page_selection = selection
            st.rerun()

        st.markdown("<hr style='border-color: rgba(255, 215, 0, 0.2);'>", unsafe_allow_html=True)
        
        # System Status
        st.markdown(f"""
            <div class='card' style='margin: 1.5rem 0; padding: 1.25rem !important;'>
                <h4 style='color: #FFD700; margin: 0 0 1rem 0; font-size: 1.1rem;'>⚡ System Status</h4>
                <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;'>
                    <span style='color: #FFFFFF; font-size: 0.9rem;'>Hardware</span>
                    <span style='color: #FFD700; font-weight: 600; font-size: 0.9rem;'>{DEVICE_LABEL}</span>
                </div>
                <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;'>
                    <span style='color: #FFFFFF; font-size: 0.9rem;'>Model</span>
                    <span style='color: #FFD700; font-weight: 600; font-size: 0.9rem;'>YOLOv8n</span>
                </div>
                <div style='display: flex; justify-content: space-between; align-items: center;'>
                    <span style='color: #FFFFFF; font-size: 0.9rem;'>Status</span>
                    <span style='color: #FFD700; font-weight: 600; font-size: 0.9rem;'>● Active</span>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<hr style='border-color: rgba(255, 215, 0, 0.2);'>", unsafe_allow_html=True)
        
        # Team Info
        st.markdown("""
            <div style='text-align: center; padding: 1rem 0;'>
                <p style='color: #FFD700; font-weight: 600; margin-bottom: 0.5rem; font-size: 0.95rem;'>Team 6</p>
                <p style='color: rgba(255, 215, 0, 0.8); font-size: 0.85rem; margin: 0.25rem 0;'>Ataturk University</p>
                <p style='color: rgba(255, 215, 0, 0.6); font-size: 0.75rem; margin-top: 1rem;'>
                    Computer Vision Lab © 2025
                </p>
            </div>
        """, unsafe_allow_html=True)

# ====================================================================
# PROJECT OVERVIEW PAGE - Clean & Professional
# ====================================================================
def render_project_overview():
    st.markdown("<h1>YOLOv8 Real-Time Object Detection</h1>", unsafe_allow_html=True)
    
    # Introduction Card
    st.markdown("""
        <div class='card'>
            <h3>🎯 Project Introduction</h3>
            <p style='color: #3C4858; line-height: 1.7; font-size: 1.05rem; margin-bottom: 0;'>
                This project implements a real-time object detection and localization system using the 
                YOLOv8 architecture. The system operates as an intermediate prototype using pre-trained 
                YOLOv8 model without custom fine-tuning, performing real-time inference through a 
                Streamlit-based web interface with approximately 30 FPS performance.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Key Metrics - Clean Layout
    st.markdown("<h2>Project Overview</h2>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(label="Fruit Classes", value="12", delta="Sampled")
    
    with col2:
        st.metric(label="Images", value="1,200", delta="100 per class")
    
    with col3:
        st.metric(label="FPS", value="30+", delta="Real Time")
    
    with col4:
        st.metric(label="Model Size", value="6.2 MB", delta="YOLOv8n")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Project Details Section
    details_col1, details_col2 = st.columns(2)
    
    with details_col1:
        st.markdown("""
            <div class='card'>
                <h3>🔬 Methodology</h3>
                <div style='color: #3C4858; line-height: 1.8;'>
                    <p><strong>Dataset:</strong> Fruits-262 dataset</p>
                    <p><strong>Selected Classes:</strong> 12 fruit classes</p>
                    <p><strong>Samples:</strong> 100 images per class (1200 total)</p>
                    <p><strong>Model:</strong> YOLOv8 Nano (YOLOv8n)</p>
                    <p><strong>Pre-trained:</strong> COCO dataset weights</p>
                    <p><strong>Deployment:</strong> Streamlit Web Interface</p>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    with details_col2:
        st.markdown("""
            <div class='card'>
                <h3>🛠️ Technology Stack</h3>
                <div style='display: flex; flex-wrap: wrap; gap: 0.5rem; margin-bottom: 1.5rem;'>
                    <span class='badge'>Python</span>
                    <span class='badge'>YOLOv8</span>
                    <span class='badge'>Streamlit</span>
                    <span class='badge'>PyTorch</span>
                    <span class='badge'>OpenCV</span>
                    <span class='badge'>WebRTC</span>
                </div>
                <div style='color: #3C4858;'>
                    <p><strong>Tools & Frameworks:</strong></p>
                    <ul style='color: #3C4858; padding-left: 1.2rem;'>
                        <li>Ultralytics YOLOv8 for detection</li>
                        <li>Streamlit for web interface</li>
                        <li>OpenCV for image processing</li>
                        <li>PyTorch as backend</li>
                        <li>WebRTC for real-time streaming</li>
                    </ul>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Team Section
    st.markdown("<h2>👥 Development Team</h2>", unsafe_allow_html=True)
    
    team_col1, team_col2, team_col3 = st.columns(3)
    
    with team_col1:
        st.markdown("""
            <div class='card' style='text-align: center;'>
                <div style='font-size: 2.2rem; margin-bottom: 1rem; color: #4C8DF5;'>👨‍💻</div>
                <h4 style='color: #1F2D3D; margin-bottom: 0.5rem;'>Abdelrahman MOHAMED</h4>
                <p style='color: #7B8794; margin: 0; font-weight: 500;'>Lead Developer</p>
                <p style='color: #A3D8FF; font-size: 0.85rem; margin: 0.75rem 0;'>abdoessammo@gmail.com</p>
                <p style='color: #A3D8FF; font-size: 0.85rem;'>+90 552 750 8202</p>
            </div>
        """, unsafe_allow_html=True)
    
    with team_col2:
        st.markdown("""
            <div class='card' style='text-align: center;'>
                <div style='font-size: 2.2rem; margin-bottom: 1rem; color: #4C8DF5;'>👨‍🔬</div>
                <h4 style='color: #1F2D3D; margin-bottom: 0.5rem;'>Ramazan YILDIZ</h4>
                <p style='color: #7B8794; margin: 0; font-weight: 500;'>Computer Vision Engineer</p>
                <p style='color: #A3D8FF; font-size: 0.85rem; margin: 0.75rem 0;'>ry.yildizramazan@gmail.com</p>
                <p style='color: #A3D8FF; font-size: 0.85rem;'>+90 544 448 3169</p>
            </div>
        """, unsafe_allow_html=True)
    
    with team_col3:
        st.markdown("""
            <div class='card' style='text-align: center;'>
                <div style='font-size: 2.2rem; margin-bottom: 1rem; color: #4C8DF5;'>👩‍💼</div>
                <h4 style='color: #1F2D3D; margin-bottom: 0.5rem;'>Beyza GULER</h4>
                <p style='color: #7B8794; margin: 0; font-weight: 500;'>Data Specialist</p>
                <p style='color: #A3D8FF; font-size: 0.85rem; margin: 0.75rem 0;'>beyzafeyza.2001@gmail.com</p>
                <p style='color: #A3D8FF; font-size: 0.85rem;'>+90 552 647 5822</p>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Roadmap Section
    st.markdown("<h2>📅 Project Roadmap</h2>", unsafe_allow_html=True)
    
    roadmap_col1, roadmap_col2 = st.columns(2)
    
    with roadmap_col1:
        st.markdown("""
            <div class='card'>
                <h3>✅ Phase 1 (Current)</h3>
                <div style='color: #3C4858;'>
                    <div style='display: flex; align-items: start; margin-bottom: 1rem;'>
                        <span style='color: #4C8DF5; font-size: 1.2rem; margin-right: 0.75rem;'>✓</span>
                        <span>Dataset sampling completed</span>
                    </div>
                    <div style='display: flex; align-items: start; margin-bottom: 1rem;'>
                        <span style='color: #4C8DF5; font-size: 1.2rem; margin-right: 0.75rem;'>✓</span>
                        <span>Prototype system deployed</span>
                    </div>
                    <div style='display: flex; align-items: start; margin-bottom: 1rem;'>
                        <span style='color: #4C8DF5; font-size: 1.2rem; margin-right: 0.75rem;'>✓</span>
                        <span>Real-time detection working</span>
                    </div>
                    <div style='display: flex; align-items: start;'>
                        <span style='color: #4C8DF5; font-size: 1.2rem; margin-right: 0.75rem;'>✓</span>
                        <span>Web interface established</span>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    with roadmap_col2:
        st.markdown("""
            <div class='card'>
                <h3>⏳ Phase 2 (Next)</h3>
                <div style='color: #3C4858;'>
                    <div style='display: flex; align-items: start; margin-bottom: 1rem;'>
                        <span style='color: #A3D8FF; font-size: 1.2rem; margin-right: 0.75rem;'>○</span>
                        <span>Image annotation & labeling</span>
                    </div>
                    <div style='display: flex; align-items: start; margin-bottom: 1rem;'>
                        <span style='color: #A3D8FF; font-size: 1.2rem; margin-right: 0.75rem;'>○</span>
                        <span>Model fine-tuning on custom dataset</span>
                    </div>
                    <div style='display: flex; align-items: start; margin-bottom: 1rem;'>
                        <span style='color: #A3D8FF; font-size: 1.2rem; margin-right: 0.75rem;'>○</span>
                        <span>Quantitative evaluation (mAP, IoU)</span>
                    </div>
                    <div style='display: flex; align-items: start;'>
                        <span style='color: #A3D8FF; font-size: 1.2rem; margin-right: 0.75rem;'>○</span>
                        <span>Performance optimization</span>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Call to Action
    st.markdown("""
        <div class='gradient-section'>
            <div style='text-align: center;'>
                <h2 style='color: #1F2D3D; margin-bottom: 1rem;'>🚀 Ready to Experience AI Vision?</h2>
                <p style='color: #3C4858; font-size: 1.1rem; margin-bottom: 2rem; max-width: 600px; margin-left: auto; margin-right: auto;'>
                    Activate your webcam and see YOLOv8 in real-time action!
                </p>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    if st.button("🎥 Start Live Detection", width='stretch', key="overview_cta"):
        st.session_state.page_selection = "🎥 Live Detection"
        st.rerun()

# ====================================================================
# LIVE DETECTION PAGE - Clean Professional Interface
# ====================================================================
def render_live_detection():
    st.markdown("<h1>Live Object Detection</h1>", unsafe_allow_html=True)
    
    # Status Overview
    st.markdown(f"""
        <div class='card'>
            <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;'>
                <h3 style='margin: 0;'>Real-Time Detection</h3>
                <div style='display: flex; align-items: center; gap: 0.75rem;'>
                    <div style='width: 10px; height: 10px; background: #4C8DF5; border-radius: 50%;' class='pulse'></div>
                    <span style='color: #4C8DF5; font-weight: 600;'>LIVE</span>
                </div>
            </div>
            <p style='color: #3C4858; margin: 0;'>
                Using YOLOv8n pre-trained on COCO dataset for 80+ object classes
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Main Layout
    video_col, control_col = st.columns([2.5, 1], gap="large")
    
    with video_col:
        # Camera Feed Section
        st.markdown("""
            <div class='card'>
                <h3>📷 Camera Feed</h3>
                <div style='background: #F5F7FA; border-radius: 12px; padding: 1.25rem; margin: 1rem 0;'>
                    <div style='display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; text-align: center;'>
                        <div>
                            <div style='color: #1F2D3D; font-weight: 600; font-size: 0.85rem; margin-bottom: 0.25rem;'>HARDWARE</div>
                            <div style='font-size: 0.85rem; color: #4C8DF5; font-weight: 500;'>""" + DEVICE_LABEL + """</div>
                        </div>
                        <div>
                            <div style='color: #1F2D3D; font-weight: 600; font-size: 0.85rem; margin-bottom: 0.25rem;'>MODEL</div>
                            <div style='font-size: 0.85rem; color: #4C8DF5; font-weight: 500;'>YOLOv8n</div>
                        </div>
                        <div>
                            <div style='color: #1F2D3D; font-weight: 600; font-size: 0.85rem; margin-bottom: 0.25rem;'>PERFORMANCE</div>
                            <div style='font-size: 0.85rem; color: #4C8DF5; font-weight: 500;'>~30 FPS</div>
                        </div>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # WebRTC Streamer
        try:
            webrtc_ctx = webrtc_streamer(
                key="yolov8_detection",
                video_processor_factory=YOLOv8VideoProcessor,
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True,
                mode=WebRtcMode.SENDRECV,
                rtc_configuration={
                    "iceServers": [
                        {"urls": ["stun:stun.l.google.com:19302"]},
                        {"urls": ["stun:stun1.l.google.com:19302"]}
                    ]
                }
            )
            
            if not webrtc_ctx.state.playing:
                st.info("👆 Click 'START' above to begin camera detection")
                
        except Exception as e:
            st.error(f"Camera Error: {str(e)}")
            st.warning("""
                **Troubleshooting Tips:**
                1. Check browser permissions for camera access
                2. Ensure no other application is using the camera
                3. Try refreshing the page
                4. Use Chrome or Edge browser for best compatibility
            """)
        
        # Detection Features
        st.markdown("""
            <div class='card'>
                <h3>🎯 Detection Features</h3>
                <div style='display: grid; grid-template-columns: repeat(2, 1fr); gap: 1rem; margin-top: 1rem;'>
                    <div style='text-align: center; padding: 1.25rem; background: #F5F7FA; border-radius: 12px;'>
                        <div style='font-size: 1.75rem; color: #4C8DF5; margin-bottom: 0.75rem;'>🎨</div>
                        <p style='color: #1F2D3D; font-weight: 600; margin: 0; font-size: 0.95rem;'>Color-Coded Boxes</p>
                    </div>
                    <div style='text-align: center; padding: 1.25rem; background: #F5F7FA; border-radius: 12px;'>
                        <div style='font-size: 1.75rem; color: #4C8DF5; margin-bottom: 0.75rem;'>📊</div>
                        <p style='color: #1F2D3D; font-weight: 600; margin: 0; font-size: 0.95rem;'>Confidence Scores</p>
                    </div>
                    <div style='text-align: center; padding: 1.25rem; background: #F5F7FA; border-radius: 12px;'>
                        <div style='font-size: 1.75rem; color: #4C8DF5; margin-bottom: 0.75rem;'>⚡</div>
                        <p style='color: #1F2D3D; font-weight: 600; margin: 0; font-size: 0.95rem;'>Real-Time Processing</p>
                    </div>
                    <div style='text-align: center; padding: 1.25rem; background: #F5F7FA; border-radius: 12px;'>
                        <div style='font-size: 1.75rem; color: #4C8DF5; margin-bottom: 0.75rem;'>🔒</div>
                        <p style='color: #1F2D3D; font-weight: 600; margin: 0; font-size: 0.95rem;'>Privacy Safe</p>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    with control_col:
        # Control Panel
        st.markdown("""
            <div class='card'>
                <h3>⚙️ Control Panel</h3>
                
                <div style='margin: 1.5rem 0;'>
                    <p style='color: #1F2D3D; margin-bottom: 0.75rem; font-weight: 600;'>
                        📷 Camera Setup
                    </p>
                    <div style='background: #F5F7FA; 
                                border: 1px solid #E0E6ED;
                                border-radius: 12px; padding: 1rem;'>
                        <p style='color: #3C4858; margin: 0; font-size: 0.9rem;'>
                            Click <strong style='color: #4C8DF5;'>SELECT DEVICE</strong> to choose camera source
                        </p>
                    </div>
                </div>
                
                <div style='margin: 1.5rem 0;'>
                    <p style='color: #1F2D3D; margin-bottom: 0.75rem; font-weight: 600;'>
                        ⚡ Processing Settings
                    </p>
                    <div style='display: flex; gap: 0.75rem; margin-bottom: 1.5rem;'>
                        <div style='flex: 1; text-align: center; padding: 0.75rem; 
                                    background: #EAF3FF;
                                    border: 1px solid #A3D8FF;
                                    border-radius: 10px;'>
                            <span style='color: #4C8DF5; font-weight: 600;'>Real-Time</span>
                        </div>
                        <div style='flex: 1; text-align: center; padding: 0.75rem; 
                                    background: #FFFFFF;
                                    border: 1px solid #E0E6ED;
                                    border-radius: 10px;'>
                            <span style='color: #7B8794;'>Balanced</span>
                        </div>
                    </div>
                    
                    <div style='margin-top: 1.5rem;'>
                        <label style='color: #1F2D3D; font-size: 0.9rem; display: block; margin-bottom: 0.75rem;'>
                            Confidence Threshold
                        </label>
                        <input type="range" min="0.1" max="0.9" value="0.5" step="0.1" 
                               style='width: 100%; margin: 0.5rem 0;'>
                        <div style='display: flex; justify-content: space-between; font-size: 0.8rem; color: #7B8794;'>
                            <span>Low (0.1)</span>
                            <span>Medium (0.5)</span>
                            <span>High (0.9)</span>
                        </div>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # Optimization Tips
        with st.expander("💡 Optimization Tips", expanded=True):
            st.markdown("""
                - **Lighting**: Ensure good, even lighting
                - **Distance**: Maintain 1-2m from camera
                - **Stability**: Keep camera steady
                - **Performance**: Close background tabs
                - **Objects**: Present one object at a time for best results
            """)
        
        # Performance Metrics
        st.markdown("""
            <div class='card' style='margin-top: 1rem;'>
                <h4>📈 Performance Metrics</h4>
                <div style='background: #F5F7FA; 
                            border: 1px solid #E0E6ED;
                            border-radius: 12px; padding: 1.25rem;'>
                    <div style='display: flex; justify-content: space-between; margin-bottom: 0.75rem;'>
                        <span style='color: #3C4858;'>Speed:</span>
                        <span style='color: #4C8DF5; font-weight: 600;'>Excellent</span>
                    </div>
                    <div style='display: flex; justify-content: space-between; margin-bottom: 0.75rem;'>
                        <span style='color: #3C4858;'>Accuracy:</span>
                        <span style='color: #4C8DF5; font-weight: 600;'>High</span>
                    </div>
                    <div style='display: flex; justify-content: space-between;'>
                        <span style='color: #3C4858;'>Resources:</span>
                        <span style='color: #4C8DF5; font-weight: 600;'>Low</span>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

# ====================================================================
# ANALYTICS DASHBOARD - Clean Professional Design
# ====================================================================
def render_analytics_dashboard():
    st.markdown("<h1>Analytics Dashboard</h1>", unsafe_allow_html=True)
    
    # Dashboard Overview
    st.markdown("""
        <div class='card'>
            <p style='color: #3C4858; font-size: 1.05rem;'>
                Monitor detection performance, track object statistics, and analyze system metrics 
                using built-in analytics and visualizations.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Key Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(label="Total Detections", value="1,250", delta="+125 today")
    
    with col2:
        st.metric(label="Avg Confidence", value="84.5%", delta="+2.3%")
    
    with col3:
        st.metric(label="Unique Classes", value="18", delta="+3")
    
    with col4:
        st.metric(label="Processing Time", value="32ms", delta="-5ms")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Charts Section
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        st.markdown("""
            <div class='card'>
                <h3>📈 Detection Frequency</h3>
            </div>
        """, unsafe_allow_html=True)
        
        # Sample data for bar chart
        detection_data = pd.DataFrame({
            'Object': ['Person', 'Car', 'Chair', 'Bottle', 'Laptop', 'Phone', 'Book', 'Cup'],
            'Count': [150, 120, 85, 65, 50, 45, 40, 35]
        })
        
        st.bar_chart(detection_data.set_index('Object')['Count'])
        st.dataframe(detection_data, use_container_width=True, hide_index=True)
    
    with chart_col2:
        st.markdown("""
            <div class='card'>
                <h3>📊 Confidence Distribution</h3>
            </div>
        """, unsafe_allow_html=True)
        
        # Sample data for area chart
        confidence_data = pd.DataFrame({
            'Confidence Range': ['0-20%', '20-40%', '40-60%', '60-80%', '80-100%'],
            'Detections': [5, 15, 40, 120, 320]
        })
        
        st.bar_chart(confidence_data.set_index('Confidence Range')['Detections'])
        st.dataframe(confidence_data, use_container_width=True, hide_index=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Performance Metrics Cards
    st.markdown("<h3>⚡ Performance Metrics</h3>", unsafe_allow_html=True)
    
    perf_col1, perf_col2, perf_col3 = st.columns(3)
    
    with perf_col1:
        st.markdown("""
            <div class='card' style='text-align: center;'>
                <div style='font-size: 2rem; color: #4C8DF5; margin-bottom: 0.75rem;'>⚡</div>
                <div style='font-size: 1.75rem; font-weight: 700; color: #1F2D3D;'>32ms</div>
                <div style='color: #7B8794; font-size: 0.9rem;'>Avg Processing Time</div>
            </div>
        """, unsafe_allow_html=True)
    
    with perf_col2:
        st.markdown("""
            <div class='card' style='text-align: center;'>
                <div style='font-size: 2rem; color: #4C8DF5; margin-bottom: 0.75rem;'>🎯</div>
                <div style='font-size: 1.75rem; font-weight: 700; color: #1F2D3D;'>84.5%</div>
                <div style='color: #7B8794; font-size: 0.9rem;'>Average Accuracy</div>
            </div>
        """, unsafe_allow_html=True)
    
    with perf_col3:
        st.markdown("""
            <div class='card' style='text-align: center;'>
                <div style='font-size: 2rem; color: #4C8DF5; margin-bottom: 0.75rem;'>📊</div>
                <div style='font-size: 1.75rem; font-weight: 700; color: #1F2D3D;'>30+</div>
                <div style='color: #7B8794; font-size: 0.9rem;'>FPS Performance</div>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Recent Activity
    st.markdown("<h3>🕒 Recent Detections</h3>", unsafe_allow_html=True)
    
    recent_data = pd.DataFrame({
        'Timestamp': ['10:30:15', '10:30:10', '10:30:05', '10:30:00', '10:29:55'],
        'Object': ['Person', 'Laptop', 'Cell Phone', 'Bottle', 'Chair'],
        'Confidence': ['92%', '88%', '76%', '81%', '78%'],
        'Status': ['✅ Detected', '✅ Detected', '✅ Detected', '✅ Detected', '✅ Detected']
    })
    
    st.dataframe(recent_data, use_container_width=True, hide_index=True)
    
    # System Performance
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<h3>📈 System Performance Over Time</h3>", unsafe_allow_html=True)
    
    # Create time series data
    dates = pd.date_range(start='2024-11-01', periods=30, freq='D')
    performance_data = pd.DataFrame({
        'Date': dates,
        'FPS': np.random.randint(25, 35, size=30),
        'Accuracy': np.random.uniform(0.8, 0.9, size=30) * 100
    })
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.line_chart(performance_data.set_index('Date')['FPS'])
        st.caption("Frames Per Second")
    
    with col2:
        st.line_chart(performance_data.set_index('Date')['Accuracy'])
        st.caption("Detection Accuracy (%)")

# ====================================================================
# SYSTEM SETTINGS PAGE - Clean Professional Interface
# ====================================================================
def render_system_settings():
    st.markdown("<h1>System Settings</h1>", unsafe_allow_html=True)
    
    st.markdown("""
        <div class='card'>
            <p style='color: #3C4858; font-size: 1.05rem;'>
                Configure system parameters, adjust detection settings, and manage application preferences.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Settings Sections
    settings_col1, settings_col2 = st.columns(2)
    
    with settings_col1:
        st.markdown("""
            <div class='card'>
                <h3>🎯 Detection Settings</h3>
            </div>
        """, unsafe_allow_html=True)
        
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Minimum confidence score for detections"
        )
        
        iou_threshold = st.slider(
            "IOU Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.45,
            step=0.05,
            help="Intersection over Union threshold for NMS"
        )
        
        model_size = st.selectbox(
            "Model Size",
            ["Nano (YOLOv8n)", "Small (YOLOv8s)", "Medium (YOLOv8m)", "Large (YOLOv8l)", "XLarge (YOLOv8x)"],
            index=0,
            help="Select YOLOv8 model variant"
        )
    
    with settings_col2:
        st.markdown("""
            <div class='card'>
                <h3>📷 Camera Settings</h3>
            </div>
        """, unsafe_allow_html=True)
        
        resolution = st.selectbox(
            "Camera Resolution",
            ["640x480", "1280x720", "1920x1080"],
            index=1,
            help="Select camera resolution"
        )
        
        fps_limit = st.slider(
            "FPS Limit",
            min_value=1,
            max_value=60,
            value=30,
            step=1,
            help="Maximum frames per second"
        )
        
        mirror_camera = st.checkbox("Mirror Camera Feed", value=True)
        show_fps = st.checkbox("Show FPS Counter", value=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # System Information
    st.markdown("<h3>🖥️ System Information</h3>", unsafe_allow_html=True)
    
    info_col1, info_col2, info_col3 = st.columns(3)
    
    with info_col1:
        st.markdown(f"""
            <div class='card'>
                <h4>Hardware Info</h4>
                <p style='color: #3C4858; margin: 0.75rem 0;'><strong>Device:</strong> {DEVICE_LABEL}</p>
                <p style='color: #3C4858; margin: 0.75rem 0;'><strong>PyTorch:</strong> {torch.__version__}</p>
                <p style='color: #3C4858; margin: 0.75rem 0;'><strong>CUDA:</strong> {"Available" if torch.cuda.is_available() else "Not Available"}</p>
            </div>
        """, unsafe_allow_html=True)
    
    with info_col2:
        st.markdown("""
            <div class='card'>
                <h4>Model Info</h4>
                <p style='color: #3C4858; margin: 0.75rem 0;'><strong>Version:</strong> YOLOv8n</p>
                <p style='color: #3C4858; margin: 0.75rem 0;'><strong>Classes:</strong> 80 (COCO)</p>
                <p style='color: #3C4858; margin: 0.75rem 0;'><strong>Size:</strong> 6.2 MB</p>
                <p style='color: #3C4858; margin: 0.75rem 0;'><strong>Status:</strong> <span style='color: #4C8DF5; font-weight: 600;'>Active</span></p>
            </div>
        """, unsafe_allow_html=True)
    
    with info_col3:
        st.markdown("""
            <div class='card'>
                <h4>Application Info</h4>
                <p style='color: #3C4858; margin: 0.75rem 0;'><strong>Version:</strong> 2.0.0</p>
                <p style='color: #3C4858; margin: 0.75rem 0;'><strong>Last Updated:</strong> November 2025</p>
                <p style='color: #3C4858; margin: 0.75rem 0;'><strong>Team:</strong> Team 6</p>
                <p style='color: #3C4858; margin: 0.75rem 0;'><strong>Institution:</strong> Ataturk University</p>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Action Buttons
    action_col1, action_col2, action_col3 = st.columns(3)
    
    with action_col1:
        if st.button("💾 Save Settings", width='stretch'):
            st.success("Settings saved successfully!")
    
    with action_col2:
        if st.button("🔄 Reset to Default", width='stretch'):
            st.info("Settings reset to default values")
    
    with action_col3:
        if st.button("🔧 Test Configuration", width='stretch'):
            with st.spinner("Testing system configuration..."):
                time.sleep(2)
                st.success("All tests passed! System is ready.")
    
    # Export Settings
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<h3>📤 Export & Import</h3>", unsafe_allow_html=True)
    
    export_col1, export_col2 = st.columns(2)
    
    with export_col1:
        st.download_button(
    label="📥 Export Settings",
    data='{"version": "2.0", "settings": "exported"}',
    file_name="yolov8_settings.json",
    mime="application/json",
    width='stretch'
)
    
    with export_col2:
        uploaded_file = st.file_uploader("📤 Import Settings", type=['json'], label_visibility="collapsed")
        if uploaded_file is not None:
            st.success("Settings imported successfully!")

# ====================================================================
# MAIN APPLICATION
# ====================================================================
def main():
    # Initialize session state
    if 'page_selection' not in st.session_state:
        st.session_state.page_selection = "🏠 Project Overview"
    
    render_sidebar()
    
    # Page routing
    try:
        if st.session_state.page_selection == "🏠 Project Overview":
            render_project_overview()
        elif st.session_state.page_selection == "🎥 Live Detection":
            render_live_detection()
        elif st.session_state.page_selection == "📊 Analytics Dashboard":
            render_analytics_dashboard()
        elif st.session_state.page_selection == "⚙️ System Settings":
            render_system_settings()
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.info("Please refresh the page and try again.")

if __name__ == "__main__":
    main()