import streamlit as st
import av
import torch
import pandas as pd
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode, RTCConfiguration
from ultralytics import YOLO
from dataclasses import dataclass, field
from threading import Lock
from pathlib import Path

# ====================================================================
# PAGE CONFIGURATION
# ====================================================================
st.set_page_config(
    page_title="Multi-Model YOLO Vision System",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ====================================================================
# THREAD-SAFE INFERENCE SETTINGS
# ====================================================================
@dataclass
class InferenceSettings:
    """Thread-safe container for inference parameters."""
    conf: float = 0.05
    iou: float = 0.45
    _lock: Lock = field(default_factory=Lock, repr=False)

    def update(self, conf: float, iou: float) -> None:
        """Safely update settings from UI thread."""
        conf = float(max(0.0, min(1.0, conf)))
        iou = float(max(0.0, min(1.0, iou)))
        with self._lock:
            self.conf = conf
            self.iou = iou

    def snapshot(self) -> tuple:
        """Safely read current settings from video processor thread."""
        with self._lock:
            return self.conf, self.iou

# ====================================================================
# ROBUST INIT - CRITICAL FIX
# ====================================================================
def init_app_state():
    """Initialize all session state variables safely."""
    # 1. State for page navigation
    if 'page_selection' not in st.session_state:
        st.session_state.page_selection = "🏠 Overview"
    
    # 2. State for sliders (primitive values)
    if "conf_threshold" not in st.session_state:
        st.session_state["conf_threshold"] = 0.05
    if "iou_threshold" not in st.session_state:
        st.session_state["iou_threshold"] = 0.45
        
    # 3. State for thread-safe settings object
    if "inference_settings" not in st.session_state:
        st.session_state["inference_settings"] = InferenceSettings(
            conf=st.session_state["conf_threshold"], 
            iou=st.session_state["iou_threshold"]
        )

# Call immediately
init_app_state()

# ====================================================================
# OPTIMIZED DARK THEME - Fast & Professional
# ====================================================================
st.markdown("""
    <style>
    /* Base styling - Minimal for performance */
    * {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 1rem;
        max-width: 95% !important;
    }
    
    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* Dark Background */
    .stApp {
        background: #050810 !important;
        color: #F9FAFB !important;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #F9FAFB !important;
        font-weight: 600 !important;
    }
    
    h1 {
        font-size: 2.5rem !important;
        margin-bottom: 1.25rem !important;
        font-weight: 700 !important;
        padding-bottom: 0.5rem !important;
        border-bottom: 3px solid #F5C453 !important;
    }
    
    h2 {
        font-size: 1.875rem !important;
        margin-bottom: 1rem !important;
        padding-bottom: 0.75rem !important;
        border-bottom: 2px solid #F5C453 !important;
    }
    
    h3 {
        font-size: 1.375rem !important;
        margin-bottom: 0.75rem !important;
        color: #F5C453 !important;
    }
    
    h4 {
        color: #F9FAFB !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        margin-bottom: 0.5rem !important;
    }
    
    /* Cards - Simplified hover */
    .dark-card {
        background: #0F172A !important;
        border-radius: 16px !important;
        padding: 1.75rem !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3) !important;
        border: 1px solid #1F2933 !important;
        margin-bottom: 1.25rem !important;
        transition: all 0.3s ease-in-out !important;
    }
    
    .dark-card:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 8px 24px rgba(245, 196, 83, 0.15) !important;
        border-color: rgba(245, 196, 83, 0.3) !important;
    }
    
    /* Metrics */
    div[data-testid="stMetric"] {
        background: #0F172A !important;
        border-radius: 12px !important;
        border: 1px solid #1F2933 !important;
        padding: 1.25rem !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2) !important;
    }
    
    div[data-testid="stMetric"] label {
        color: #9CA3AF !important;
        font-weight: 500 !important;
        font-size: 0.9rem !important;
    }
    
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #F5C453 !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
    }
    
    /* Buttons - Black & Gold */
    .stButton > button {
        background: linear-gradient(135deg, #1a1a1a 0%, #2d2416 100%) !important;
        color: #F5C453 !important;
        border: 1px solid rgba(245, 196, 83, 0.3) !important;
        padding: 0.75rem 2rem !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        border-radius: 10px !important;
        transition: all 0.25s ease-in-out !important;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #2d2416 0%, #3d3118 100%) !important;
        border-color: #F5C453 !important;
        box-shadow: 0 4px 15px rgba(245, 196, 83, 0.3) !important;
        transform: translateY(-2px) !important;
    }
    
    /* Sidebar - Enhanced with better spacing and transitions */
    [data-testid="stSidebar"] {
        background: #0B1120 !important;
        border-right: 1px solid #1F2933 !important;
    }
    
    [data-testid="stSidebar"] .stRadio > div > label {
        background: transparent !important;
        border-radius: 8px !important;
        padding: 0.75rem 1rem !important;
        border: 1px solid #1F2933 !important;
        color: #E5E7EB !important;
        transition: all 0.3s ease-in-out !important;
        margin: 0.35rem 0 !important;
    }
    
    [data-testid="stSidebar"] .stRadio > div > label:hover {
        background: rgba(245, 196, 83, 0.05) !important;
        border-color: rgba(245, 196, 83, 0.3) !important;
        transform: translateX(2px) !important;
    }
    
    [data-testid="stSidebar"] .stRadio > div > label[data-baseweb="radio"]:has(input:checked) {
        background: rgba(245, 196, 83, 0.12) !important;
        border-left: 3px solid #F5C453 !important;
        border-color: #F5C453 !important;
        color: #F9FAFB !important;
        font-weight: 600 !important;
        box-shadow: 0 0 12px rgba(245, 196, 83, 0.15) !important;
    }
    
    /* Video - Enhanced container with subtle depth */
    video {
        border-radius: 12px !important;
        border: 2px solid #1F2933 !important;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.15) !important;
        transition: all 0.25s ease-in-out !important;
    }
    
    video:hover {
        box-shadow: 0 6px 24px rgba(0, 0, 0, 0.2), 0 0 0 1px rgba(245, 196, 83, 0.1) !important;
    }
    
    /* Text */
    p {
        color: #D1D5DB !important;
        line-height: 1.6 !important;
        font-size: 1rem !important;
    }
    
    /* Lists */
    ul, ol {
        color: #D1D5DB !important;
        line-height: 1.7 !important;
        padding-left: 1.5rem !important;
    }
    
    li {
        margin-bottom: 0.5rem !important;
    }
    
    /* Badge */
    .tech-badge {
        display: inline-block;
        padding: 0.4rem 0.875rem;
        background: rgba(34, 197, 94, 0.1);
        color: #22C55E;
        border-radius: 20px;
        font-size: 0.875rem;
        font-weight: 500;
        margin: 0.25rem;
        border: 1px solid rgba(34, 197, 94, 0.3);
    }
    
    /* Status */
    .status-active {
        color: #22C55E;
        font-weight: 600;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #0B1120;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #F5C453;
        border-radius: 4px;
    }
    
    /* Team card - Simplified */
    .team-card {
        text-align: center;
        padding: 1.5rem;
        background: #0F172A;
        border: 1px solid #1F2933;
        border-radius: 12px;
        transition: all 0.3s ease-in-out !important;
        height: 100%;
    }
    
    .team-card:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 8px 20px rgba(245, 196, 83, 0.15) !important;
        border-color: rgba(245, 196, 83, 0.3) !important;
    }
    
    .team-card h4 {
        color: #F9FAFB;
        margin: 0.5rem 0 0.25rem 0;
        font-size: 1.05rem;
        font-weight: 600;
    }
    
    .team-card p {
        color: #F5C453 !important;
        margin: 0;
        font-size: 0.9rem;
        font-weight: 500;
    }
    
    .team-emoji {
        font-size: 2rem;
        margin-bottom: 0.5rem;
        display: block;
    }
    
    /* Dataframes */
    .stDataFrame {
        background: #0F172A !important;
    }
    
    /* Info/warning boxes */
    .stAlert {
        background: #0F172A !important;
        border: 1px solid #1F2933 !important;
        border-radius: 8px !important;
    }
    
    /* Selection boxes */
    .stSelectbox [data-baseweb="select"] {
        background: #0F172A !important;
        border-color: #1F2933 !important;
        color: #F9FAFB !important;
    }
    
    /* Number input */
    .stNumberInput input {
        background: #0F172A !important;
        border-color: #1F2933 !important;
        color: #F9FAFB !important;
    }
    
    /* Sliders - Enhanced for premium feel */
    .stSlider > div > div > div {
        background: #1F2933 !important;
    }
    
    .stSlider > div > div > div > div {
        background: linear-gradient(135deg, #F5C453 0%, #d4a947 100%) !important;
        box-shadow: 0 2px 8px rgba(245, 196, 83, 0.3) !important;
    }
    
    /* Slider thumb enhancement */
    .stSlider [role="slider"] {
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.3) !important;
        transition: all 0.2s ease-in-out !important;
    }
    
    .stSlider [role="slider"]:hover {
        box-shadow: 0 3px 10px rgba(245, 196, 83, 0.4) !important;
    }
    
    /* Subtle pulse animation for value updates */
    @keyframes subtle-pulse {
        0%, 100% { 
            opacity: 1;
            transform: scale(1);
        }
        50% { 
            opacity: 0.95;
            transform: scale(0.99);
        }
    }
    
    /* Expander refinements */
    .streamlit-expanderHeader {
        transition: all 0.3s ease-in-out !important;
    }
    
    .streamlit-expanderHeader:hover {
        background: rgba(245, 196, 83, 0.03) !important;
    }
    
    /* Divider spacing enhancement */
    hr {
        margin: 1.5rem 0 !important;
        opacity: 0.6 !important;
    }
    </style>
""", unsafe_allow_html=True)

# ====================================================================
# DEVICE CONFIGURATION - Optimized
# ====================================================================
@st.cache_resource
def get_device():
    """Get device with fallback"""
    if torch.backends.mps.is_available():
        return "mps", "🤖 Apple Silicon (MPS)"
    elif torch.cuda.is_available():
        return "cuda", "🚀 NVIDIA GPU"
    else:
        return "cpu", "💻 CPU"

DEVICE, DEVICE_LABEL = get_device()

# ====================================================================
# OPTIMIZED MODEL LOADING
# ====================================================================
# ====================================================================
# OPTIMIZED MODEL LOADING
# ====================================================================
MODEL_CONFIG = {
    "YOLOv8 (Balanced)": Path("training_results") / "yolov8_fruits" / "weights_best.pt",
    "YOLOv11 (High Accuracy)": Path("training_results") / "yolo11_fruits" / "weights_best.pt",
    "YOLOv5 (High Speed)": Path("training_results") / "yolov5_fruits" / "yolov5_fruits" / "weights_best.pt",
    "YOLOv8 Nano (Default)": Path("yolov8n.pt")
}

@st.cache_resource(show_spinner=False)
def load_yolo_model(model_path):
    """Load YOLO model efficiently with fallback logic."""
    path_obj = Path(model_path)
    
    # 1. Validate existence
    if not path_obj.exists():
        st.warning(f"⚠️ Weights file not found: `{path_obj}`. Falling back to default YOLOv8n model.")
        path_obj = Path("yolov8n.pt")
    
    try:
        # 2. Prevent loading YOLOv5 weights with YOLOv8/11 loader if incompatible
        # Ultralytics can load some YOLOv5 models but not all. 
        # If it fails, we catch it below.
        model = YOLO(str(path_obj))
        model.to(DEVICE)
        
        # 3. Fuse for speed
        if hasattr(model, 'fuse'):
            model.fuse()
            
        return model
    except Exception as e:
        st.error(f"Error loading model `{path_obj}`: {e}. using default fallback.")
        try:
            return YOLO("yolov8n.pt")
        except Exception as e2:
            st.error(f"Critical: Failed to load fallback model: {e2}")
            return None

MODEL = None

# ====================================================================
# OPTIMIZED VIDEO PROCESSOR
# ====================================================================
class YOLOv8VideoProcessor(VideoProcessorBase):
    def __init__(self, settings: InferenceSettings):
        super().__init__()
        self.model = MODEL
        self.settings = settings
        
    def recv(self, frame):
        if self.model is None:
            return frame
            
        try:
            img = frame.to_ndarray(format="bgr24")
            # Get current settings safely from the settings object
            conf, iou = self.settings.snapshot()
            # Optimized inference with user-controlled conf and iou
            results = self.model(img, 
                               conf=conf,
                               iou=iou,
                               verbose=False,
                               max_det=10,  # Limit detections for speed
                               imgsz=480)   # Smaller size for faster processing
            processed_img = results[0].plot()
            return av.VideoFrame.from_ndarray(processed_img, format="bgr24")
        except Exception as e:
            print(f"Processing error: {e}")
            return frame

# ====================================================================
# SIDEBAR NAVIGATION - Clean
# ====================================================================
def render_sidebar():
    with st.sidebar:
        # Header - Clean with icon
        st.markdown("""
            <div style='text-align: center; margin-bottom: 2rem;'>
                <div style='font-size: 2.5rem; margin-bottom: 0.5rem;'>🎯</div>
                <h3 style='margin: 0; color: #F5C453;'>Multi-Model Vision</h3>
                <p style='color: #9CA3AF; margin: 0.5rem 0 0 0; font-size: 0.9rem;'>
                    Real-Time Object Detection
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        st.divider()
        
        # Navigation
        options = ["🏠 Overview", "🎥 Live Detection", "📊 Analytics"]
        
        if "page_selection" not in st.session_state:
            st.session_state.page_selection = options[0]

        selection = st.radio(
            "Navigation", 
            options, 
            index=options.index(st.session_state.page_selection),
            label_visibility="collapsed"
        )
        
        if selection != st.session_state.page_selection:
            st.session_state.page_selection = selection
            st.rerun()

        st.divider()

        # Model Selection
        st.markdown(f"""
            <div style='margin-bottom: 0.5rem;'>
                <h4 style='color: #F5C453; margin: 0 0 0.5rem 0; font-size: 1rem;'>⚙️ Model Config</h4>
            </div>
        """, unsafe_allow_html=True)

        selected_model_name = st.selectbox(
            "Select AI Model",
            list(MODEL_CONFIG.keys()),
            index=0,
            label_visibility="collapsed",
            key="model_selection"
        )
        
        # Load the selected model globally
        global MODEL
        MODEL = load_yolo_model(MODEL_CONFIG[selected_model_name])
        
        st.divider()
        
        # Inference Settings
        st.markdown(f"""
            <div style='margin-bottom: 0.5rem;'>
                <h4 style='color: #F5C453; margin: 0 0 0.5rem 0; font-size: 1rem;'>🛠️ Inference Settings</h4>
            </div>
        """, unsafe_allow_html=True)
        
        with st.expander("Adjust Thresholds", expanded=True):
            # Sliders use session state keys directly (initialized in init_app_state)
            conf_val = st.slider(
                "Confidence Threshold",
                min_value=0.00,
                max_value=1.00,
                value=st.session_state["conf_threshold"],
                step=0.01,
                help="Minimum confidence to show a detection. Lower values show more detections.",
                key="conf_slider"
            )
            
            iou_val = st.slider(
                "IoU Threshold",
                min_value=0.00,
                max_value=1.00,
                value=st.session_state["iou_threshold"],
                step=0.01,
                help="NMS IoU threshold; lower values remove more overlapping boxes.",
                key="iou_slider"
            )
            
            # Update both session state keys AND settings object
            st.session_state["conf_threshold"] = conf_val
            st.session_state["iou_threshold"] = iou_val
            st.session_state["inference_settings"].update(conf_val, iou_val)
            
            # Display current values
            st.markdown(f"""
                <div style='margin-top: 0.75rem; padding: 0.75rem; background: rgba(245, 196, 83, 0.05); border-radius: 8px; border: 1px solid rgba(245, 196, 83, 0.2);'>
                    <div style='display: flex; justify-content: space-between; margin-bottom: 0.25rem;'>
                        <span style='color: #9CA3AF; font-size: 0.85rem;'>Confidence:</span>
                        <span style='color: #F5C453; font-weight: 600; font-size: 0.85rem;'>{{conf_val:.2f}}</span>
                    </div>
                    <div style='display: flex; justify-content: space-between;'>
                        <span style='color: #9CA3AF; font-size: 0.85rem;'>IoU:</span>
                        <span style='color: #F5C453; font-weight: 600; font-size: 0.85rem;'>{{iou_val:.2f}}</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        
        st.divider()
        
        # System Status
        st.markdown(f"""
            <div style='background: #0F172A; border: 1px solid #1F2933; border-radius: 12px; padding: 1.25rem; margin: 1rem 0;'>
                <h4 style='color: #F5C453; margin: 0 0 1rem 0; font-size: 1rem;'>System Status</h4>
                <div style='display: flex; justify-content: space-between; margin-bottom: 0.5rem;'>
                    <span style='color: #9CA3AF; font-size: 0.9rem;'>Hardware</span>
                    <span style='color: #22C55E; font-weight: 600; font-size: 0.9rem;'>{DEVICE_LABEL}</span>
                </div>
                <div style='display: flex; justify-content: space-between; margin-bottom: 0.5rem;'>
                    <span style='color: #9CA3AF; font-size: 0.9rem;'>Model</span>
                    <span style='color: #22C55E; font-weight: 600; font-size: 0.9rem;'>{selected_model_name.split()[0]}</span>
                </div>
                <div style='display: flex; justify-content: space-between;'>
                    <span style='color: #9CA3AF; font-size: 0.9rem;'>Status</span>
                    <span style='color: #22C55E; font-weight: 600; font-size: 0.9rem;'>● Active</span>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        st.divider()
        
        # Footer
        st.markdown("""
            <div style='text-align: center; padding: 1rem 0;'>
                <p style='color: #F5C453; font-weight: 600; margin-bottom: 0.25rem;'>Team 6</p>
                <p style='color: #9CA3AF; font-size: 0.85rem;'>Ataturk University</p>
                <p style='color: #6B7280; font-size: 0.75rem; margin-top: 0.75rem;'>
                    Computer Vision Lab
                </p>
            </div>
        """, unsafe_allow_html=True)

# ====================================================================
# PROJECT OVERVIEW PAGE - Optimized
# ====================================================================
def render_project_overview():
    st.markdown("<h1>Multi-Model YOLO Vision System</h1>", unsafe_allow_html=True)
    
    # 1. Project Introduction
    st.markdown("""
        <div class='dark-card'>
            <h3>🎯 Project Introduction</h3>
            <p>
                This system implements real-time object detection and localization using YOLOv8 architecture. 
                The prototype uses pre-trained YOLOv8 model for inference through a Streamlit web interface 
                with optimized performance.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # 2. Project Overview - Key Metrics
    st.markdown("<h2>Project Overview</h2>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(label="Model", value="Multi-Model", delta="YOLO v5/v8/v11")
    
    with col2:
        st.metric(label="Object Classes", value="80+", delta="Classes")
    
    with col3:
        st.metric(label="Performance", value="~30 FPS", delta="Real-Time")
    
    # 3. Technology Stack
    st.markdown("<h2>Technology Stack</h2>", unsafe_allow_html=True)
    
    st.markdown("""
        <div class='dark-card'>
            <div style='display: flex; flex-wrap: wrap; gap: 0.5rem;'>
                <span class='tech-badge'>Python</span>
                <span class='tech-badge'>YOLO (v5/v8/v11)</span>
                <span class='tech-badge'>Streamlit</span>
                <span class='tech-badge'>PyTorch</span>
                <span class='tech-badge'>OpenCV</span>
                <span class='tech-badge'>WebRTC</span>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # 4. Methodology
    st.markdown("<h2>Methodology & Performance</h2>", unsafe_allow_html=True)

    # Performance Comparison Section
    st.markdown("""
        <div class='dark-card'>
            <h3>🏆 Model Performance Comparison</h3>
            <p>We trained and compared three different YOLO architectures on the same dataset. Here are the results:</p>
            <table style="width:100%; color: #D1D5DB; border-collapse: collapse;">
                <tr style="border-bottom: 2px solid #F5C453;">
                    <th style="padding: 10px; text-align: left; color: #F5C453;">Model Architecture</th>
                    <th style="padding: 10px; text-align: center; color: #F5C453;">Accuracy (mAP50)</th>
                    <th style="padding: 10px; text-align: center; color: #F5C453;">Recall</th>
                    <th style="padding: 10px; text-align: center; color: #F5C453;">Speed (Inference)</th>
                    <th style="padding: 10px; text-align: center; color: #F5C453;">Training Time</th>
                    <th style="padding: 10px; text-align: left; color: #F5C453;">Best Use Case</th>
                </tr>
                <tr style="border-bottom: 1px solid #1F2933;">
                    <td style="padding: 10px;"><strong>YOLOv11 Medium</strong> <span style="color:#22C55E">★ Winner</span></td>
                    <td style="padding: 10px; text-align: center;"><strong>77.8%</strong></td>
                    <td style="padding: 10px; text-align: center;"><strong>75.5%</strong></td>
                    <td style="padding: 10px; text-align: center;">12.8 ms</td>
                    <td style="padding: 10px; text-align: center;">~1.49 hrs</td>
                    <td style="padding: 10px;">High-Accuracy Applications</td>
                </tr>
                <tr style="border-bottom: 1px solid #1F2933;">
                    <td style="padding: 10px;"><strong>YOLOv5 Medium</strong> <span style="color:#3B82F6">⚡ Fastest</span></td>
                    <td style="padding: 10px; text-align: center;">76.7%</td>
                    <td style="padding: 10px; text-align: center;">72.1%</td>
                    <td style="padding: 10px; text-align: center;"><strong>10.4 ms</strong></td>
                    <td style="padding: 10px; text-align: center;">~1.25 hrs</td>
                    <td style="padding: 10px;">Real-Time / Heavy Load</td>
                </tr>
                <tr>
                    <td style="padding: 10px;"><strong>YOLOv8 Medium</strong></td>
                    <td style="padding: 10px; text-align: center;">76.5%</td>
                    <td style="padding: 10px; text-align: center;">74.8%</td>
                    <td style="padding: 10px; text-align: center;">11.6 ms</td>
                    <td style="padding: 10px; text-align: center;">~1.72 hrs</td>
                    <td style="padding: 10px;">Balanced Performance</td>
                </tr>
            </table>
            <p style="margin-top: 1rem; font-size: 0.85rem; color: #9CA3AF;">
                * Comparison based on 50 epochs training on Google Colab T4 GPU.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
            <div class='dark-card'>
                <h3>Dataset & Model</h3>
                <ul>
                    <li><strong>Dataset:</strong> Fruit Classification dataset</li>
                    <li><strong>Selected Classes:</strong> 9 fruit classes</li>
                    <li><strong>Samples:</strong> ~3,000 images total</li>
                    <li><strong>Train/Val:</strong> 2697 train, 187 val</li>
                    <li><strong>Model:</strong> Multi-Architecture YOLO</li>
                    <li><strong>Weights:</strong> Custom trained weights</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class='dark-card'>
                <h3>Deployment & Tools</h3>
                <ul>
                    <li><strong>Framework:</strong> Ultralytics Multi-Model</li>
                    <li><strong>Interface:</strong> Streamlit Web Interface</li>
                    <li><strong>Processing:</strong> OpenCV + PyTorch</li>
                    <li><strong>Streaming:</strong> WebRTC real-time</li>
                    <li><strong>Deployment:</strong> Local/Web optimized</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    # 5. Development Team (Emojis only, no contact info)
    st.markdown("<h2>Development Team</h2>", unsafe_allow_html=True)
    
    team_col1, team_col2, team_col3 = st.columns(3)
    
    with team_col1:
        st.markdown("""
            <div class='team-card'>
                <span class='team-emoji'>👨‍💻</span>
                <h4>Abdelrahman MOHAMED</h4>
                <p style="color: #F5C453; font-weight: bold;">YOLOv5 Specialist & Web Dev.</p>
                <div style="text-align: left; margin-top: 0.5rem; font-size: 0.8rem; color: #D1D5DB;">
                    • Trained YOLOv5 Model<br>
                    • Analyzed YOLOv5 Metric Results<br>
                    • Developed Web Interface<br>
                    • Front-end Integration
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    with team_col2:
        st.markdown("""
            <div class='team-card'>
                <span class='team-emoji'>👨‍🔬</span>
                <h4>Ramazan YILDIZ</h4>
                <p style="color: #F5C453; font-weight: bold;">Project Planning & AI Research</p>
                <div style="text-align: left; margin-top: 0.5rem; font-size: 0.8rem; color: #D1D5DB;">
                    • Trained YOLOv8m & YOLOv8n Models<br>
                    • Project Initialization & Prep<br>
                    • Research & Dataset Preparation<br>
                    • Technical Documentation<br>
                    • Model Comparison Analysis<br>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    with team_col3:
        st.markdown("""
            <div class='team-card'>
                <span class='team-emoji'>👩‍💼</span>
                <h4>Beyza GÜLER</h4>
                <p style="color: #F5C453; font-weight: bold;">YOLOv11 Specialist & Reporting</p>
                <div style="text-align: left; margin-top: 0.5rem; font-size: 0.8rem; color: #D1D5DB;">
                    • Trained YOLOv11 Model<br>
                    • Analyzed YOLOv11 Metric Results<br>
                    • Technical Documentation<br>
                    • Presentation Slides Design<br>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    # 6. Project Roadmap
    st.markdown("<h2>Project Roadmap</h2>", unsafe_allow_html=True)
    
    roadmap_col1, roadmap_col2 = st.columns(2)
    
    with roadmap_col1:
        st.markdown("""
            <div class='dark-card'>
                <h3>✅ Phase 1: Research & Training</h3>
                <ul>
                    <li>Utilized Pre-labeled Dataset (9 Classes)</li>
                    <li>Trained Multi-Model Architecture</li>
                    <li>Evaluated YOLOv5 vs v8 vs v11</li>
                    <li>Captured Performance Metrics</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    with roadmap_col2:
        st.markdown("""
            <div class='dark-card'>
                <h3>✅ Phase 2: Deployment (Complete)</h3>
                <ul>
                    <li>Built Streamlit Web Interface</li>
                    <li>Integrated Real-Time WebRTC</li>
                    <li>Implemented Model Switching</li>
                    <li>Final System Validation</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    # 7. Call to Action
    st.markdown("""
        <div class='dark-card' style='text-align: center; border-color: #F5C453;'>
            <h3 style='margin-bottom: 1rem;'>🚀 Ready to Experience AI Vision?</h3>
            <p style='margin-bottom: 1.5rem;'>
                Activate your webcam and see Multi-Model YOLO in real-time action
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    if st.button("🎥 Start Live Detection", width='stretch'):
        st.session_state.page_selection = "🎥 Live Detection"
        st.rerun()

# ====================================================================
# LIVE DETECTION PAGE - Fixed WebRTC Configuration
# ====================================================================
def render_live_detection():
    st.markdown("<h1>Live Object Detection</h1>", unsafe_allow_html=True)
    
    # Status info
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.markdown("""
            <div class='dark-card' style='margin-bottom: 1rem;'>
                <p style='margin: 0;'>
                    <span style='color: #22C55E;'>●</span> 
                    <strong>LIVE</strong> - Using YOLOv8n pre-trained on COCO dataset (80+ object classes)
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.metric(label="Hardware", value=DEVICE_LABEL)
    
    with col3:
        st.metric(label="Performance", value="~30 FPS")
    
    # Main camera feed - FIXED WebRTC configuration
    st.markdown("<br>", unsafe_allow_html=True)
    
    try:
        # Use a dynamic key based on the model to force Streamlit to recreate 
        # the streamer when the model changes
        # Create a dynamic key to force streamer reset on model change
        model_key = f"yolo_detection_{st.session_state.get('model_selection', 'default')}"
        
        # CRITICAL FIX: Capture settings object here in main thread
        # Do NOT access st.session_state inside the factory function
        current_settings = st.session_state["inference_settings"]
        
        def video_processor_factory():
            return YOLOv8VideoProcessor(settings=current_settings)
        
        webrtc_ctx = webrtc_streamer(
            key=model_key,
            video_processor_factory=video_processor_factory,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
            mode=WebRtcMode.SENDRECV,
            rtc_configuration={
                "iceServers": [
                    {"urls": ["stun:stun.l.google.com:19302"]},
                    {"urls": ["stun:stun1.l.google.com:19302"]},
                    {"urls": ["stun:stun2.l.google.com:19302"]},
                    {"urls": ["stun:stun3.l.google.com:19302"]},
                    {"urls": ["stun:stun4.l.google.com:19302"]},
                ]
            }
        )
        
        if not webrtc_ctx.state.playing:
            st.info("👆 Click **START** above to begin camera detection")
            
    except Exception as e:
        st.error(f"Camera Error: {str(e)}")
        st.warning("""
            **Troubleshooting Tips:**
            - Check browser camera permissions
            - Ensure no other app is using camera
            - Try refreshing the page
            - Use Chrome or Edge browser
        """)
    
    # Detection info
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
            <div class='dark-card'>
                <h3>📋 Detection Capabilities</h3>
                <ul>
                    <li>Apple, Banana, Orange</li>
                    <li>Grapes, Kiwi, Mango</li>
                    <li>Pineapple, Watermelon</li>
                    <li>Sugerapple</li>
                    <li><strong>9 Fruit Classes Total</strong></li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class='dark-card'>
                <h3>💡 Tips for Best Results</h3>
                <ul>
                    <li><strong>Lighting:</strong> Ensure good lighting</li>
                    <li><strong>Distance:</strong> Stay 1-2 meters from camera</li>
                    <li><strong>Stability:</strong> Keep camera steady</li>
                    <li><strong>Objects:</strong> Present clearly visible objects</li>
                    <li><strong>Background:</strong> Simple background helps</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

# ====================================================================
# ANALYTICS DASHBOARD - Optimized
# ====================================================================
@st.cache_data(ttl=60, show_spinner=False)
def generate_sample_analytics():
    """Generate sample analytics data"""
    detection_data = pd.DataFrame({
        'Object': ['Person', 'Car', 'Chair', 'Bottle', 'Laptop', 'Phone', 'Book', 'Cup'],
        'Count': [150, 120, 85, 65, 50, 45, 40, 35],
        'Confidence': ['89%', '87%', '85%', '82%', '88%', '84%', '81%', '83%']
    })
    return detection_data

def render_analytics_dashboard():
    st.markdown("<h1>Analytics Dashboard</h1>", unsafe_allow_html=True)
    
    st.markdown("""
        <div class='dark-card'>
            <p>
                Detection performance metrics and statistics based on system usage.
                <em>(Simulated data for demonstration)</em>
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Key Performance Indicators
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(label="Total Detections", value="1,250")
    
    with col2:
        st.metric(label="Avg Confidence", value="84.5%")
    
    with col3:
        st.metric(label="Unique Classes", value="18")
    
    with col4:
        st.metric(label="Processing Time", value="32ms")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Detection Statistics
    st.markdown("<h2>Detection Statistics</h2>", unsafe_allow_html=True)
    
    detection_data = generate_sample_analytics()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
            <div class='dark-card'>
                <h3>📊 Detection Frequency</h3>
            </div>
        """, unsafe_allow_html=True)
        
        st.bar_chart(detection_data.set_index('Object')['Count'], color="#F5C453")
    
    with col2:
        st.markdown("""
            <div class='dark-card'>
                <h3>📋 Top Detections</h3>
            </div>
        """, unsafe_allow_html=True)
        
        top_data = detection_data.nlargest(5, 'Count')
        for _, row in top_data.iterrows():
            st.markdown(f"""
                <div style='display: flex; justify-content: space-between; padding: 0.5rem; 
                    border-bottom: 1px solid #1F2933;'>
                    <span>{row['Object']}</span>
                    <span><strong>{row['Count']}</strong></span>
                </div>
            """, unsafe_allow_html=True)
    
    # Simple Table
    st.markdown("<h2>Detection Details</h2>", unsafe_allow_html=True)
    
    st.dataframe(
        detection_data,
        column_config={
            "Object": "Object Class",
            "Count": "Detection Count",
            "Confidence": "Avg Confidence"
        },
        hide_index=True,
        width='stretch'
    )

# ====================================================================
# MAIN APPLICATION
# ====================================================================
def main():
    # Session state is guaranteed to be initialized by init_app_state() call at top of file
    render_sidebar()
    
    try:
        if st.session_state.page_selection == "🏠 Overview":
            render_project_overview()
        elif st.session_state.page_selection == "🎥 Live Detection":
            render_live_detection()
        elif st.session_state.page_selection == "📊 Analytics":
            render_analytics_dashboard()
    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.info("Please refresh the page and try again.")

if __name__ == "__main__":
    main()