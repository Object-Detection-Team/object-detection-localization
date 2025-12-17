import streamlit as st
import av
import torch
import pandas as pd
import numpy as np
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
        transition: transform 0.2s ease, box-shadow 0.2s ease !important;
    }
    
    .dark-card:hover {
        transform: translateY(-4px) !important;
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
        transition: all 0.2s ease !important;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #2d2416 0%, #3d3118 100%) !important;
        border-color: #F5C453 !important;
        box-shadow: 0 4px 15px rgba(245, 196, 83, 0.3) !important;
        transform: translateY(-2px) !important;
    }
    
    /* Sidebar */
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
        transition: all 0.2s ease !important;
        margin: 0.25rem 0 !important;
    }
    
    [data-testid="stSidebar"] .stRadio > div > label:hover {
        background: rgba(245, 196, 83, 0.05) !important;
        border-color: rgba(245, 196, 83, 0.3) !important;
    }
    
    [data-testid="stSidebar"] .stRadio > div > label[data-baseweb="radio"]:has(input:checked) {
        background: rgba(245, 196, 83, 0.1) !important;
        border-left: 3px solid #F5C453 !important;
        border-color: #F5C453 !important;
        color: #F9FAFB !important;
        font-weight: 600 !important;
    }
    
    /* Video */
    video {
        border-radius: 12px !important;
        border: 2px solid #1F2933 !important;
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
        transition: all 0.2s ease !important;
        height: 100%;
    }
    
    .team-card:hover {
        transform: translateY(-4px) !important;
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
@st.cache_resource(show_spinner=False)
def load_yolo_model():
    """Load YOLO model efficiently"""
    try:
        model = YOLO("./training_results/fruit_model_best.pt")
        model.to(DEVICE)
        model.fuse()  # Fuse layers for faster inference
        return model
    except Exception as e:
        st.error(f"Model loading error: {e}")
        return None

MODEL = load_yolo_model()

# ====================================================================
# OPTIMIZED VIDEO PROCESSOR
# ====================================================================
class YOLOv8VideoProcessor(VideoProcessorBase):
    def __init__(self):
        super().__init__()
        self.model = MODEL
        self.conf_threshold = 0.5
        
    def recv(self, frame):
        if self.model is None:
            return frame
            
        try:
            img = frame.to_ndarray(format="bgr24")
            # Optimized inference with minimal settings
            results = self.model(img, 
                               conf=self.conf_threshold,
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
                <h3 style='margin: 0; color: #F5C453;'>AI Vision System</h3>
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
                    <span style='color: #22C55E; font-weight: 600; font-size: 0.9rem;'>YOLOv8n</span>
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
    st.markdown("<h1>YOLOv8 Real-Time Object Detection</h1>", unsafe_allow_html=True)
    
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
        st.metric(label="Model", value="YOLOv8n", delta="COCO Pretrained")
    
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
                <span class='tech-badge'>YOLOv8</span>
                <span class='tech-badge'>Streamlit</span>
                <span class='tech-badge'>PyTorch</span>
                <span class='tech-badge'>OpenCV</span>
                <span class='tech-badge'>WebRTC</span>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # 4. Methodology
    st.markdown("<h2>Methodology</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
            <div class='dark-card'>
                <h3>Dataset & Model</h3>
                <ul>
                    <li><strong>Dataset:</strong> Frute-262 dataset</li>
                    <li><strong>Selected Classes:</strong> 12 fruit classes</li>
                    <li><strong>Samples:</strong> 100 images per class</li>
                    <li><strong>Total:</strong> 1,200 images sampled</li>
                    <li><strong>Model:</strong> YOLOv8 Nano</li>
                    <li><strong>Pre-trained:</strong> COCO dataset weights</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class='dark-card'>
                <h3>Deployment & Tools</h3>
                <ul>
                    <li><strong>Framework:</strong> Ultralytics YOLOv8</li>
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
                <p>Lead Developer</p>
            </div>
        """, unsafe_allow_html=True)
    
    with team_col2:
        st.markdown("""
            <div class='team-card'>
                <span class='team-emoji'>👨‍🔬</span>
                <h4>Ramazan YILDIZ</h4>
                <p>Vision Engineer</p>
            </div>
        """, unsafe_allow_html=True)
    
    with team_col3:
        st.markdown("""
            <div class='team-card'>
                <span class='team-emoji'>👩‍💼</span>
                <h4>Beyza GÜLER</h4>
                <p>Data Specialist</p>
            </div>
        """, unsafe_allow_html=True)
    
    # 6. Project Roadmap
    st.markdown("<h2>Project Roadmap</h2>", unsafe_allow_html=True)
    
    roadmap_col1, roadmap_col2 = st.columns(2)
    
    with roadmap_col1:
        st.markdown("""
            <div class='dark-card'>
                <h3>✅ Phase 1 (Current)</h3>
                <ul>
                    <li>Dataset sampling completed</li>
                    <li>Prototype system deployed</li>
                    <li>Real-time detection working</li>
                    <li>Web interface established</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    with roadmap_col2:
        st.markdown("""
            <div class='dark-card'>
                <h3>⏳ Phase 2 (Next)</h3>
                <ul>
                    <li>Image annotation & labeling</li>
                    <li>Model fine-tuning on dataset</li>
                    <li>Quantitative evaluation</li>
                    <li>Performance optimization</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    # 7. Call to Action
    st.markdown("""
        <div class='dark-card' style='text-align: center; border-color: #F5C453;'>
            <h3 style='margin-bottom: 1rem;'>🚀 Ready to Experience AI Vision?</h3>
            <p style='margin-bottom: 1.5rem;'>
                Activate your webcam and see YOLOv8 in real-time action
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
                    <li>People, animals, vehicles</li>
                    <li>Furniture, electronics</li>
                    <li>Food items, bottles</li>
                    <li>Sports equipment</li>
                    <li>80+ object classes total</li>
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
    # Initialize session state
    if 'page_selection' not in st.session_state:
        st.session_state.page_selection = "🏠 Overview"
    
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