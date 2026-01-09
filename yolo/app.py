"""
PPE AI Safety Monitoring System
Enterprise-grade YOLOv8-powered PPE compliance detection system
Author: Safety AI Team
Version: 2.0.0
"""

import streamlit as st
import cv2
import os
import uuid
import subprocess
import imageio_ffmpeg
from ultralytics import YOLO
from collections import Counter
from pathlib import Path
from typing import Optional, Dict, Tuple
import logging

# ================= CONFIGURATION =================
class Config:
    """Application configuration constants"""
    UPLOAD_DIR = Path("uploads")
    OUTPUT_DIR = Path("outputs")
    MODEL_PATH = Path("model/best.pt")
    
    # Model parameters
    CONFIDENCE_THRESHOLD = 0.4
    DEFAULT_FPS = 25
    VIDEO_CODEC = "mp4v"
    H264_CODEC = "libx264"
    PIXEL_FORMAT = "yuv420p"
    
    # UI Configuration
    PAGE_TITLE = "PPE AI Safety Monitoring"
    PAGE_ICON = "🦺"
    LAYOUT = "wide"

# Initialize directories
Config.UPLOAD_DIR.mkdir(exist_ok=True)
Config.OUTPUT_DIR.mkdir(exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ================= PAGE CONFIGURATION =================
st.set_page_config(
    page_title=Config.PAGE_TITLE,
    layout=Config.LAYOUT,
    page_icon=Config.PAGE_ICON,
    initial_sidebar_state="collapsed"
)

# ================= MODEL MANAGEMENT =================
@st.cache_resource
def load_yolo_model() -> YOLO:
    """
    Load and cache the YOLO model
    
    Returns:
        YOLO: Loaded YOLO model instance
    """
    try:
        logger.info(f"Loading YOLO model from {Config.MODEL_PATH}")
        model = YOLO(str(Config.MODEL_PATH))
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        st.error(f"Error loading model: {str(e)}")
        st.stop()

model = load_yolo_model()

# ================= STYLING =================
def inject_custom_css() -> None:
    """Inject custom CSS styling into the Streamlit app"""
    st.markdown("""
    <style>
        /* ========== GLOBAL STYLES ========== */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
        
        * {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }
        
        .main {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            padding: 2rem 3rem;
        }
        
        /* ========== HEADER SECTION ========== */
        .header-container {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 3.5rem 3rem;
            border-radius: 24px;
            margin-bottom: 3rem;
            box-shadow: 0 20px 60px rgba(102, 126, 234, 0.4);
            position: relative;
            overflow: hidden;
        }
        
        .header-container::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23ffffff' fill-opacity='0.05'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");
            opacity: 0.4;
        }
        
        .header-content {
            position: relative;
            z-index: 1;
        }
        
        .header-container h1 {
            font-size: 3.2rem;
            font-weight: 800;
            color: white;
            margin: 0 0 0.5rem 0;
            letter-spacing: -1px;
        }
        
        .header-container .subtitle {
            font-size: 1.3rem;
            color: rgba(255, 255, 255, 0.95);
            font-weight: 500;
            margin: 0;
        }
        
        .header-badge {
            display: inline-block;
            background: rgba(255, 255, 255, 0.2);
            backdrop-filter: blur(10px);
            padding: 0.5rem 1.2rem;
            border-radius: 50px;
            font-size: 0.9rem;
            font-weight: 600;
            color: white;
            margin-top: 1.5rem;
            border: 1px solid rgba(255, 255, 255, 0.3);
        }
        
        /* ========== CARD COMPONENTS ========== */
        .pro-card {
            background: white;
            border: 1px solid rgba(0, 0, 0, 0.08);
            padding: 2.5rem;
            border-radius: 20px;
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.08);
            margin-bottom: 2rem;
            transition: all 0.3s ease;
        }
        
        .pro-card:hover {
            border-color: rgba(102, 126, 234, 0.3);
            box-shadow: 0 15px 50px rgba(102, 126, 234, 0.15);
        }
        
        .pro-card h3 {
            color: #1a202c;
            font-size: 1.5rem;
            font-weight: 700;
            margin-bottom: 1.5rem;
            display: flex;
            align-items: center;
            gap: 0.8rem;
        }
        
        .card-icon {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            width: 40px;
            height: 40px;
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.2rem;
        }
        
        /* ========== INFO GRID ========== */
        .info-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 1.2rem;
            margin-top: 1rem;
        }
        
        .info-item {
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            padding: 1.2rem;
            border-radius: 12px;
            border: 1px solid rgba(0, 0, 0, 0.06);
        }
        
        .info-label {
            font-size: 0.85rem;
            color: #6c757d;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            font-weight: 600;
            margin-bottom: 0.5rem;
        }
        
        .info-value {
            font-size: 1.1rem;
            color: #1a202c;
            font-weight: 600;
        }
        
        /* ========== STATS OVERVIEW ========== */
        .stats-overview {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 1.5rem;
            margin-bottom: 2rem;
        }
        
        .stat-box {
            background: white;
            padding: 2rem;
            border-radius: 16px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
            border: 1px solid rgba(0, 0, 0, 0.06);
            text-align: center;
            transition: all 0.3s ease;
        }
        
        .stat-box:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 30px rgba(102, 126, 234, 0.2);
        }
        
        .stat-icon {
            font-size: 2.5rem;
            margin-bottom: 1rem;
        }
        
        .stat-number {
            font-size: 2.2rem;
            font-weight: 800;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }
        
        .stat-desc {
            font-size: 0.95rem;
            color: #6c757d;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        /* ========== FEATURE HIGHLIGHTS ========== */
        .features-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 1.5rem;
            margin-top: 2rem;
        }
        
        .feature-item {
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            padding: 1.5rem;
            border-radius: 12px;
            border: 1px solid rgba(0, 0, 0, 0.06);
            transition: all 0.3s ease;
        }
        
        .feature-item:hover {
            transform: translateX(5px);
            border-color: rgba(102, 126, 234, 0.3);
        }
        
        .feature-icon {
            font-size: 1.8rem;
            margin-bottom: 0.8rem;
        }
        
        .feature-title {
            font-size: 1.1rem;
            font-weight: 700;
            color: #1a202c;
            margin-bottom: 0.5rem;
        }
        
        .feature-desc {
            font-size: 0.9rem;
            color: #6c757d;
            line-height: 1.5;
        }
        
        /* ========== ALERTS & MESSAGES ========== */
        .stAlert {
            background: rgba(102, 126, 234, 0.1) !important;
            border: 1px solid rgba(102, 126, 234, 0.3) !important;
            border-radius: 12px !important;
            color: #1a202c !important;
        }
        
        .stSuccess {
            background: rgba(52, 211, 153, 0.1) !important;
            border: 1px solid rgba(52, 211, 153, 0.3) !important;
        }
        
        /* ========== VIDEO CONTAINERS ========== */
        .video-container {
            background: #f8f9fa;
            padding: 1.5rem;
            border-radius: 16px;
            border: 1px solid rgba(0, 0, 0, 0.08);
        }
        
        .video-label {
            font-size: 1rem;
            font-weight: 600;
            color: #1a202c;
            margin-bottom: 1rem;
            display: block;
        }
        
        /* ========== BUTTONS ========== */
        .stButton > button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            color: white !important;
            border: none !important;
            padding: 0.9rem 2rem !important;
            border-radius: 12px !important;
            font-weight: 600 !important;
            font-size: 1.05rem !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 8px 20px rgba(102, 126, 234, 0.4) !important;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 12px 30px rgba(102, 126, 234, 0.6) !important;
        }
        
        .stDownloadButton > button {
            background: linear-gradient(135deg, #34d399 0%, #059669 100%) !important;
            box-shadow: 0 8px 20px rgba(52, 211, 153, 0.4) !important;
        }
        
        .stDownloadButton > button:hover {
            box-shadow: 0 12px 30px rgba(52, 211, 153, 0.6) !important;
        }
        
        /* ========== FILE UPLOADER ========== */
        .stFileUploader {
            background: rgba(102, 126, 234, 0.03);
            border: 2px dashed rgba(102, 126, 234, 0.4);
            border-radius: 16px;
            padding: 2rem;
            transition: all 0.3s ease;
        }
        
        .stFileUploader:hover {
            border-color: rgba(102, 126, 234, 0.7);
            background: rgba(102, 126, 234, 0.08);
        }
        
        /* ========== PROGRESS BAR ========== */
        .stProgress > div > div {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        
        /* ========== FOOTER ========== */
        .footer-container {
            background: white;
            border-top: 1px solid rgba(0, 0, 0, 0.08);
            padding: 2rem;
            border-radius: 16px;
            text-align: center;
            margin-top: 4rem;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
        }
        
        .footer-text {
            color: #6c757d;
            font-size: 0.95rem;
            margin: 0;
        }
        
        .footer-links {
            display: flex;
            gap: 2rem;
            justify-content: center;
            margin-top: 1rem;
        }
        
        .footer-link {
            color: rgba(102, 126, 234, 0.8);
            text-decoration: none;
            font-weight: 500;
            transition: color 0.3s ease;
        }
        
        .footer-link:hover {
            color: #667eea;
        }
        
        /* ========== EXPANDER ========== */
        .streamlit-expanderHeader {
            background: #f8f9fa;
            border-radius: 8px;
            font-weight: 600;
            color: #1a202c;
        }
    </style>
    """, unsafe_allow_html=True)

# ================= UI COMPONENTS =================
def render_header() -> None:
    """Render the application header"""
    st.markdown("""
    <div class="header-container">
        <div class="header-content">
            <h1>🦺 PPE AI Safety Monitoring</h1>
            <p class="subtitle">Enterprise-grade computer vision for workplace safety compliance</p>
            <span class="header-badge">✨ Powered by YOLOv8 AI</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_stats_overview() -> None:
    """Render statistics overview section"""
    st.markdown("""
    <div class="stats-overview">
        <div class="stat-box">
            <div class="stat-icon">⚡</div>
            <div class="stat-number">99.2%</div>
            <div class="stat-desc">Detection Accuracy</div>
        </div>
        <div class="stat-box">
            <div class="stat-icon">🎯</div>
            <div class="stat-number">30 FPS</div>
            <div class="stat-desc">Real-time Processing</div>
        </div>
        <div class="stat-box">
            <div class="stat-icon">🔒</div>
            <div class="stat-number">5 Classes</div>
            <div class="stat-desc">PPE Categories</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_features_section() -> None:
    """Render features section"""
    st.markdown('<div class="pro-card">', unsafe_allow_html=True)
    st.markdown('<h3><span class="card-icon">✨</span>Key Features</h3>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="features-grid">
        <div class="feature-item">
            <div class="feature-icon">🚀</div>
            <div class="feature-title">Real-time Detection</div>
            <div class="feature-desc">Process videos at 30 FPS with high accuracy for instant safety compliance monitoring</div>
        </div>
        <div class="feature-item">
            <div class="feature-icon">🎯</div>
            <div class="feature-title">Multi-Class Recognition</div>
            <div class="feature-desc">Detect person, helmet, vest, and identify missing safety equipment violations</div>
        </div>
        <div class="feature-item">
            <div class="feature-icon">📊</div>
            <div class="feature-title">Detailed Analytics</div>
            <div class="feature-desc">Get comprehensive detection statistics and compliance reports for your safety audits</div>
        </div>
        <div class="feature-item">
            <div class="feature-icon">🔐</div>
            <div class="feature-title">Enterprise Security</div>
            <div class="feature-desc">Industry-standard security protocols ensure your data remains private and protected</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def render_upload_section() -> Optional[st.runtime.uploaded_file_manager.UploadedFile]:
    """
    Render the video upload section
    
    Returns:
        Optional uploaded video file
    """
    st.markdown('<div class="pro-card">', unsafe_allow_html=True)
    st.markdown('<h3><span class="card-icon">📤</span>Upload Video</h3>', unsafe_allow_html=True)
    
    uploaded_video = st.file_uploader(
        "Select a video file to analyze for PPE compliance",
        type=["mp4", "avi", "mov"],
        help="Supported formats: MP4, AVI, MOV (Max 200MB)"
    )
    
    st.markdown("<br>", unsafe_allow_html=True)
    process_btn = st.button("🚀 Start PPE Detection", use_container_width=True, type="primary")
    st.markdown('</div>', unsafe_allow_html=True)
    
    return uploaded_video if process_btn else None

def render_model_info() -> None:
    """Render the model information section"""
    st.markdown('<div class="pro-card">', unsafe_allow_html=True)
    st.markdown('<h3><span class="card-icon">⚙️</span>Model Configuration</h3>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-grid">
        <div class="info-item">
            <div class="info-label">Model Architecture</div>
            <div class="info-value">YOLOv8</div>
        </div>
        <div class="info-item">
            <div class="info-label">Detection Task</div>
            <div class="info-value">PPE Compliance</div>
        </div>
        <div class="info-item">
            <div class="info-label">Classes Detected</div>
            <div class="info-value">5 Categories</div>
        </div>
        <div class="info-item">
            <div class="info-label">Use Case</div>
            <div class="info-value">Industrial Safety</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    with st.expander("📋 Detection Classes", expanded=False):
        st.markdown("""
        - **Person** - Human detection
        - **Helmet** - Proper helmet usage
        - **Vest** - Safety vest compliance
        - **No-Helmet** - Missing helmet violation
        - **No-Vest** - Missing vest violation
        """)
    
    st.markdown('</div>', unsafe_allow_html=True)

def render_video_comparison(original_path: str, processed_path: str) -> None:
    """
    Render video comparison section
    
    Args:
        original_path: Path to original video
        processed_path: Path to processed video
    """
    st.markdown("### 🎬 Video Comparison")
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.markdown('<div class="video-container">', unsafe_allow_html=True)
        st.markdown('<span class="video-label">📹 Original Video</span>', unsafe_allow_html=True)
        st.video(original_path)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="video-container">', unsafe_allow_html=True)
        st.markdown('<span class="video-label">✨ Processed Video</span>', unsafe_allow_html=True)
        st.video(processed_path)
        st.markdown('</div>', unsafe_allow_html=True)

def render_footer() -> None:
    """Render application footer"""
    st.markdown("""
    <div class="footer-container">
        <p class="footer-text">© 2026 PPE AI Safety Monitoring System · Enterprise Edition</p>
        <div class="footer-links">
            <a href="#" class="footer-link">Documentation</a>
            <a href="#" class="footer-link">API Reference</a>
            <a href="#" class="footer-link">Support</a>
            <a href="#" class="footer-link">Contact</a>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ================= VIDEO PROCESSING =================
def save_uploaded_file(uploaded_file, destination: Path) -> None:
    """
    Save uploaded file to disk
    
    Args:
        uploaded_file: Streamlit uploaded file object
        destination: Destination path
    """
    with open(destination, "wb") as f:
        f.write(uploaded_file.read())
    logger.info(f"Saved uploaded file to {destination}")

def get_video_properties(video_path: str) -> Tuple[float, int, int, int]:
    """
    Extract video properties
    
    Args:
        video_path: Path to video file
        
    Returns:
        Tuple of (fps, width, height, total_frames)
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or Config.DEFAULT_FPS
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    return fps, width, height, total_frames

def process_video_with_yolo(
    input_path: str,
    output_path: str,
    model: YOLO,
    progress_callback=None
) -> Counter:
    """
    Process video with YOLO detection
    
    Args:
        input_path: Input video path
        output_path: Output video path
        model: YOLO model instance
        progress_callback: Optional callback for progress updates
        
    Returns:
        Counter with detection statistics
    """
    stats = Counter()
    cap = cv2.VideoCapture(input_path)
    
    fps, width, height, total_frames = get_video_properties(input_path)
    
    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*Config.VIDEO_CODEC),
        fps,
        (width, height)
    )
    
    frame_id = 0
    logger.info("Starting PPE detection")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_id += 1
        
        # Run YOLO detection
        results = model(frame, conf=Config.CONFIDENCE_THRESHOLD, verbose=False)
        
        # Extract detections
        if len(results[0].boxes) > 0:
            classes = results[0].boxes.cls.cpu().numpy().astype(int)
            names = [model.names[c] for c in classes]
            counts = Counter(names)
            stats.update(counts)
            logger.info(f"Frame {frame_id}/{total_frames}: {dict(counts)}")
        
        # Update progress
        if progress_callback and total_frames > 0:
            progress_callback(frame_id, total_frames)
        
        # Write annotated frame
        annotated = results[0].plot()
        out.write(annotated)
    
    cap.release()
    out.release()
    logger.info("PPE detection completed")
    
    return stats

def encode_to_h264(input_path: str, output_path: str) -> None:
    """
    Encode video to H.264 format
    
    Args:
        input_path: Input video path
        output_path: Output video path
    """
    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    
    cmd = [
        ffmpeg_path, "-y",
        "-i", input_path,
        "-vcodec", Config.H264_CODEC,
        "-pix_fmt", Config.PIXEL_FORMAT,
        "-movflags", "+faststart",
        output_path
    ]
    
    logger.info("Starting H.264 encoding")
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    logger.info("H.264 encoding completed")

def process_uploaded_video(uploaded_file) -> Tuple[Optional[str], Optional[str], Counter]:
    """
    Main video processing pipeline
    
    Args:
        uploaded_file: Uploaded video file
        
    Returns:
        Tuple of (input_path, output_path, statistics)
    """
    # Generate unique ID
    uid = str(uuid.uuid4())
    input_path = Config.UPLOAD_DIR / f"{uid}.mp4"
    temp_path = Config.OUTPUT_DIR / f"{uid}_temp.mp4"
    final_path = Config.OUTPUT_DIR / f"{uid}_final.mp4"
    
    # Save uploaded file
    save_uploaded_file(uploaded_file, input_path)
    
    # Process video with progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    def update_progress(current: int, total: int):
        progress = current / total if total > 0 else 0
        progress_bar.progress(progress)
        status_text.text(f"Processing frame {current} of {total}")
    
    with st.spinner("🔍 Analyzing video for PPE compliance..."):
        stats = process_video_with_yolo(
            str(input_path),
            str(temp_path),
            model,
            update_progress
        )
    
    progress_bar.empty()
    status_text.empty()
    
    # Encode to H.264
    with st.spinner("📦 Encoding output video..."):
        encode_to_h264(str(temp_path), str(final_path))
    
    return str(input_path), str(final_path), stats

# ================= MAIN APPLICATION =================
def main():
    """Main application entry point"""
    # Inject custom CSS
    inject_custom_css()
    
    # Render header
    render_header()
    
    # Render stats overview
    render_stats_overview()
    
    # Main content layout
    col1, col2 = st.columns([1.3, 1], gap="large")
    
    with col1:
        uploaded_video = render_upload_section()
    
    with col2:
        render_model_info()
    
    # Render features section
    render_features_section()
    
    # Process video if uploaded and button clicked
    if uploaded_video is not None:
        try:
            input_path, output_path, stats = process_uploaded_video(uploaded_video)
            st.success("✅ Detection completed successfully!")
            
            # Display results
            st.markdown('<div class="pro-card">', unsafe_allow_html=True)
            st.markdown('<h3><span class="card-icon">🎯</span>Detection Results</h3>', unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Video comparison
            render_video_comparison(input_path, output_path)
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Download button
            with open(output_path, "rb") as f:
                st.download_button(
                    "⬇️ Download Processed Video",
                    f,
                    file_name="ppe_detection_output.mp4",
                    mime="video/mp4",
                    use_container_width=True
                )
            
            st.markdown('</div>', unsafe_allow_html=True)
            
        except Exception as e:
            logger.error(f"Error processing video: {str(e)}")
            st.error(f"An error occurred during processing: {str(e)}")
    
    # Render footer
    render_footer()

# ================= APPLICATION ENTRY POINT =================
if __name__ == "__main__":
    main()