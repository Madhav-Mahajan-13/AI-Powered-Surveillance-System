# main_app.py
import streamlit as st
import cv2
import torch
import numpy as np
from collections import defaultdict
import tempfile

# Import your custom modules
from sort_tracker import Sort
from logic_engine import SuspiciousActivityLogic
from drawing_utils import draw_tracked_objects, draw_loitering_zone

# --- Configuration ---
st.set_page_config(page_title="AI Surveillance Dashboard", layout="wide")
st.title("🚨 AI-Powered Real-Time Surveillance Dashboard")

# --- Model Loading ---
@st.cache_resource
def load_model():
    """Loads the YOLOv5 model from PyTorch Hub."""
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    return model

model = load_model()
tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3)

# --- UI Sidebar ---
st.sidebar.header("Configuration")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
abandonment_time = st.sidebar.slider("Abandonment Time (seconds)", 5, 60, 10)
proximity_radius = st.sidebar.slider("Proximity Radius (pixels)", 50, 300, 150)
loitering_time = st.sidebar.slider("Loitering Time (seconds)", 10, 120, 20)

# --- Video Source ---
source_option = st.sidebar.radio("Select Video Source", ["Upload a video", "Use sample video"])

video_file = None
if source_option == "Upload a video":
    uploaded_file = st.sidebar.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        # Use a temporary file to handle the uploaded video
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_file.read())
        video_file = tfile.name
else:
    video_file = "videos/example_video.mp4" # Make sure this path is correct

# --- Main Application Logic ---
if video_file:
    cap = cv2.VideoCapture(video_file)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Convert time thresholds to frame counts
    abandonment_threshold_frames = int(abandonment_time * fps)
    loitering_threshold_frames = int(loitering_time * fps)

    # Initialize logic engine with UI parameters
    logic_engine = SuspiciousActivityLogic(
        abandonment_threshold_frames=abandonment_threshold_frames,
        proximity_threshold_pixels=proximity_radius,
        loitering_threshold_frames=loitering_threshold_frames
    )

    # Define a static loitering zone for demonstration
    # In a real app, this could be drawn by the user
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    loitering_zone = np.array([[int(frame_width * 0.2), int(frame_height * 0.3)],
                               [int(frame_width * 0.8), int(frame_height * 0.3)],
                               [int(frame_width * 0.8), int(frame_height * 0.9)],
                               [int(frame_width * 0.2), int(frame_height * 0.9)]], np.int32)

    # --- Streamlit Layout ---
    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader("Live Video Feed")
        frame_placeholder = st.empty()
    with col2:
        st.subheader("Real-Time Alerts")
        alerts_placeholder = st.empty()

    # --- State Management ---
    if 'alerts' not in st.session_state:
        st.session_state.alerts = []

    class_names_map = {} # To map track_id to class_name

    # Process video frame by frame
    frame_count = 0
    max_frames = 1000  # Limit processing for demo purposes
    
    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            st.write("Video processing finished.")
            break

        frame_count += 1

        # 1. Detection
        results = model(frame)
        detections_df = results.pandas().xyxy[0]  # Get first image results
        detections_df = detections_df[detections_df['confidence'] > confidence_threshold]
        
        # 2. Tracking
        if len(detections_df) > 0:
            detections_for_sort = detections_df[['xmin', 'ymin', 'xmax', 'ymax', 'confidence']].values
            tracked_objects = tracker.update(detections_for_sort)
        else:
            tracked_objects = tracker.update(np.empty((0, 5)))

        # Map class names to tracker IDs
        for _, row in detections_df.iterrows():
            x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            det_center = ((x1 + x2) // 2, (y1 + y2) // 2)
            
            for tobj in tracked_objects:
                tx1, ty1, tx2, ty2, tid = map(int, tobj)
                track_center = ((tx1 + tx2) // 2, (ty1 + ty2) // 2)
                
                # Check if detection and track are close enough to be the same object
                distance = np.sqrt((det_center[0] - track_center[0])**2 + (det_center[1] - track_center[1])**2)
                if distance < 50:  # Threshold for matching detection to track
                    class_names_map[tid] = row['name']
                    break

        # 3. State Analysis
        logic_engine.update_states(tracked_objects, class_names_map)
        
        abandoned_alerts, abandoned_ids = logic_engine.check_abandoned_objects()
        loitering_alerts, loitering_ids = logic_engine.check_loitering(loitering_zone)

        # Update alerts list
        for alert in abandoned_alerts + loitering_alerts:
            if alert not in st.session_state.alerts:
                st.session_state.alerts.insert(0, alert)
                # Keep only last 10 alerts
                if len(st.session_state.alerts) > 10:
                    st.session_state.alerts = st.session_state.alerts[:10]

        # 4. Visualization
        frame = draw_loitering_zone(frame, loitering_zone)
        frame = draw_tracked_objects(frame, tracked_objects, class_names_map, logic_engine, abandoned_ids, loitering_ids)
        
        # 5. Display (update every 5 frames to improve performance)
        if frame_count % 5 == 0:
            frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
            
            with alerts_placeholder.container():
                if st.session_state.alerts:
                    for i, alert in enumerate(st.session_state.alerts[:5]):  # Show only last 5 alerts
                        st.warning(f"Alert {i+1}: {alert}")
                else:
                    st.info("No alerts detected")

    cap.release()
else:
    st.info("Please select a video source from the sidebar to begin.")