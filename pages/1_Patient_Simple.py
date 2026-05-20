import streamlit as st
import cv2
import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.pose_estimator import PoseEstimator
from utils.exercise_analyzer import ExerciseAnalyzer
from utils.session_manager import SessionManager

st.set_page_config(page_title="Patient (Simple) – PhysioForm", layout="wide")
st.title("🧑‍⚕️ Patient Exercise Session")

patient_id = st.text_input("Enter your Patient ID", value="patient_001")
exercise_choice = st.selectbox("Choose your exercise", ["Biceps Curl", "Squat"])

pose = PoseEstimator()
analyzer = ExerciseAnalyzer()
session_mgr = SessionManager()

# Initialize session state
if "running" not in st.session_state:
    st.session_state.running = False
if "rep_count" not in st.session_state:
    st.session_state.rep_count = 0
if "rep_state" not in st.session_state:
    st.session_state.rep_state = {"curl": "down", "squat": "up"}

col1, col2 = st.columns(2)
start_stop = col1.button("Start" if not st.session_state.running else "Stop")
if start_stop:
    st.session_state.running = not st.session_state.running
    if st.session_state.running:
        st.session_state.rep_count = 0
        st.session_state.rep_state = {"curl": "down", "squat": "up"}

frame_placeholder = st.empty()
rep_metric = st.empty()

if st.session_state.running:
    # Camera input widget (native browser camera – works on any network)
    camera_image = st.camera_input("📸 Point your camera", key="live_cam")
    
    if camera_image is not None:
        # Convert to OpenCV image
        bytes_data = camera_image.getvalue()
        nparr = np.frombuffer(bytes_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Run pose estimation
        keypoints, bbox = pose.get_keypoints(img)

        if keypoints is not None:
            feedback, color_guide, rep_done = analyzer.evaluate_form(
                exercise_choice, keypoints, st.session_state.rep_state
            )
            if rep_done:
                st.session_state.rep_count += 1
            img = pose.draw_skeleton(img, keypoints, color_guide)
            img = analyzer.draw_feedback(img, feedback, st.session_state.rep_count, exercise_choice)
        else:
            img = pose.draw_text(img, "No person detected", (50, 50))

        # Display processed image
        frame_placeholder.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), channels="RGB")

        # Update rep metric
        rep_metric.metric("Reps Completed", st.session_state.rep_count)

        # Auto-refresh after a short delay (by causing a rerun)
        time.sleep(0.2)
        st.rerun()

    else:
        frame_placeholder.info("Tap 'Take Photo' to capture a frame.")
else:
    frame_placeholder.info("Press 'Start' to begin your exercise session.")
    rep_metric.metric("Reps Completed", 0)

# Save session
if st.button("End Session & Save") and st.session_state.running:
    # Simulate a duration (since we don't track exact time)
    duration = 60  # approximate
    session_mgr.save_session(patient_id, exercise_choice, st.session_state.rep_count, 0.8, duration)
    st.success("Session saved! Clinician can now view your progress.")
    st.session_state.running = False
    st.session_state.rep_count = 0
