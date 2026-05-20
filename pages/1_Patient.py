import streamlit as st
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode, RTCConfiguration
import numpy as np
import time
import threading
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.pose_estimator import PoseEstimator
from utils.exercise_analyzer import ExerciseAnalyzer
from utils.session_manager import SessionManager

st.set_page_config(page_title="Patient – PhysioForm", layout="wide")
st.title("🧑‍⚕️ Patient Exercise Session (Live Stream)")

patient_id = st.text_input("Enter your Patient ID", value="patient_001")
exercise_choice = st.selectbox("Choose your exercise (or auto-detect)", ["Auto-detect", "Biceps Curl", "Squat"])

if "recognized_exercise" not in st.session_state:
    st.session_state.recognized_exercise = None
if "rep_count" not in st.session_state:
    st.session_state.rep_count = 0

pose_estimator = PoseEstimator()
analyzer = ExerciseAnalyzer()
session_manager = SessionManager()

class PhysioVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.pose = pose_estimator
        self.analyzer = analyzer
        self.lock = threading.Lock()
        self.start_time = time.time()
        # Fresh rep_state dict – no old keys
        self.rep_state = {}
        self.rep_count = 0
        self.form_quality_history = []
        self.exercise = None
        self.feedback_text = ""
        self.angle_buffer = []

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        try:
            img = frame.to_ndarray(format="bgr24")
            keypoints, bbox = self.pose.get_keypoints(img)

            with self.lock:
                if (exercise_choice == "Auto-detect" and self.exercise is None) or st.session_state.get("exercise_override"):
                    if len(self.angle_buffer) < 60:
                        if keypoints is not None:
                            angles = self.analyzer.calc_all_angles(keypoints)
                            self.angle_buffer.append(angles)
                    else:
                        self.exercise = self.analyzer.recognize_exercise(self.angle_buffer)
                        st.session_state.recognized_exercise = self.exercise
                        self.angle_buffer.clear()
                else:
                    self.exercise = exercise_choice if exercise_choice != "Auto-detect" else self.exercise

                if self.exercise and keypoints is not None:
                    feedback, color_guide, rep_done = self.analyzer.evaluate_form(
                        self.exercise, keypoints, self.rep_state
                    )
                    self.feedback_text = feedback
                    if rep_done:
                        self.rep_count += 1
                        quality = self.analyzer.get_rep_quality(self.feedback_text)
                        self.form_quality_history.append(quality)
                        st.session_state.rep_count = self.rep_count
                    img = self.pose.draw_skeleton(img, keypoints, color_guide)
                    img = self.analyzer.draw_feedback(img, self.feedback_text, self.rep_count, self.exercise)
                else:
                    img = self.pose.draw_text(img, "Position yourself in frame", (50, 50))
                    if self.exercise is None and exercise_choice == "Auto-detect":
                        img = self.pose.draw_text(img, "Auto-detecting exercise...", (50, 90))

            return av.VideoFrame.from_ndarray(img, format="bgr24")
        except Exception as e:
            img = frame.to_ndarray(format="bgr24")
            img = self.pose.draw_text(img, f"Error: {str(e)[:50]}", (50, 50))
            return av.VideoFrame.from_ndarray(img, format="bgr24")

webrtc_ctx = webrtc_streamer(
    key="physio-camera",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTCConfiguration(
        {"iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["turn:freestun.net:3478"], "username": "free", "credential": "free"},
            {"urls": ["turn:openrelay.metered.ca:443?transport=tcp"],
             "username": "openrelayproject", "credential": "openrelayproject"},
            {"urls": ["turn:34.203.254.2:3478"], "username": "test", "credential": "test"}
        ]}
    ),
    media_stream_constraints={
        "video": {"width": 320, "height": 240, "frameRate": 15},
        "audio": False
    },
    video_processor_factory=PhysioVideoProcessor,
    async_processing=True,
)

col1, col2 = st.columns(2)
with col1:
    st.metric("Reps Completed", st.session_state.rep_count)
with col2:
    if st.session_state.recognized_exercise:
        st.metric("Exercise", st.session_state.recognized_exercise)
    else:
        st.metric("Exercise", "Detecting...")

if st.button("End Session & Save"):
    end_time = time.time()
    duration = end_time - st.session_state.get("start_time", time.time())
    if webrtc_ctx.video_processor:
        processor = webrtc_ctx.video_processor
        avg_quality = np.mean(processor.form_quality_history) if processor.form_quality_history else 0.0
        session_manager.save_session(
            patient_id=patient_id,
            exercise=st.session_state.recognized_exercise or "unknown",
            reps=processor.rep_count,
            avg_form_quality=avg_quality,
            duration=duration
        )
    st.success("Session saved! Clinician can now view your progress.")
    st.session_state.rep_count = 0
    st.session_state.recognized_exercise = None
