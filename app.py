import streamlit as st

st.set_page_config(page_title="PhysioForm", page_icon="🩺", layout="wide")
st.title("PhysioForm – AI-Powered Home Physiotherapy")
st.markdown("**Real-time exercise tracking with YOLO pose estimation**")
col1, col2 = st.columns(2)
with col1:
    st.page_link("pages/1_Patient.py", label="🧑‍⚕️ Patient – Start Exercise", icon="🧑‍⚕️")
with col2:
    st.page_link("pages/2_Clinician.py", label="👨‍⚕️ Clinician – View Dashboard", icon="👨‍⚕️")
