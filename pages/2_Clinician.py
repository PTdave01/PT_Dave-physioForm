import streamlit as st
import pandas as pd
import sys
sys.path.insert(0, "..")
from utils.session_manager import SessionManager

st.set_page_config(page_title="Clinician Dashboard – PhysioForm", layout="wide")
st.title("👨‍⚕️ Clinician Dashboard")

session_manager = SessionManager()
sessions = session_manager.load_all_sessions()

if not sessions:
    st.info("No sessions recorded yet.")
else:
    df = pd.DataFrame(sessions)
    df['timestamp'] = pd.to_datetime(df['start_time'], unit='s')
    df['date'] = df['timestamp'].dt.date

    patients = df['patient_id'].unique()
    selected_patient = st.selectbox("Select Patient", ["All"] + list(patients))

    if selected_patient != "All":
        df = df[df['patient_id'] == selected_patient]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Sessions", len(df))
    col2.metric("Avg Reps/Session", round(df['reps'].mean(), 1) if not df.empty else 0)
    col3.metric("Avg Form Quality", f"{round(df['avg_form_quality'].mean()*100, 1)}%" if not df.empty else "N/A")
    col4.metric("Patients Active", df['patient_id'].nunique())

    st.subheader("Session Log")
    st.dataframe(df[['patient_id', 'exercise', 'reps', 'avg_form_quality', 'duration', 'date']])

    st.subheader("Form Quality Trend")
    if not df.empty:
        quality_df = df.groupby('date')['avg_form_quality'].mean().reset_index()
        st.line_chart(quality_df.set_index('date'))
