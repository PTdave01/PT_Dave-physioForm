import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils.session_manager import SessionManager
from datetime import datetime

st.set_page_config(page_title="Clinician Dashboard – PhysioForm", layout="wide")

# ─── Authentication ─────────────────────────────────────────────────
if "clinician_auth" not in st.session_state:
    st.session_state.clinician_auth = False

if not st.session_state.clinician_auth:
    st.title("🔐 Clinician Login")
    password = st.text_input("Enter password", type="password")
    if st.button("Login"):
        correct_password = st.secrets.get("CLINICIAN_PASSWORD", "admin")
        if password == correct_password:
            st.session_state.clinician_auth = True
            st.rerun()
        else:
            st.error("Wrong password")
    st.stop()

# Logout button
st.sidebar.button("🔒 Logout", on_click=lambda: st.session_state.update({"clinician_auth": False}))

# ─── Dashboard ─────────────────────────────────────────────────────
st.title("👨‍⚕️ Clinician Dashboard")

session_manager = SessionManager()
sessions = session_manager.load_all_sessions()

if not sessions:
    st.info("No sessions recorded yet. Ask a patient to complete an exercise session.")
else:
    df = pd.DataFrame(sessions)
    df['timestamp'] = pd.to_datetime(df['start_time'], unit='s')
    df['date'] = df['timestamp'].dt.date
    df['week'] = df['timestamp'].dt.isocalendar().week

    # ---- Sidebar filters ----
    st.sidebar.header("Filters")
    patients = df['patient_id'].unique()
    selected_patient = st.sidebar.selectbox("Select Patient", ["All"] + list(patients))
    exercise_types = df['exercise'].unique()
    selected_exercise = st.sidebar.selectbox("Select Exercise", ["All"] + list(exercise_types))

    mask = pd.Series(True, index=df.index)
    if selected_patient != "All":
        mask &= df['patient_id'] == selected_patient
    if selected_exercise != "All":
        mask &= df['exercise'] == selected_exercise
    filtered_df = df[mask]

    # ---- Top KPI cards ----
    st.subheader("Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Sessions", len(filtered_df))
    col2.metric("Avg Reps / Session", round(filtered_df['reps'].mean(), 1) if not filtered_df.empty else 0)
    avg_quality = filtered_df['avg_form_quality'].mean() * 100 if not filtered_df.empty else 0
    col3.metric("Avg Form Quality", f"{avg_quality:.1f}%")
    col4.metric("Unique Patients", filtered_df['patient_id'].nunique())

    # ---- Charts ----
    st.subheader("Progress Over Time")
    if not filtered_df.empty:
        # Average Form Quality over time
        quality_over_time = filtered_df.groupby('date')['avg_form_quality'].mean().reset_index()
        quality_over_time['avg_form_quality'] *= 100
        fig1 = px.line(quality_over_time, x='date', y='avg_form_quality',
                       title='Average Form Quality (%)', markers=True)
        fig1.update_yaxes(range=[0, 100])
        fig1.update_layout(height=350)
        st.plotly_chart(fig1, use_container_width=True)

        # Total Reps per Day
        reps_over_time = filtered_df.groupby('date')['reps'].sum().reset_index()
        fig2 = px.bar(reps_over_time, x='date', y='reps',
                      title='Total Reps per Day', color='reps', color_continuous_scale='blues')
        fig2.update_layout(height=350)
        st.plotly_chart(fig2, use_container_width=True)

        # Form Quality by Exercise
        st.subheader("Form Quality by Exercise")
        quality_by_exercise = filtered_df.groupby('exercise')['avg_form_quality'].mean().reset_index()
        quality_by_exercise['avg_form_quality'] *= 100
        fig3 = px.bar(quality_by_exercise, x='exercise', y='avg_form_quality',
                      color='exercise', title='Average Form Quality per Exercise', text_auto='.1f')
        fig3.update_layout(height=400)
        st.plotly_chart(fig3, use_container_width=True)

        # Patient Activity Heatmap
        st.subheader("Patient Activity Heatmap")
        heatmap_data = filtered_df.groupby(['patient_id', 'week']).size().reset_index(name='sessions')
        fig4 = px.density_heatmap(heatmap_data, x='week', y='patient_id', z='sessions',
                                  color_continuous_scale='greens', title='Sessions per Week per Patient')
        fig4.update_layout(height=400)
        st.plotly_chart(fig4, use_container_width=True)
    else:
        st.warning("No data with current filters.")

    # Raw data table
    st.subheader("Session Log")
    st.dataframe(filtered_df[['patient_id', 'exercise', 'reps', 'avg_form_quality', 'duration', 'date']],
                 use_container_width=True)
