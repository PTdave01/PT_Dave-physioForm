# PhysioForm – AI-Powered Home Physiotherapy

Real‑time exercise tracking using **YOLOv8 pose estimation** (COCO keypoints) and **Streamlit**.  
Designed for patients and clinicians to monitor biceps curls and squats from any smartphone browser.

## 📸 What It Does
- **Live webcam / camera processing** – works with both WebRTC streaming (Wi‑Fi) and simple photo‑based mode (mobile data).
- **Auto‑detection** of Biceps Curls and Squats (or manual selection).
- **Rep counting** with real‑time form correction (shoulder stability, knee tracking, back angle).
- **Colour‑coded skeleton overlay** – green for correct, yellow for marginal, red for error.
- **Clinician dashboard** – view all patient sessions, average form quality, reps, and adherence trends.

## 🚀 Live App
Access the app directly from your phone’s browser:  
'https://ptdave-physioform-ez35tc4j6tcjuqsppglqfc.streamlit.app'

## ⚙️ How to Use (Patient)
1. Open the app and tap **Patient – Start Exercise** (live stream) or **Patient (Simple)** (photo mode).
2. Enter your Patient ID (e.g., “patient_001”).
3. Choose an exercise or leave on **Auto‑detect**.
4. Allow camera access and position yourself so your full body is visible.
5. Start moving – the skeleton appears, angles are shown, and reps count automatically.
6. When finished, tap **End Session & Save** – your data is stored for the clinician.

## 🧑‍⚕️ Clinician Dashboard
1. From the home page, tap **Clinician – View Dashboard**.
2. View all sessions, filter by patient, see charts of form quality over time.

## 🧰 Tech Stack
- **YOLOv8n‑pose** (Ultralytics) – real‑time 17‑keypoint COCO skeleton
- **OpenCV** – angle calculation, drawing overlays
- **Streamlit** + **streamlit‑webrtc** – frontend and WebRTC integration
- **NumPy, Pandas** – data processing
- **JSON** – lightweight file‑based session storage

## 🖥️ Running Locally (for developers)
1. Clone the repo:
   ```bash
   git clone https://github.com/YOUR_USERNAME/physioform.git
   cd physioform
```

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the app:
   ```bash
   streamlit run app.py
   ```
3. Open http://localhost:8501 in your browser.

📁 Project Structure

```
physioform/
├── app.py                  # Home page
├── pages/
│   ├── 1_Patient.py        # Live WebRTC exercise session
│   ├── 1_Patient_Simple.py # Photo‑based fallback for mobile data
│   └── 2_Clinician.py      # Clinician dashboard
├── utils/
│   ├── __init__.py
│   ├── pose_estimator.py   # YOLO pose model loader & skeleton drawing
│   ├── exercise_analyzer.py # Angle, rep, form logic
│   └── session_manager.py  # JSON session storage
├── requirements.txt
├── packages.txt            # System dependencies for Streamlit Cloud
└── README.md
```
