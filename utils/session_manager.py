import json
import os
import time

class SessionManager:
    def __init__(self, filepath="sessions.json"):
        self.filepath = filepath
        if not os.path.exists(filepath):
            with open(filepath, 'w') as f:
                json.dump([], f)

    def save_session(self, patient_id, exercise, reps, avg_form_quality, duration):
        session = {
            "patient_id": patient_id,
            "exercise": exercise,
            "reps": reps,
            "avg_form_quality": avg_form_quality,
            "duration": duration,
            "start_time": time.time() - duration
        }
        with open(self.filepath, 'r+') as f:
            data = json.load(f)
            data.append(session)
            f.seek(0)
            json.dump(data, f, indent=2)

    def load_all_sessions(self):
        try:
            with open(self.filepath, 'r') as f:
                return json.load(f)
        except:
            return []
