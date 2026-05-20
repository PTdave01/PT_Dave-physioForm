import numpy as np
import cv2

class ExerciseAnalyzer:
    KP = {
        "left_shoulder": 5, "right_shoulder": 6,
        "left_elbow": 7, "right_elbow": 8,
        "left_wrist": 9, "right_wrist": 10,
        "left_hip": 11, "right_hip": 12,
        "left_knee": 13, "right_knee": 14,
        "left_ankle": 15, "right_ankle": 16
    }

    def calc_angle(self, a, b, c):
        a, b, c = np.array(a), np.array(b), np.array(c)
        ba = a - b
        bc = c - b
        cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
        return angle

    def calc_all_angles(self, keypoints):
        angles = {}
        if keypoints is None:
            return angles
        for side in ['left', 'right']:
            sh = self.KP[f"{side}_shoulder"]
            el = self.KP[f"{side}_elbow"]
            wr = self.KP[f"{side}_wrist"]
            if all(keypoints[i][0] > 0 for i in [sh, el, wr]):
                angles[f"{side}_elbow"] = self.calc_angle(keypoints[sh], keypoints[el], keypoints[wr])
        for side in ['left', 'right']:
            hip = self.KP[f"{side}_hip"]
            knee = self.KP[f"{side}_knee"]
            ank = self.KP[f"{side}_ankle"]
            if all(keypoints[i][0] > 0 for i in [hip, knee, ank]):
                angles[f"{side}_knee"] = self.calc_angle(keypoints[hip], keypoints[knee], keypoints[ank])
                if keypoints[self.KP[f"{side}_shoulder"]][0] > 0:
                    angles[f"{side}_hip"] = self.calc_angle(
                        keypoints[self.KP[f"{side}_shoulder"]], keypoints[hip], keypoints[knee]
                    )
        return angles

    def recognize_exercise(self, angle_buffer):
        elbow_vars, knee_vars = [], []
        for angles in angle_buffer:
            left_elbow = angles.get('left_elbow', None)
            right_elbow = angles.get('right_elbow', None)
            if left_elbow and right_elbow:
                elbow_vars.append((left_elbow + right_elbow) / 2)
            left_knee = angles.get('left_knee', None)
            right_knee = angles.get('right_knee', None)
            if left_knee and right_knee:
                knee_vars.append((left_knee + right_knee) / 2)
        if len(elbow_vars) > 10 and len(knee_vars) > 10:
            var_elbow = np.var(elbow_vars)
            var_knee = np.var(knee_vars)
            if var_elbow > var_knee * 2:
                return "Biceps Curl"
            elif var_knee > var_elbow * 2:
                return "Squat"
        return None

    def evaluate_form(self, exercise, keypoints, rep_state):
        feedback, color_guide, rep_done = "", {}, False
        if exercise == "Biceps Curl":
            feedback, color_guide, rep_done = self._curl_form(keypoints, rep_state)
        elif exercise == "Squat":
            feedback, color_guide, rep_done = self._squat_form(keypoints, rep_state)
        return feedback, color_guide, rep_done

    def _curl_form(self, kp, rep_state):
        feedback, color_guide, rep_done = "", {}, False
        # Make sure prev_curl is initialized
        if 'prev_curl' not in rep_state:
            rep_state['prev_curl'] = None

        for side in ['right', 'left']:
            sh = self.KP[f"{side}_shoulder"]
            el = self.KP[f"{side}_elbow"]
            wr = self.KP[f"{side}_wrist"]
            if all(kp[i][0] > 0 for i in [sh, el, wr]):
                angle = self.calc_angle(kp[sh], kp[el], kp[wr])

                # Widened thresholds
                DOWN_THRESH = 130   # arm nearly straight
                UP_THRESH = 70      # arm well curled

                if angle > DOWN_THRESH:
                    rep_state["prev_curl"] = "down"
                    feedback = "Arm extended"
                    color_guide[(sh, el)] = (0, 255, 0)
                    color_guide[(el, wr)] = (0, 255, 0)
                elif angle < UP_THRESH:
                    if rep_state.get("prev_curl") == "down":
                        rep_done = True
                    rep_state["prev_curl"] = "up"
                    feedback = "Curl up – good"
                    color_guide[(sh, el)] = (0, 255, 0)
                    color_guide[(el, wr)] = (0, 255, 0)
                else:
                    feedback = "Lower down" if rep_state.get("prev_curl") == "up" else "Curl up"
                    color_guide[(sh, el)] = (0, 255, 255)
                    color_guide[(el, wr)] = (0, 255, 255)

                # Show angle on screen
                elbow_pos = tuple(kp[el].astype(int))
                cv2.putText(kp, f"{int(angle)}", elbow_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

                # Shoulder check
                shoulder_y = kp[sh][1]
                hip_y = kp[self.KP[f"{side}_hip"]][1]
                if shoulder_y - hip_y > 20:
                    feedback += " | Keep shoulders steady!"
                    color_guide[(sh, el)] = (0, 0, 255)
                break
        return feedback, color_guide, rep_done

    def _squat_form(self, kp, rep_state):
        feedback, color_guide, rep_done = "", {}, False
        if 'prev_squat' not in rep_state:
            rep_state['prev_squat'] = None

        for side in ['right', 'left']:
            hip = self.KP[f"{side}_hip"]
            knee = self.KP[f"{side}_knee"]
            ank = self.KP[f"{side}_ankle"]
            sh = self.KP[f"{side}_shoulder"]
            if all(kp[i][0] > 0 for i in [hip, knee, ank]):
                knee_angle = self.calc_angle(kp[hip], kp[knee], kp[ank])
                STAND_THRESH = 140
                DOWN_THRESH = 100

                if knee_angle > STAND_THRESH:
                    rep_state["prev_squat"] = "up"
                    feedback = "Standing"
                    color_guide[(hip, knee)] = (0, 255, 0)
                    color_guide[(knee, ank)] = (0, 255, 0)
                elif knee_angle < DOWN_THRESH:
                    if rep_state.get("prev_squat") == "up":
                        rep_done = True
                    rep_state["prev_squat"] = "down"
                    feedback = "Squat depth good"
                    color_guide[(hip, knee)] = (0, 255, 0)
                    color_guide[(knee, ank)] = (0, 255, 0)
                else:
                    feedback = "Go deeper" if rep_state.get("prev_squat") == "up" else "Rise up"
                    color_guide[(hip, knee)] = (0, 255, 255)
                    color_guide[(knee, ank)] = (0, 255, 255)

                # Show knee angle
                knee_pos = tuple(kp[knee].astype(int))
                cv2.putText(kp, f"{int(knee_angle)}", knee_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

                # Knee over toes
                if kp[knee][0] - kp[ank][0] > 30:
                    feedback += " | Don't let knees go too far forward!"
                    color_guide[(knee, ank)] = (0, 0, 255)

                # Back angle
                if kp[sh][0] > 0:
                    torso_angle = np.degrees(np.arctan2(kp[sh][0] - kp[hip][0], kp[sh][1] - kp[hip][1]))
                    if abs(torso_angle) < 30:
                        feedback += " | Keep back straighter"
                        color_guide[(sh, hip)] = (0, 0, 255)
                break
        return feedback, color_guide, rep_done

    def get_rep_quality(self, last_feedback=""):
        if "Keep" in last_feedback or "Don't" in last_feedback:
            return 0.6
        return 1.0

    def draw_feedback(self, img, feedback, rep_count, exercise):
        h, w = img.shape[:2]
        cv2.putText(img, f"Exercise: {exercise}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(img, f"Reps: {rep_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        if feedback:
            y0, dy = 100, 30
            for i, line in enumerate(feedback.split("|")):
                y = y0 + i * dy
                cv2.putText(img, line.strip(), (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        return img
