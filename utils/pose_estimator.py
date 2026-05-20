from ultralytics import YOLO
import cv2
import numpy as np

class PoseEstimator:
    def __init__(self, model_name="yolov8n-pose.pt"):
        self.model = YOLO(model_name)
        self.skeleton = [
            (5,6), (5,7), (7,9), (6,8), (8,10),
            (5,11), (6,12), (11,12),
            (11,13), (13,15), (12,14), (14,16)
        ]
        self.colors = {
            "good": (0,255,0),
            "warning": (0,255,255),
            "bad": (0,0,255),
            "normal": (255,255,255)
        }

    def get_keypoints(self, img):
        results = self.model(img, verbose=False)
        if not results or len(results[0].keypoints) == 0:
            return None, None
        keypoints = results[0].keypoints
        if keypoints.xy.numel() == 0:
            return None, None
        boxes = results[0].boxes
        if boxes is not None and len(boxes) > 0:
            areas = (boxes.xyxy[:,2] - boxes.xyxy[:,0]) * (boxes.xyxy[:,3] - boxes.xyxy[:,1])
            idx = areas.argmax().item()
        else:
            idx = 0
        kpts = keypoints.xy[idx].cpu().numpy()
        conf = keypoints.conf[idx].cpu().numpy() if keypoints.conf is not None else np.ones(17)
        if np.mean(conf) < 0.3:
            return None, None
        x1, y1, x2, y2 = boxes.xyxy[idx].cpu().numpy() if boxes is not None else (0,0,img.shape[1], img.shape[0])
        return kpts, (int(x1), int(y1), int(x2), int(y2))

    def draw_skeleton(self, img, keypoints, line_colors=None):
        if keypoints is None:
            return img
        overlay = img.copy()
        for (i, j) in self.skeleton:
            if i >= len(keypoints) or j >= len(keypoints):
                continue
            pt1 = tuple(keypoints[i].astype(int))
            pt2 = tuple(keypoints[j].astype(int))
            if pt1[0]<=0 or pt1[1]<=0 or pt2[0]<=0 or pt2[1]<=0:
                continue
            color = self.colors["normal"]
            if line_colors and (i,j) in line_colors:
                color = line_colors[(i,j)]
            cv2.line(overlay, pt1, pt2, color, 2)
        for pt in keypoints:
            x, y = int(pt[0]), int(pt[1])
            if x>0 and y>0:
                cv2.circle(overlay, (x,y), 4, (0,255,255), -1)
        return cv2.addWeighted(img, 0.6, overlay, 0.4, 0)

    def draw_text(self, img, text, pos, font_scale=0.7, color=(0,255,255)):
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2)
        return img
