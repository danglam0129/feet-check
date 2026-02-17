import math

import cv2
import numpy as np


class GroundingAnalyzer:
    GROUND_THRESHOLD_PX = 12

    POSE_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 7),
        (0, 4), (4, 5), (5, 6), (6, 8),
        (9, 10),
        (11, 12),
        (11, 13), (13, 15),
        (12, 14), (14, 16),
        (15, 17), (16, 18),
        (11, 23), (12, 24),
        (23, 24),
        (23, 25), (25, 27), (27, 29), (29, 31),
        (24, 26), (26, 28), (28, 30), (30, 32),
    ]

    def __init__(self, pose_detector):
        self.pose = pose_detector

    @staticmethod
    def classify_foot(heel_y, toe_y, floor_y, threshold=8):
        heel_touch = abs(heel_y - floor_y) <= threshold
        toe_touch = abs(toe_y - floor_y) <= threshold
        if heel_touch and toe_touch:
            return "Fully Grounded"
        if heel_touch or toe_touch:
            return "Partially Grounded"
        return "Not Grounded"


    @staticmethod
    def fit_floor_line_hough(edges, h, w):
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=3.14159 / 180,
            threshold=80,
            minLineLength=int(w * 0.3),
            maxLineGap=40,
        )
        if lines is None:
            return None
        candidates = []
        for x1, y1, x2, y2 in lines[:, 0]:
            if abs(y1 - y2) <= 5 and y1 > int(h * 0.5) and y2 > int(h * 0.5):
                candidates.append((x1, y1, x2, y2))
        if not candidates:
            return None
        x1, y1, x2, y2 = max(candidates, key=lambda l: abs(l[2] - l[0]))
        if x2 == x1:
            return 0.0
        return abs((y2 - y1) / (x2 - x1))
    def analyze(self, img_rgb):
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        h, w, _ = img_rgb.shape

        landmarks = self.pose.get_landmarks(img_rgb)
        if landmarks == "MODEL_DOWNLOAD_FAILED":
            return None, "Failed to download MediaPipe pose model."
        if not landmarks:
            return None, "No person detected."

        # MediaPipe indexes
        LEFT_HEEL = 29
        RIGHT_HEEL = 30
        LEFT_TOE = 31
        RIGHT_TOE = 32

        def to_pixel(lm):
            visibility = getattr(lm, "visibility", getattr(lm, "presence", 1.0))
            return int(lm.x * w), int(lm.y * h), visibility

        lh_x, lh_y, lh_v = to_pixel(landmarks[LEFT_HEEL])
        rh_x, rh_y, rh_v = to_pixel(landmarks[RIGHT_HEEL])
        lt_x, lt_y, lt_v = to_pixel(landmarks[LEFT_TOE])
        rt_x, rt_y, rt_v = to_pixel(landmarks[RIGHT_TOE])

        VIS_THRESHOLD = 0.5
        left_visible = min(lh_v, lt_v) > VIS_THRESHOLD
        right_visible = min(rh_v, rt_v) > VIS_THRESHOLD

        if not left_visible and not right_visible:
            return None, "Feet not visible."

        left_floor_y = max(lh_y, lt_y) if left_visible else None
        right_floor_y = max(rh_y, rt_y) if right_visible else None

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)

        slope = self.fit_floor_line_hough(edges, h, w)
        slope_boost = 0
        if slope is not None:
            slope_boost = min(8, int(slope * 100))
        thresh = self.GROUND_THRESHOLD_PX + slope_boost


        if left_visible:
            left_class = self.classify_foot(lh_y, lt_y, left_floor_y, threshold=thresh)
        else:
            left_class = "Unknown"

        if right_visible:
            right_class = self.classify_foot(rh_y, rt_y, right_floor_y, threshold=thresh)
        else:
            right_class = "Unknown"

        if left_class == "Fully Grounded" and right_class == "Fully Grounded":
            person_class = "Fully Grounded"
        elif left_class == "Not Grounded" and right_class == "Not Grounded":
            person_class = "Not Grounded"
        elif left_class == "Unknown" and right_class == "Unknown":
            person_class = "Unknown"
        else:
            person_class = "Partially Grounded"

        confidence = round(min(lh_v, lt_v, rh_v, rt_v), 2)

        notes = []
        if left_visible:
            if abs(lt_y - left_floor_y) > thresh:
                notes.append("Left toe lifted")
            if abs(lh_y - left_floor_y) > thresh:
                notes.append("Left heel lifted")
        if right_visible:
            if abs(rt_y - right_floor_y) > thresh:
                notes.append("Right toe lifted")
            if abs(rh_y - right_floor_y) > thresh:
                notes.append("Right heel lifted")
        if left_class != right_class:
            notes.append("Mixed grounding states")

        reason = (
            "Both feet Fully Grounded"
            if person_class == "Fully Grounded"
            else "Both feet Not Grounded"
            if person_class == "Not Grounded"
            else "Mixed grounding states"
            if person_class == "Partially Grounded"
            else "Insufficient visibility"
        )

        output = {
            "classification": person_class,
            "confidence": confidence,
            "reason": reason,
            "left_foot": {
                "visibility": "Visible" if left_visible else "Occluded",
                "classification": left_class,
            },
            "right_foot": {
                "visibility": "Visible" if right_visible else "Occluded",
                "classification": right_class,
            },
            "notes": notes,
        }

        annotated = img_bgr.copy()
        return {
            "output": output,
            "annotated": annotated,
            "landmarks": landmarks,
            "pose_connections": self.POSE_CONNECTIONS,
            "left_floor_y": left_floor_y,
            "right_floor_y": right_floor_y,
            "points": {
                "lh": (lh_x, lh_y),
                "lt": (lt_x, lt_y),
                "rh": (rh_x, rh_y),
                "rt": (rt_x, rt_y),
            },
        }, None
