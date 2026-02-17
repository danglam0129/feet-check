import os
import base64
import json
import re
import requests
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import streamlit as st


class MediaPipePose:
    def __init__(self):
        self.pose_model_path = self._ensure_pose_model()
        self.seg_model_path = self._ensure_segmentation_model()
        if self.pose_model_path is None:
            self.pose_landmarker = None
        else:
            self.pose_landmarker = self._get_pose_landmarker(str(self.pose_model_path))
        if self.seg_model_path is None:
            self.segmenter = None
        else:
            self.segmenter = self._get_segmenter(str(self.seg_model_path))

    @staticmethod
    def _ensure_pose_model():
        model_url = (
            "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
            "pose_landmarker_full/float16/1/pose_landmarker_full.task"
        )
        cache_dir = Path.home() / ".cache" / "mediapipe"
        cache_dir.mkdir(parents=True, exist_ok=True)
        model_path = cache_dir / "pose_landmarker_full.task"
        if not model_path.exists():
            try:
                urllib.request.urlretrieve(model_url, model_path)
            except Exception:
                return None
        return model_path

    @staticmethod
    def _ensure_segmentation_model():
        model_url = (
            "https://storage.googleapis.com/mediapipe-models/image_segmenter/"
            "selfie_multiclass_256x256/float16/1/selfie_multiclass_256x256.tflite"
        )
        cache_dir = Path.home() / ".cache" / "mediapipe"
        cache_dir.mkdir(parents=True, exist_ok=True)
        model_path = cache_dir / "selfie_multiclass_256x256.tflite"
        if not model_path.exists():
            try:
                urllib.request.urlretrieve(model_url, model_path)
            except Exception:
                return None
        return model_path

    @staticmethod
    @st.cache_resource
    def _get_pose_landmarker(model_path_str):
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        base_options = mp_python.BaseOptions(model_asset_path=model_path_str)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        return vision.PoseLandmarker.create_from_options(options)

    @staticmethod
    @st.cache_resource
    def _get_segmenter(model_path_str):
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        base_options = mp_python.BaseOptions(model_asset_path=model_path_str)
        options = vision.ImageSegmenterOptions(
            base_options=base_options,
            output_category_mask=True,
            running_mode=vision.RunningMode.IMAGE,
        )
        return vision.ImageSegmenter.create_from_options(options)

    def get_landmarks(self, img_rgb):
        if self.pose_landmarker is None:
            return "MODEL_DOWNLOAD_FAILED"
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        result = self.pose_landmarker.detect(mp_image)
        if not result.pose_landmarks:
            return None
        return result.pose_landmarks[0]

    def get_person_mask(self, img_rgb):
        if self.segmenter is None:
            return None
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        result = self.segmenter.segment(mp_image)
        return np.array(result.category_mask)


class GeminiClient:
    def __init__(self, model="gemini-2.5-flash"):
        self.model = model
        self.api_key = self._load_key()

    @staticmethod
    def _load_key():
        key = os.environ.get("GEMINI_API_KEY")
        if key:
            return key.strip()
        env_path = Path(".env")
        if not env_path.exists():
            return None
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("GEMINI_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"')
        return None

    @staticmethod
    def _normalize_model_name(model):
        m = model.strip()
        if m.startswith("models/"):
            m = m.split("/", 1)[1]
        return m

    def classify(self, image_rgb):
        import base64
        import json

        if not self.api_key:
            return None, "GEMINI_API_KEY not found in env or .env"

        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
        if not ok:
            return None, "Failed to encode image"
        img_b64 = base64.b64encode(buf.tobytes()).decode("utf-8")

        model = self._normalize_model_name(self.model)
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        headers = {"x-goog-api-key": self.api_key, "Content-Type": "application/json"}
        prompt = (
            "You are an ergonomic assessment assistant. "
            "Given a side-profile seated person image, classify feet grounding. "
            "Return ONLY valid JSON with keys: classification, confidence, left_foot{visibility,classification}, "
            "right_foot{visibility,classification}, notes (array of strings), reason. "
            "Valid classifications: Fully Grounded, Partially Grounded, Not Grounded, Unknown. "
            "Visibility values: Visible, Occluded, Unknown. "
            "Confidence is 0-1. "
            "If you cannot see a foot, set its classification to Unknown and visibility to Occluded. "
            "Do not include code fences or extra text."
        )

        body = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": prompt},
                        {"inline_data": {"mime_type": "image/jpeg", "data": img_b64}},
                    ],
                }
            ]
        }

        try:
            resp = requests.post(url, headers=headers, data=json.dumps(body), timeout=60)
        except Exception as e:
            return None, f"Request failed: {e}"

        if resp.status_code != 200:
            return None, f"Gemini error: {resp.status_code} {resp.text}"

        try:
            data = resp.json()
            text_out = data["candidates"][0]["content"]["parts"][0]["text"]
        except Exception as e:
            return None, f"Bad response format: {e}"

        text_out = text_out.strip()
        if text_out.startswith("```"):
            text_out = re.sub(r"^```[a-zA-Z]*", "", text_out).strip()
            if text_out.endswith("```"):
                text_out = text_out[:-3].strip()

        try:
            result = json.loads(text_out)
        except Exception as e:
            return None, f"JSON parse failed: {e}\n{text_out}"

        return result, None
