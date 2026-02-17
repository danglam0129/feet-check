import cv2
import numpy as np
import streamlit as st
from PIL import Image

from core.detector import GeminiClient, MediaPipePose
from core.grounding import GroundingAnalyzer
from core.classifier import ClassifierRenderer
from utils.visualizer import draw_point


st.set_page_config(layout="wide")
st.title("Feet Grounding Classification")

uploaded = st.file_uploader("Upload side-profile seated image", type=["jpg", "png", "jpeg", "webp"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    img = np.array(image)

    st.subheader("Original Image")
    st.image(image)

    col_a, col_b = st.columns(2)
    run_mediapipe = col_a.button("MediaPipe + Logic")
    run_gemini = col_b.button("API AI")

    if run_gemini:
        gemini_model = st.session_state.get("gemini_model", "gemini-2.5-flash")
        client = GeminiClient(model=gemini_model)
        result, err = client.classify(img)
        if err:
            st.error(err)
        else:
            st.subheader("Result (Gemini)")
            st.json(result)

    if run_mediapipe:
        analyzer = GroundingAnalyzer(MediaPipePose())
        result, err = analyzer.analyze(img)
        if err:
            st.error(err)
        else:
            annotated = result["annotated"]
            h, w, _ = annotated.shape
            ClassifierRenderer.draw_skeleton(
                annotated,
                result["landmarks"],
                result["pose_connections"],
                w,
                h,
            )

            if result.get("left_floor_y") is not None:
                ClassifierRenderer.draw_floor_line(annotated, result["left_floor_y"], color=(255, 0, 0), thickness=2)
            if result.get("right_floor_y") is not None:
                ClassifierRenderer.draw_floor_line(annotated, result["right_floor_y"], color=(255, 0, 255), thickness=2)

            points = result.get("points", {})
            if "lh" in points:
                draw_point(annotated, points["lh"][0], points["lh"][1], (0, 255, 0))
            if "lt" in points:
                draw_point(annotated, points["lt"][0], points["lt"][1], (0, 255, 0))
            if "rh" in points:
                draw_point(annotated, points["rh"][0], points["rh"][1], (0, 255, 255))
            if "rt" in points:
                draw_point(annotated, points["rt"][0], points["rt"][1], (0, 255, 255))

            output = result["output"]
            ClassifierRenderer.draw_label(annotated, output["classification"])

            st.subheader("Result (MediaPipe)")
            col1, col2 = st.columns(2)
            with col1:
                st.image(annotated, caption="Annotated Image")
            with col2:
                st.json(output)
