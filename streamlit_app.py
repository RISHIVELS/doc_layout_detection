# Streamlit demo UI for Hugging Face Spaces. Reuses the same Detector and
# reasoning pipeline as the FastAPI app - no duplicated logic, just a
# visual layer for showing the thing actually working.
from __future__ import annotations

import os

import streamlit as st
from PIL import Image, ImageDraw

from app.constants import CLASS_NAMES, MODEL_VERSION
from app.detector import Detector
from app.reasoning.pipeline import answer_question
from scripts._render_utils import load_label_font

st.set_page_config(page_title="Document Layout Detection", page_icon="\U0001F4C4", layout="wide")

PALETTE = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4",
    "#46f0f0", "#f032e6", "#bcf60c", "#fabebe", "#008080",
]


@st.cache_resource
def get_detector() -> Detector:
    # cache_resource so weights load once per container, not per request
    detector = Detector()
    try:
        detector.load()
    except FileNotFoundError:
        pass  # surfaced in the UI below instead of crashing the app
    return detector


def draw_detections(image: Image.Image, detections) -> Image.Image:
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    font = load_label_font(18)

    for det in detections:
        colour = PALETTE[det.class_id % len(PALETTE)]
        box = det.bbox
        draw.rectangle([box.x1, box.y1, box.x2, box.y2], outline=colour, width=3)
        label = f"{det.class_name} {det.confidence:.2f}"
        text_box = draw.textbbox((box.x1, box.y1), label, font=font)
        draw.rectangle(
            [text_box[0] - 2, text_box[1] - 2, text_box[2] + 2, text_box[3] + 2],
            fill=colour,
        )
        draw.text((box.x1, box.y1), label, font=font, fill="white")

    return annotated


detector = get_detector()

st.title("Constrained Document Layout Detection")
st.caption(f"RT-DETR fine-tuned on DocLayNet - {MODEL_VERSION}")

if not detector.is_loaded:
    st.warning(
        f"Model weights not found at `{os.environ.get('MODEL_PATH', './weights/best.pt')}`. "
        "Detection and Q&A won't work until weights are available - see the README "
        "for the download link, or set the MODEL_PATH secret on this Space."
    )

with st.sidebar:
    st.subheader("Classes")
    st.write(", ".join(CLASS_NAMES))
    st.subheader("Model")
    st.write("loaded" if detector.is_loaded else "not loaded")
    if not os.environ.get("GROQ_API_KEY"):
        st.info("GROQ_API_KEY not set - the Ask tab needs it for the reasoning layer.")

tab_detect, tab_ask = st.tabs(["Detect", "Ask"])

with tab_detect:
    st.write("Upload a document page to see the detected layout regions.")
    uploaded = st.file_uploader("Image", type=["png", "jpg", "jpeg"], key="detect_upload")

    if uploaded and st.button("Run detection", disabled=not detector.is_loaded):
        image = Image.open(uploaded).convert("RGB")
        with st.spinner("Running RT-DETR..."):
            detections, inference_ms = detector.predict(image)

        col1, col2 = st.columns(2)
        col1.image(image, caption="Original", use_container_width=True)
        col2.image(draw_detections(image, detections), caption="Detections", use_container_width=True)

        st.caption(f"{len(detections)} detections in {inference_ms:.1f} ms")
        if detections:
            st.table([
                {"class": d.class_name, "confidence": round(d.confidence, 3)}
                for d in sorted(detections, key=lambda d: -d.confidence)
            ])

with tab_ask:
    st.write("Ask a question about the document's layout - not its text content.")
    uploaded_q = st.file_uploader("Image", type=["png", "jpg", "jpeg"], key="ask_upload")
    question = st.text_input("Question", placeholder="How many tables are on this page?")

    ask_disabled = not detector.is_loaded or not os.environ.get("GROQ_API_KEY")
    if uploaded_q and question and st.button("Ask", disabled=ask_disabled):
        image = Image.open(uploaded_q).convert("RGB")
        with st.spinner("Thinking..."):
            response = answer_question(image=image, question=question, detector=detector)

        if response.insufficient_information:
            st.warning(response.answer)
        else:
            st.success(response.answer)

        with st.expander("Reasoning trace"):
            st.json(response.reasoning_trace)
