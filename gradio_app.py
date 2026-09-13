# Gradio UI for Hugging Face Spaces - same Detector and reasoning pipeline
# as the FastAPI app and streamlit_app.py, just a different front end.
# HF's current Space creation flow only offers Gradio/Docker/Static as
# SDKs (no standalone Streamlit option), so this is the path that works
# without a paid plan.
from __future__ import annotations

import os

import gradio as gr
from dotenv import load_dotenv

load_dotenv()

from PIL import Image, ImageDraw

from app.constants import CLASS_NAMES, MODEL_VERSION
from app.detector import Detector
from app.reasoning.pipeline import answer_question
from scripts._render_utils import load_label_font

PALETTE = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4",
    "#46f0f0", "#f032e6", "#bcf60c", "#fabebe", "#008080",
]

_detector: Detector | None = None


def get_detector() -> Detector:
    # module-level singleton, same reasoning as main.py's get_detector -
    # loads once per process, not per request
    global _detector
    if _detector is None:
        _detector = Detector()
        try:
            _detector.load()
        except FileNotFoundError:
            pass  # surfaced in the UI instead of crashing the app
    return _detector


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


def run_detect(image: Image.Image):
    detector = get_detector()
    if not detector.is_loaded:
        return None, "Model not loaded - check MODEL_PATH/MODEL_URL.", None
    if image is None:
        return None, "Upload an image first.", None

    detections, inference_ms = detector.predict(image)
    annotated = draw_detections(image, detections)
    table = [[d.class_name, round(d.confidence, 3)] for d in sorted(detections, key=lambda d: -d.confidence)]
    summary = f"{len(detections)} detections in {inference_ms:.1f} ms"
    return annotated, summary, table


def run_ask(image: Image.Image, question: str):
    detector = get_detector()
    if not detector.is_loaded:
        return "Model not loaded - check MODEL_PATH/MODEL_URL.", {}
    if image is None or not question:
        return "Upload an image and enter a question.", {}
    if not os.environ.get("GROQ_API_KEY"):
        return "GROQ_API_KEY not set - the reasoning layer needs it.", {}

    response = answer_question(image=image, question=question, detector=detector)
    return response.answer, response.reasoning_trace


with gr.Blocks(title="Document Layout Detection") as demo:
    gr.Markdown(f"# Constrained Document Layout Detection\nRT-DETR fine-tuned on DocLayNet - `{MODEL_VERSION}`")
    gr.Markdown(f"Classes: {', '.join(CLASS_NAMES)}")

    with gr.Tab("Detect"):
        with gr.Row():
            detect_input = gr.Image(type="pil", label="Upload a document page")
            detect_output = gr.Image(type="pil", label="Detections")
        detect_button = gr.Button("Run detection")
        detect_summary = gr.Textbox(label="Summary", interactive=False)
        detect_table = gr.Dataframe(headers=["class", "confidence"], label="Detections")
        detect_button.click(run_detect, inputs=detect_input, outputs=[detect_output, detect_summary, detect_table])

    with gr.Tab("Ask"):
        ask_input = gr.Image(type="pil", label="Upload a document page")
        ask_question = gr.Textbox(label="Question", placeholder="How many tables are on this page?")
        ask_button = gr.Button("Ask")
        ask_answer = gr.Textbox(label="Answer", interactive=False)
        ask_trace = gr.JSON(label="Reasoning trace")
        ask_button.click(run_ask, inputs=[ask_input, ask_question], outputs=[ask_answer, ask_trace])

if __name__ == "__main__":
    demo.launch()
