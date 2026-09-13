# FastAPI app: /health, /detect, /ask. Kept thin on purpose - upload
# validation, error mapping, logging, then hand off to Detector or the
# reasoning pipeline, both already tested elsewhere.
from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError

from app.constants import MODEL_VERSION
from app.detector import Detector
from app.logging_config import configure_logging, log_request
from app.reasoning.pipeline import answer_question
from app.schemas import DetectResponse

logger = configure_logging()

# Read once at import, not per-request. Tests override this directly
# rather than faking an env var.
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_MB", "10")) * 1024 * 1024

_detector_instance: Detector | None = None


def get_detector() -> Detector:
    """FastAPI dependency - same Detector instance for every route.
    Not a bare global so tests can override it with a stub."""
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = Detector()
    return _detector_instance


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Tries to load weights at startup but doesn't crash if that fails -
    a service that's up and reporting "model not loaded" is way easier to
    debug than one that won't start at all."""
    try:
        get_detector().load()
        logger.info("startup: model loaded", extra={"extra_fields": {}})
    except Exception as error:
        logger.error(
            "startup: model failed to load, API will report 503 on inference routes",
            extra={"extra_fields": {"error": str(error)}},
        )
    yield


app = FastAPI(title="Constrained Document Layout Detection API", lifespan=lifespan)


@app.exception_handler(Exception)
async def handle_unexpected_error(request: Request, exc: Exception):
    """Catches anything a route didn't handle and returns a safe generic
    message instead of leaking a stack trace. Full error still goes to logs."""
    logger.error(
        "unhandled exception",
        extra={"extra_fields": {"path": str(request.url), "error": str(exc)}},
    )
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. See server logs for details."},
    )


def _read_and_validate_image(contents: bytes, content_type: str | None) -> Image.Image:
    """Size check, then content-type check, then an actual decode attempt -
    a file can claim image/png and still be garbage."""
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Upload exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit.",
        )

    if content_type is None or not content_type.startswith("image/"):
        raise HTTPException(
            status_code=415,
            detail=f"Expected an image upload, got content-type '{content_type}'.",
        )

    try:
        import io

        return Image.open(io.BytesIO(contents)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(status_code=415, detail="Could not decode the upload as an image.")


@app.get("/health")
def health(detector: Detector = Depends(get_detector)):
    return {"status": "ok", "model_loaded": detector.is_loaded, "model_version": MODEL_VERSION}


@app.post("/detect", response_model=DetectResponse)
async def detect(file: UploadFile = File(...), detector: Detector = Depends(get_detector)):
    with log_request(logger, "/detect") as ctx:
        if not detector.is_loaded:
            raise HTTPException(
                status_code=503,
                detail="Model is not loaded. Check MODEL_PATH and server startup logs.",
            )

        contents = await file.read()
        image = _read_and_validate_image(contents, file.content_type)

        detections, inference_ms = detector.predict(image)
        ctx["detection_count"] = len(detections)

        return DetectResponse(
            detections=detections,
            image_size={"width": image.width, "height": image.height},
            inference_ms=round(inference_ms, 2),
            model_version=MODEL_VERSION,
        )


@app.post("/ask")
async def ask(
    file: UploadFile = File(...),
    question: str = Form(..., min_length=1, max_length=500),
    detector: Detector = Depends(get_detector),
):
    """Part B. All the actual routing/evidence/guardrail/synthesis logic
    lives in app.reasoning.pipeline - this is just upload validation + wiring."""
    with log_request(logger, "/ask", question_length=len(question)) as ctx:
        contents = await file.read()
        image = _read_and_validate_image(contents, file.content_type)

        response = answer_question(image=image, question=question, detector=detector)
        ctx["used_detection"] = response.used_detection
        ctx["insufficient_information"] = response.insufficient_information

        return response
