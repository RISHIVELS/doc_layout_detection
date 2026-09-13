"""
The FastAPI app: two endpoints, /detect and /ask, plus /health.

I kept this file thin on purpose - it does upload validation, error mapping,
and logging, and then hands off to code that already has its own tests
(Detector for inference, the reasoning pipeline for Part B). Nothing about
the detection or reasoning logic itself lives here; if I ever need to check
"is the model good", I look at evaluate.py, not this file.

The one deliberate architectural choice worth calling out: the detector is
provided via FastAPI's dependency injection (`get_detector`) rather than a
bare module-level global the route handlers reach into directly. That is
what lets tests/test_api.py swap in a stub detector per test and never touch
a GPU or real weights file - dependency overrides are scoped to the test and
cannot leak into another test's assertions the way patching a module global
can.
"""

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

# Read once at import time rather than on every request - the limit does not
# change at runtime, and re-reading an env var per request is pure overhead.
# Tests override this module attribute directly (monkeypatch) rather than
# faking an environment variable, which is simpler for a single int.
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_MB", "10")) * 1024 * 1024

_detector_instance: Detector | None = None


def get_detector() -> Detector:
    """
    FastAPI dependency that hands every route the same Detector instance.

    Deliberately not a bare global the routes import directly - see the
    module docstring for why. `app.dependency_overrides[get_detector] = ...`
    is how tests substitute a stub without a GPU.
    """
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = Detector()
    return _detector_instance


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Tries to load the model weights once at startup, but does not crash the
    app if that fails.

    I want /health to report accurate state from the moment the container is
    up, rather than only discovering a bad MODEL_PATH on someone's first real
    request. But a missing or bad checkpoint should not take the whole API
    down - a service that is up and correctly reporting "model not loaded"
    is far more debuggable than a container that will not even start.
    """
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
    """
    Catches anything a route did not explicitly handle and turns it into a
    plain, safe JSON response instead of leaking an internal stack trace or
    exception message to the caller.

    The real error still goes to the server logs in full - this handler is
    about what a caller outside the process gets to see, not about hiding
    the problem from me.
    """
    logger.error(
        "unhandled exception",
        extra={"extra_fields": {"path": str(request.url), "error": str(exc)}},
    )
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. See server logs for details."},
    )


def _read_and_validate_image(contents: bytes, content_type: str | None) -> Image.Image:
    """
    Shared validation for both endpoints: size, then declared type, then
    actually decodable as an image.

    I check the declared content-type first because it is the cheap check -
    rejecting an obviously-wrong upload (someone posting a .txt file) should
    not require decoding anything. But I do not stop there: a file can claim
    `image/png` in its header and still be garbage bytes, so I also try to
    actually open it and treat a decode failure the same as a wrong content
    type, rather than letting it turn into an unhandled 500 deeper in the
    detector.
    """
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
    """
    Reports whether the model is actually loaded, not just whether the
    process is running. A process that is up but has no usable weights is
    not healthy in any sense a caller cares about.
    """
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
    """
    Part B: a natural-language question about the uploaded image.

    All of the actual routing/evidence/guardrail/synthesis logic lives in
    app.reasoning.pipeline.answer_question - this route is just upload
    validation plus wiring, same as /detect. See that module (and its tests)
    for the interesting behaviour: whether detection is even needed, and
    when the response should honestly say it cannot answer.
    """
    with log_request(logger, "/ask", question_length=len(question)) as ctx:
        contents = await file.read()
        image = _read_and_validate_image(contents, file.content_type)

        response = answer_question(image=image, question=question, detector=detector)
        ctx["used_detection"] = response.used_detection
        ctx["insufficient_information"] = response.insufficient_information

        return response
