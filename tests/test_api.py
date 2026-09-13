"""
Tests the HTTP layer itself - status codes, error handling, response shape -
against a stubbed detector so none of this needs a GPU or real weights.
Whether the *model* is any good is evaluate.py's job; this file only checks
that the API around it behaves correctly.

I lean on FastAPI's dependency-override mechanism to swap in a fake detector,
rather than patching app.detector.Detector globally, because dependency
overrides are scoped to the test and cannot leak into a different test's
assertions the way a module-level patch can.
"""

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app, get_detector
from app.schemas import BBox, Detection


class _StubDetector:
    is_loaded = True
    query_budget = 300

    def __init__(self, detections=None, raise_on_predict=False):
        self._detections = detections or []
        self._raise = raise_on_predict

    def predict(self, image, conf=0.25):
        if self._raise:
            raise RuntimeError("simulated model failure")
        return self._detections, 12.3


def _tiny_png_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (100, 100), "white").save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def client():
    """
    raise_server_exceptions=False is required here, not optional. By default
    TestClient re-raises an unhandled exception straight into the test,
    bypassing the app's own exception handler entirely - which is the right
    default for catching bugs during ordinary testing, but it is exactly
    backwards for test_detector_failure_returns_500_..., whose whole point is
    verifying what a real caller receives when app.main's global handler
    catches something. Without this flag that test would see a raised
    RuntimeError instead of a Response to assert on.
    """
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client
    app.dependency_overrides.clear()


class TestHealth:
    def test_health_reports_model_state(self, client):
        app.dependency_overrides[get_detector] = lambda: _StubDetector()
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["model_loaded"] is True


class TestDetectEndpoint:
    def test_valid_image_returns_the_documented_response_shape(self, client):
        detection = Detection(
            class_name="Table", class_id=8, confidence=0.94,
            bbox=BBox(x1=91.0, y1=604.2, x2=934.5, y2=889.1),
        )
        app.dependency_overrides[get_detector] = lambda: _StubDetector([detection])

        response = client.post(
            "/detect", files={"file": ("page.png", _tiny_png_bytes(), "image/png")},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["detections"][0]["class_name"] == "Table"
        assert body["model_version"]
        assert "inference_ms" in body

    def test_non_image_upload_is_rejected_with_415(self, client):
        app.dependency_overrides[get_detector] = lambda: _StubDetector()

        response = client.post(
            "/detect", files={"file": ("notes.txt", b"not an image", "text/plain")},
        )

        assert response.status_code == 415

    def test_oversized_upload_is_rejected_with_413(self, client, monkeypatch):
        """
        I do not want to actually construct a 10MB+ file in a test - that is
        slow and wasteful for something the endpoint should reject before
        ever trying to decode it. So I shrink the configured limit instead of
        growing the payload.
        """
        import app.main as main_module

        monkeypatch.setattr(main_module, "MAX_UPLOAD_BYTES", 10)
        app.dependency_overrides[get_detector] = lambda: _StubDetector()

        response = client.post(
            "/detect", files={"file": ("page.png", _tiny_png_bytes(), "image/png")},
        )

        assert response.status_code == 413

    def test_model_not_loaded_returns_503_not_a_500(self, client):
        """
        A model that failed to load is a known, expected operational state -
        wrong MODEL_PATH, weights not downloaded yet - not a bug in the
        request handling. 503 tells a caller "try again later / check
        config", which a generic 500 does not.
        """
        broken = _StubDetector()
        broken.is_loaded = False
        app.dependency_overrides[get_detector] = lambda: broken

        response = client.post(
            "/detect", files={"file": ("page.png", _tiny_png_bytes(), "image/png")},
        )

        assert response.status_code == 503

    def test_detector_failure_returns_500_with_a_clear_message_not_a_stack_trace(self, client):
        app.dependency_overrides[get_detector] = lambda: _StubDetector(raise_on_predict=True)

        response = client.post(
            "/detect", files={"file": ("page.png", _tiny_png_bytes(), "image/png")},
        )

        assert response.status_code == 500
        assert "simulated model failure" not in response.text  # no raw internals leaked
        assert response.json()["detail"]


class TestAskEndpoint:
    def test_missing_question_field_returns_422(self, client):
        app.dependency_overrides[get_detector] = lambda: _StubDetector()

        response = client.post(
            "/ask", files={"file": ("page.png", _tiny_png_bytes(), "image/png")},
        )

        assert response.status_code == 422

    def test_empty_question_returns_422(self, client):
        app.dependency_overrides[get_detector] = lambda: _StubDetector()

        response = client.post(
            "/ask",
            files={"file": ("page.png", _tiny_png_bytes(), "image/png")},
            data={"question": ""},
        )

        assert response.status_code == 422
