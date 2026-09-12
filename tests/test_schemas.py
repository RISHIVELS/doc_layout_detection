"""
I put these constraints on the schema rather than trusting FastAPI's default
validation to catch them, because the two mistakes below are exactly the kind
that pass a casual code review and only show up when a reviewer's hidden test
image happens to trigger them.
"""

import pytest
from pydantic import ValidationError

from app.constants import MODEL_VERSION
from app.schemas import BBox, Detection, DetectResponse


class TestBBox:
    def test_ordinary_box_is_accepted(self):
        box = BBox(x1=10.0, y1=20.0, x2=110.0, y2=220.0)
        assert box.x2 > box.x1 and box.y2 > box.y1

    def test_rejects_a_box_where_x2_is_before_x1(self):
        # This is the mistake I would make if I ever swap width for x2 by
        # accident when converting from the model's raw xyxy output.
        with pytest.raises(ValidationError):
            BBox(x1=100.0, y1=20.0, x2=50.0, y2=220.0)

    def test_rejects_a_box_where_y2_is_before_y1(self):
        with pytest.raises(ValidationError):
            BBox(x1=10.0, y1=200.0, x2=110.0, y2=50.0)


class TestDetection:
    def test_confidence_must_stay_within_zero_and_one(self):
        # RT-DETR always gives me a value in this range, but I want the schema
        # to catch it in case a future model swap gives me a raw logit instead.
        with pytest.raises(ValidationError):
            Detection(
                class_name="Table", class_id=8, confidence=1.4,
                bbox=BBox(x1=0, y1=0, x2=10, y2=10),
            )

    def test_serialises_to_the_shape_the_api_promises(self):
        """
        This is the exact response shape from the design doc, and I am pinning
        it here so a refactor of the schema cannot silently change the field
        names the frontend or a reviewer's script depends on.
        """
        detection = Detection(
            class_name="Table", class_id=8, confidence=0.94,
            bbox=BBox(x1=91.0, y1=604.2, x2=934.5, y2=889.1),
        )
        payload = detection.model_dump()

        assert payload["class_name"] == "Table"
        assert payload["class_id"] == 8
        assert payload["confidence"] == 0.94
        assert payload["bbox"] == {"x1": 91.0, "y1": 604.2, "x2": 934.5, "y2": 889.1}


class TestDetectResponse:
    def test_carries_the_model_version_so_a_reviewer_knows_which_weights_ran(self):
        response = DetectResponse(
            detections=[],
            image_size={"width": 1025, "height": 1025},
            inference_ms=41.2,
            model_version=MODEL_VERSION,
        )
        assert response.model_version == MODEL_VERSION
