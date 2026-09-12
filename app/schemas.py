"""
Every shape that crosses the API boundary lives here, so /detect and /ask (and
the tests) are all reading from the same definitions instead of each endpoint
building its own dict and hoping the field names line up.

I lean on pydantic's validators rather than checking things by hand in the
route handlers, for a reason that bit me once already in this project: a
subtly wrong number does not crash, it just quietly produces a bad answer.
A box where x2 < x1 is a real thing that can happen if I ever mix up "width"
and "x2" while converting the model's raw output, and I would rather find out
from a validation error than from a reviewer asking why my bounding boxes
look inside-out.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class BBox(BaseModel):
    """A box in absolute pixel coordinates, corners rather than centre+size.

    I chose corner format for the API response specifically because it is
    what a consumer actually wants to draw a rectangle - centre+width+height
    is what the training pipeline needs internally, and there is no reason to
    make an API caller redo that arithmetic.
    """

    x1: float
    y1: float
    x2: float
    y2: float

    @field_validator("x2")
    @classmethod
    def x2_after_x1(cls, x2: float, info) -> float:
        x1 = info.data.get("x1")
        if x1 is not None and x2 <= x1:
            raise ValueError(f"x2 ({x2}) must be greater than x1 ({x1})")
        return x2

    @field_validator("y2")
    @classmethod
    def y2_after_y1(cls, y2: float, info) -> float:
        y1 = info.data.get("y1")
        if y1 is not None and y2 <= y1:
            raise ValueError(f"y2 ({y2}) must be greater than y1 ({y1})")
        return y2


class Detection(BaseModel):
    """One region the detector found, with its class and how sure it is."""

    class_name: str
    class_id: int
    bbox: BBox
    confidence: float = Field(ge=0.0, le=1.0)


class ImageSize(BaseModel):
    width: int
    height: int


class DetectResponse(BaseModel):
    """
    The full response for POST /detect.

    model_version is here on purpose, not as an afterthought. If I retrain the
    weights later, or a reviewer compares my hidden-set run against a run from
    a different checkpoint, this field is what lets either of us tell which
    model actually produced a given answer.
    """

    detections: list[Detection]
    image_size: ImageSize
    inference_ms: float
    model_version: str


class AskRequest(BaseModel):
    """The natural-language half of the /ask endpoint. Image comes in as a
    separate multipart file, not as JSON, so this only carries the question."""

    question: str = Field(min_length=1, max_length=500)


class ConfidenceStats(BaseModel):
    max: float
    mean: float
    min: float


class RouteDecision(BaseModel):
    """
    What the intent router decided about a question, before any detection runs.

    task_type is a closed set rather than a free string because the guardrail
    in app/reasoning/guardrail.py branches on it directly - "count" behaves
    differently from "presence" - and a typo in a free-form string would fail
    silently by falling through to no guardrail rule at all, which is the
    opposite of what a guardrail is for.
    """

    needs_detection: bool
    target_classes: list[str]
    task_type: Literal["count", "presence", "compare", "describe", "out_of_scope"]
    reason: str
    source: Literal["prefilter", "llm"]


class GuardrailVerdict(BaseModel):
    """
    Whether the evidence is good enough to answer confidently, and if not,
    which rule said so.

    I keep `detail` as a free dict rather than a fixed schema because each
    rule wants to explain itself differently - "all_below_tau" wants to show
    the observed max confidence, "query_saturated" wants to show the page's
    region count against the budget - and forcing one shape onto all of them
    would mean throwing away the specific number that makes the refusal
    legible to whoever reads the response.
    """

    triggered: bool
    rule: str | None = None
    detail: dict = Field(default_factory=dict)


class AskResponse(BaseModel):
    """
    The full response for POST /ask.

    insufficient_information is a plain bool set by my own code from the
    guardrail's verdict, never by asking the LLM whether it is confident. I
    made that decision deliberately: if I let the model self-report its own
    confidence, a well-written justification for guessing anyway is one bad
    prompt away, and the deterministic gate is the difference between the
    guardrail meaning something and it being a suggestion.
    """

    answer: str
    used_detection: bool
    insufficient_information: bool
    detections: list[Detection] = Field(default_factory=list)
    evidence: dict | None = None
    reasoning_trace: dict = Field(default_factory=dict)
