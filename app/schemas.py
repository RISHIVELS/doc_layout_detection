# Request/response shapes for both endpoints, all in one place so /detect,
# /ask and the tests aren't each building their own dict by hand.
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class BBox(BaseModel):
    """Corner-format box (not centre+size) - easier for a caller to draw."""

    x1: float
    y1: float
    x2: float
    y2: float

    # Catches a mixed-up width/x2 conversion before it ships as an
    # inside-out box - happened once, don't want it happening again.
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
    """Full response for POST /detect. model_version tags which checkpoint
    produced this, useful once there's more than one."""

    detections: list[Detection]
    image_size: ImageSize
    inference_ms: float
    model_version: str


class AskRequest(BaseModel):
    """Image comes in as multipart, not JSON - this is just the question."""

    question: str = Field(min_length=1, max_length=500)


class ConfidenceStats(BaseModel):
    max: float
    mean: float
    min: float


class RouteDecision(BaseModel):
    """What the router decided before any detection runs. task_type is a
    closed set, not a free string - guardrail.py branches on it directly,
    and a typo would just silently skip the guardrail."""

    needs_detection: bool
    target_classes: list[str]
    task_type: Literal["count", "presence", "compare", "describe", "out_of_scope"]
    reason: str
    source: Literal["prefilter", "llm"]


class GuardrailVerdict(BaseModel):
    """Whether the evidence is good enough to answer, and why not if not.
    detail stays a free dict since each rule explains itself differently."""

    triggered: bool
    rule: str | None = None
    detail: dict = Field(default_factory=dict)


class AskResponse(BaseModel):
    """Full response for POST /ask. insufficient_information is set from
    the guardrail's verdict in code - never from the LLM self-reporting
    confidence, since that's one prompt away from a confident-sounding
    guess."""

    answer: str
    used_detection: bool
    insufficient_information: bool
    detections: list[Detection] = Field(default_factory=list)
    evidence: dict | None = None
    reasoning_trace: dict = Field(default_factory=dict)
