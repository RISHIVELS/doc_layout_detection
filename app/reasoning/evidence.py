"""
Turns a raw list of detections into the structured summary that the guardrail
checks and the LLM writes its answer from.

This file is deliberately the most boring code in the reasoning layer, and
that is the point. Everything in here is a pure function over numbers I
already have - no network calls, no model inference, nothing that could reach
out and grab a pixel. That is not an implementation detail I happen to like;
it is the load-bearing claim in my memo. I say the LLM cannot hallucinate what
is on the page because it never sees the page, only this summary. That claim
is only true if this file stays exactly this boring forever.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.schemas import Detection


@dataclass
class ConfStats:
    max: float
    mean: float
    min: float


@dataclass
class Relation:
    """One spatial fact between two detected regions, e.g. a Caption sitting
    inside a Table's extent. Kept as its own small type rather than a dict so
    a typo in a key name (`"suject"`) fails at construction time instead of
    silently vanishing into a KeyError three calls later."""

    kind: str
    subject: str
    object: str


@dataclass
class Evidence:
    class_counts: dict[str, int]
    confidence_stats: dict[str, ConfStats]
    total_detections: int
    reading_order: list[str]
    relations: list[Relation] = field(default_factory=list)
    # Raw per-detection confidence scores, kept alongside the summary stats
    # above. I need these in the guardrail for the ambiguous-band rule, which
    # asks "what fraction of the scores sit in a shaky range" - a question
    # max/mean/min cannot answer on their own.
    confidences_by_class: dict[str, list[float]] = field(default_factory=dict)


def _box_area(detection: Detection) -> float:
    b = detection.bbox
    return (b.x2 - b.x1) * (b.y2 - b.y1)


def _contains(outer: Detection, inner: Detection, threshold: float = 0.85) -> bool:
    """
    True when `inner`'s box sits mostly inside `outer`'s box.

    I use "mostly inside" (85% of the inner box's own area overlapping) rather
    than requiring perfect containment, because DocLayNet's own human
    annotators do not draw pixel-perfect boxes - a caption box drawn a couple
    of pixels outside its parent table's border is completely normal and I do
    not want that kind of ordinary annotation noise to make a real
    caption-inside-table relationship disappear.
    """
    ox1, oy1, ox2, oy2 = outer.bbox.x1, outer.bbox.y1, outer.bbox.x2, outer.bbox.y2
    ix1, iy1, ix2, iy2 = inner.bbox.x1, inner.bbox.y1, inner.bbox.x2, inner.bbox.y2

    overlap_x1, overlap_y1 = max(ox1, ix1), max(oy1, iy1)
    overlap_x2, overlap_y2 = min(ox2, ix2), min(oy2, iy2)
    if overlap_x2 <= overlap_x1 or overlap_y2 <= overlap_y1:
        return False

    overlap_area = (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1)
    inner_area = _box_area(inner)
    return inner_area > 0 and (overlap_area / inner_area) >= threshold


def build_evidence(detections: list[Detection], image_width: int, image_height: int) -> Evidence:
    """
    The single entry point the rest of the reasoning layer uses to go from
    "here is what the detector saw" to "here is a summary an LLM can reason
    over safely".

    image_width and image_height are accepted even though the current
    functions do not use them, because the moment I add a relation that needs
    to reason about position relative to the page edge - "is this a
    Page-header" - I want that plumbing already in place rather than having to
    thread it through every caller a second time.
    """
    class_counts: dict[str, int] = {}
    confidences_by_class: dict[str, list[float]] = {}

    for detection in detections:
        class_counts[detection.class_name] = class_counts.get(detection.class_name, 0) + 1
        confidences_by_class.setdefault(detection.class_name, []).append(detection.confidence)

    confidence_stats = {
        class_name: ConfStats(
            max=max(values), mean=sum(values) / len(values), min=min(values)
        )
        for class_name, values in confidences_by_class.items()
    }

    """
    Reading order: sort top-to-bottom by the box's top edge, then left-to-right
    for anything roughly on the same line.

    This is a genuine simplification, not a real reading-order algorithm - a
    true two-column page would interleave the columns under this sort, which
    is wrong. I chose it anyway because DocLayNet's pages are mostly single-
    column, it costs nothing to compute, and it is honest about what it is: a
    reasonable default, not a claim that I solved document reading order. I
    say so explicitly in the memo rather than let a confusion matrix quietly
    imply the model got column ordering wrong when actually my sort did.
    """
    ordered = sorted(detections, key=lambda det: (det.bbox.y1, det.bbox.x1))
    reading_order = [det.class_name for det in ordered]

    relations: list[Relation] = []
    for outer in detections:
        for inner in detections:
            if outer is inner:
                continue
            if _box_area(inner) >= _box_area(outer):
                # A region cannot "contain" another region that is the same
                # size or larger than it - that would just be two overlapping
                # regions, which is a different (and, for DocLayNet, much
                # rarer and less meaningful) relationship.
                continue
            if _contains(outer, inner):
                relations.append(Relation(kind="contains", subject=outer.class_name, object=inner.class_name))

    return Evidence(
        class_counts=class_counts,
        confidence_stats=confidence_stats,
        total_detections=len(detections),
        reading_order=reading_order,
        relations=relations,
        confidences_by_class=confidences_by_class,
    )
