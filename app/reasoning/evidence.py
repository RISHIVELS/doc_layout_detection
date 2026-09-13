# Detections -> structured summary. Pure functions only, no I/O, no model
# calls, nothing that touches pixels - this is what makes it true that the
# LLM downstream can't hallucinate what's on the page: it never sees it,
# only this summary.
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
    """A spatial fact between two regions, e.g. Caption inside a Table.
    Own type instead of a dict so a typo'd key fails loudly, not silently."""

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
    # Raw scores per class - guardrail's ambiguous-band rule needs the full
    # distribution, not just max/mean/min.
    confidences_by_class: dict[str, list[float]] = field(default_factory=dict)


def _box_area(detection: Detection) -> float:
    b = detection.bbox
    return (b.x2 - b.x1) * (b.y2 - b.y1)


def _contains(outer: Detection, inner: Detection, threshold: float = 0.85) -> bool:
    """True if inner sits mostly (85%+) inside outer. Not 100% - annotators
    don't draw pixel-perfect boxes, a caption box a few px outside its
    table shouldn't make the relationship disappear."""
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
    """Main entry point: detections -> Evidence.

    image_width/height aren't used yet - kept for when a relation needs
    page-edge position (e.g. "is this near the top" for Page-header)."""
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

    # Top-to-bottom, then left-to-right. Not real reading-order logic - a
    # two-column page would interleave columns wrong under this sort. Good
    # enough for DocLayNet's mostly-single-column pages, noted in the memo.
    ordered = sorted(detections, key=lambda det: (det.bbox.y1, det.bbox.x1))
    reading_order = [det.class_name for det in ordered]

    relations: list[Relation] = []
    for outer in detections:
        for inner in detections:
            if outer is inner:
                continue
            if _box_area(inner) >= _box_area(outer):
                continue  # can't "contain" something the same size or bigger
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
