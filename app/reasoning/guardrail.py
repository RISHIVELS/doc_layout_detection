# Decides if the evidence is good enough to answer confidently - before the
# LLM writes anything. Kept as a separate deterministic check because asking
# a model "are you confident?" just gets more text, not a real signal - it's
# equally fluent at a confident-sounding guess as an honest refusal.
from __future__ import annotations

from dataclasses import dataclass, field

from app.reasoning.evidence import Evidence

# Below this, a single detection isn't trustworthy enough to answer from.
# 0.50 rather than something stricter like 0.7 - correctly localised
# regions on this dataset tend to score well above 0.6 in my own runs.
CONFIDENCE_FLOOR = 0.50

# For counting: don't report an exact number if too much of it sits in this
# shaky band. Below 0.25 is just noise; 0.25-0.50 is "plausible but I
# wouldn't bet on the count."
AMBIGUOUS_BAND = (0.25, CONFIDENCE_FLOOR)
AMBIGUOUS_BAND_FRACTION = 0.30


@dataclass
class GuardrailVerdict:
    triggered: bool
    rule: str | None = None
    detail: dict = field(default_factory=dict)


def _confidences_for(evidence: Evidence, target_classes: list[str]) -> list[float]:
    """Max confidence per target class, ignoring unrelated classes on the page."""
    values: list[float] = []
    for class_name in target_classes:
        stats = evidence.confidence_stats.get(class_name)
        if stats is None:
            continue
        values.append(stats.max)
    return values


def evaluate_guardrail(
    evidence: Evidence,
    target_classes: list[str],
    task_type: str,
    saturated: bool = False,
) -> GuardrailVerdict:
    """Four rules, checked in order, most specific first - some can overlap
    and I want the more useful explanation to win (e.g. "found nothing"
    beats "page was saturated" even when both are technically true)."""
    raw_confidences = [
        value
        for class_name in target_classes
        for value in evidence.confidences_by_class.get(class_name, [])
    ]
    target_counts = {
        class_name: evidence.class_counts.get(class_name, 0)
        for class_name in target_classes
    }
    total_target_detections = sum(target_counts.values())

    # nothing of the target class found at all
    if total_target_detections == 0:
        return GuardrailVerdict(
            triggered=True,
            rule="no_detections",
            detail={"target_classes": target_classes},
        )

    max_confidences = _confidences_for(evidence, target_classes)
    observed_max = max(max_confidences) if max_confidences else 0.0

    # found something, but nothing clears the confidence floor
    if observed_max < CONFIDENCE_FLOOR:
        return GuardrailVerdict(
            triggered=True,
            rule="all_below_tau",
            detail={"tau": CONFIDENCE_FLOOR, "observed_max": observed_max},
        )

    # counting only: too much of the count sits in the shaky band.
    # doesn't apply to "is there one?" - one confident hit is enough for that.
    if task_type == "count" and raw_confidences:
        low, high = AMBIGUOUS_BAND
        in_band = sum(1 for c in raw_confidences if low <= c < high)
        if raw_confidences and (in_band / len(raw_confidences)) > AMBIGUOUS_BAND_FRACTION:
            return GuardrailVerdict(
                triggered=True,
                rule="ambiguous_band",
                detail={
                    "band": list(AMBIGUOUS_BAND),
                    "fraction_in_band": round(in_band / len(raw_confidences), 2),
                },
            )

    # page had more regions than RT-DETR's query budget could emit -
    # a count here is unreliable regardless of confidence
    if saturated:
        return GuardrailVerdict(triggered=True, rule="query_saturated", detail={})

    return GuardrailVerdict(triggered=False)
