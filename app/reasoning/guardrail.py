"""
Decides whether the evidence is good enough to answer confidently, before the
LLM ever gets a turn to write anything.

I built this as a separate, pure module rather than folding the check into the
synthesis prompt because of a specific thing I do not trust: asking a model
"are you confident enough to answer?" gets you text, and a model asked that
question is roughly as capable of writing a confident-sounding justification
for guessing as it is of writing an honest refusal. Both are just tokens it is
equally fluent at producing.

So the check happens here, in code, before the LLM sees anything, and the
result is handed to it as a fact rather than a question. Section 5 and 6 of
the design spec walk through why this is the load-bearing decision in Part B.

The four rules are checked in a fixed order, most specific first, because two
of them can technically both be true at once and I want the more informative
explanation to win. "No detections at all" is a more useful thing to tell a
user than "the page happened to be at its query budget", even on a page where
both are technically true.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.reasoning.evidence import Evidence

# Below this, I do not trust a single detection enough to build an answer on
# it. I picked 0.50 rather than something more conservative like 0.7 because
# RT-DETR's confidence scores for correctly localised regions on this dataset
# cluster well above 0.6 in my own validation runs - 0.50 sits below the
# "genuinely doing its job" range without being so low that it never fires.
CONFIDENCE_FLOOR = 0.50

# For a counting question specifically, I do not want to report an exact
# number when a large chunk of the supposed count is sitting in the shaky
# 0.25-0.50 band. Below 0.25 I already treat as noise the model should not
# have reported in the first place; between 0.25 and CONFIDENCE_FLOOR is
# "plausible but not something I'd stake a specific number on".
AMBIGUOUS_BAND = (0.25, CONFIDENCE_FLOOR)
AMBIGUOUS_BAND_FRACTION = 0.30


@dataclass
class GuardrailVerdict:
    triggered: bool
    rule: str | None = None
    detail: dict = field(default_factory=dict)


def _confidences_for(evidence: Evidence, target_classes: list[str]) -> list[float]:
    """Pulls every confidence score belonging to the classes the question is
    actually about, ignoring detections of unrelated classes on the same page."""
    values: list[float] = []
    for class_name in target_classes:
        stats = evidence.confidence_stats.get(class_name)
        if stats is None:
            continue
        # confidence_stats only carries max/mean/min, not the raw list, which
        # is enough for every rule except the ambiguous-band fraction below.
        # That rule needs the raw values, so evaluate_guardrail re-derives
        # them from the caller-supplied per-detection list instead.
        values.append(stats.max)
    return values


def evaluate_guardrail(
    evidence: Evidence,
    target_classes: list[str],
    task_type: str,
    saturated: bool = False,
    raw_confidences: list[float] | None = None,
) -> GuardrailVerdict:
    """
    Checks the evidence against the four honesty rules, most specific first.

    `raw_confidences` is optional and only needed for the ambiguous-band rule,
    which has to look at the actual distribution of scores rather than the
    summary stats in `Evidence`. I made it optional rather than mandatory
    because most callers (and most of these tests) only care about the
    simpler rules, and computing it is the pipeline's job, not every test's.
    """
    target_counts = {
        class_name: evidence.class_counts.get(class_name, 0)
        for class_name in target_classes
    }
    total_target_detections = sum(target_counts.values())

    # Rule 1: nothing of the target class was found at all. This is the most
    # informative thing to tell the user, so it wins even on a saturated page.
    if total_target_detections == 0:
        return GuardrailVerdict(
            triggered=True,
            rule="no_detections",
            detail={"target_classes": target_classes},
        )

    max_confidences = _confidences_for(evidence, target_classes)
    observed_max = max(max_confidences) if max_confidences else 0.0

    # Rule 2: something was found, but nothing about it clears the floor.
    if observed_max < CONFIDENCE_FLOOR:
        return GuardrailVerdict(
            triggered=True,
            rule="all_below_tau",
            detail={"tau": CONFIDENCE_FLOOR, "observed_max": observed_max},
        )

    # Rule 3: only matters for counting. A presence question ("is there a
    # Table on this page?") is answered by the single best detection, so a
    # couple of shaky extra candidates next to a confident one do not make
    # the answer to "is there one" any less true.
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

    # Rule 4: the page had more ground-truth-scale regions than RT-DETR's
    # fixed query budget could possibly emit. A count off a saturated page is
    # unreliable for a reason that has nothing to do with confidence - some
    # regions never got the chance to be detected at any score.
    if saturated:
        return GuardrailVerdict(triggered=True, rule="query_saturated", detail={})

    return GuardrailVerdict(triggered=False)
