"""
The whole reason I built a separate guardrail module instead of just asking
the LLM "are you confident?" is that I do not trust that question. A model
asked to self-report confidence will produce a plausible-sounding
justification for guessing anyway roughly as often as it produces an honest
refusal, because both are just text it is equally capable of generating.

So the guardrail is plain Python, evaluated on the evidence before the LLM is
asked to write anything, and its verdict is a fact the LLM is handed, not a
question it answers. These tests are really testing my own judgment calls
about where the lines sit (tau=0.50, the 30% ambiguous-band rule) as much as
the code, so I wanted them written down explicitly.
"""

from app.reasoning.evidence import build_evidence
from app.reasoning.guardrail import evaluate_guardrail
from app.schemas import BBox, Detection


def d(name, class_id, confidence, x1=0, y1=0, x2=10, y2=10):
    return Detection(
        class_name=name, class_id=class_id, confidence=confidence,
        bbox=BBox(x1=x1, y1=y1, x2=x2, y2=y2),
    )


def ev(detections):
    return build_evidence(detections, 1025, 1025)


def test_no_detections_of_the_target_class_triggers_a_refusal():
    """
    The most basic honesty case: asked about Tables, found none. I want this
    to say "I found no tables" rather than let the LLM improvise around an
    empty evidence dict.
    """
    verdict = evaluate_guardrail(ev([d("Text", 9, 0.9)]), ["Table"], "count")
    assert verdict.triggered is True
    assert verdict.rule == "no_detections"


def test_target_class_present_but_every_detection_is_below_the_confidence_floor():
    """
    This is my primary 'insufficient information' example for the memo: a
    dense financial report where the detector finds three plausible Table
    regions but is not confident about any of them.
    """
    verdict = evaluate_guardrail(
        ev([d("Table", 8, 0.31), d("Table", 8, 0.38), d("Table", 8, 0.44)]),
        ["Table"], "count",
    )
    assert verdict.triggered is True
    assert verdict.rule == "all_below_tau"
    assert verdict.detail["observed_max"] == 0.44


def test_a_single_confident_detection_does_not_trigger_anything():
    verdict = evaluate_guardrail(ev([d("Table", 8, 0.94)]), ["Table"], "count")
    assert verdict.triggered is False
    assert verdict.rule is None


def test_tau_boundary_is_exclusive_below_not_at():
    # I picked 0.50 as the floor. A detection sitting exactly at 0.50 should
    # count as confident enough - the rule is "below", not "at or below".
    verdict = evaluate_guardrail(ev([d("Table", 8, 0.50)]), ["Table"], "count")
    assert verdict.triggered is False


def test_ambiguous_band_triggers_when_counting_and_a_large_minority_is_shaky():
    """
    A case that is not confidently absent (that's no_detections) and not
    uniformly unconfident (that's all_below_tau), but where enough of the
    count is shaky that reporting an exact number would be dishonest. Three
    Tables, one solid, two sitting in the 0.25-0.50 band - I do not want to
    report "3 tables" as if all three were equally certain.
    """
    detections = [d("Table", 8, 0.95), d("Table", 8, 0.30), d("Table", 8, 0.35)]
    verdict = evaluate_guardrail(ev(detections), ["Table"], "count")
    assert verdict.triggered is True
    assert verdict.rule == "ambiguous_band"


def test_ambiguous_band_rule_does_not_apply_to_a_presence_question():
    """
    'Is there a table on this page?' only needs one confident hit to answer
    yes - the shakiness of two other candidate boxes is irrelevant to that
    question, so this rule should not fire for task_type=presence even with
    the exact same detections as the counting case above.
    """
    detections = [d("Table", 8, 0.95), d("Table", 8, 0.30), d("Table", 8, 0.35)]
    verdict = evaluate_guardrail(ev(detections), ["Table"], "presence")
    assert verdict.triggered is False


def test_query_saturation_triggers_when_the_page_exceeded_the_detection_budget():
    """
    If the page had more regions than RT-DETR can emit, a count is
    structurally unreliable regardless of how confident the returned
    detections happen to be - the ones that got cut off never had a chance to
    appear at any confidence.
    """
    verdict = evaluate_guardrail(
        ev([d("Text", 9, 0.99)]), ["Text"], "count", saturated=True
    )
    assert verdict.triggered is True
    assert verdict.rule == "query_saturated"


def test_rule_precedence_favours_the_most_specific_explanation():
    # If there are simply no detections at all, that is the more informative
    # thing to tell the user - "saturated" would be misleading here since a
    # page with zero of the target class did not run out of query budget for
    # that class specifically.
    verdict = evaluate_guardrail(ev([]), ["Table"], "count", saturated=True)
    assert verdict.rule == "no_detections"
