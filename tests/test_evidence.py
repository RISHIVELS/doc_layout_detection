"""
build_evidence() is where raw detections turn into the summary the LLM and the
guardrail both work from. It has to be pure and deterministic - no I/O, no
model calls - because it is the one place I can guarantee the LLM never sees
anything except numbers I computed myself. If this function ever quietly
started passing image bytes through, the whole "the LLM cannot hallucinate
visual content because it never sees pixels" argument in my memo would be
false.
"""

from app.reasoning.evidence import build_evidence
from app.schemas import BBox, Detection


def d(name, class_id, confidence, x1=0, y1=0, x2=10, y2=10):
    """Shorthand for building a Detection in tests without the ceremony."""
    return Detection(
        class_name=name, class_id=class_id, confidence=confidence,
        bbox=BBox(x1=x1, y1=y1, x2=x2, y2=y2),
    )


def test_counts_detections_by_class():
    evidence = build_evidence(
        [d("Table", 8, 0.9), d("Table", 8, 0.8), d("Text", 9, 0.7)], 1025, 1025
    )
    assert evidence.class_counts == {"Table": 2, "Text": 1}


def test_confidence_stats_are_computed_per_class_not_globally():
    """
    Per-class stats matter because the guardrail asks "is the model sure about
    Table specifically", not "is the model sure about something on this page".
    A page with one confident Text block and three shaky Table guesses should
    not look confident just because the average across everything is high.
    """
    evidence = build_evidence([d("Table", 8, 0.9), d("Table", 8, 0.5)], 1025, 1025)
    assert evidence.confidence_stats["Table"].max == 0.9
    assert evidence.confidence_stats["Table"].mean == 0.7
    assert evidence.confidence_stats["Table"].min == 0.5


def test_an_image_with_no_detections_produces_empty_evidence_not_a_crash():
    # A blank page or a page where nothing crossed the confidence floor is a
    # completely normal input, and the guardrail relies on this case existing
    # cleanly to trigger its own "no_detections" rule.
    evidence = build_evidence([], 1025, 1025)
    assert evidence.class_counts == {}
    assert evidence.total_detections == 0


def test_reading_order_sorts_top_to_bottom_then_left_to_right():
    """
    This is my stand-in for real reading-order reconstruction, which is out of
    scope for this project (see the design doc). It is a genuine
    simplification and I say so in the memo - a real two-column page would
    break this sort - but top-to-bottom-then-left-to-right is still the right
    default assumption for the single-column documents this dataset mostly
    contains.
    """
    lower_left = d("Text", 9, 0.9, x1=0, x2=100, y1=500, y2=600)
    upper_right = d("Table", 8, 0.9, x1=600, x2=700, y1=100, y2=200)
    upper_left = d("Title", 10, 0.9, x1=0, x2=100, y1=100, y2=200)

    evidence = build_evidence([lower_left, upper_right, upper_left], 1025, 1025)

    assert evidence.reading_order == ["Title", "Table", "Text"]


def test_detects_when_one_region_visually_contains_another():
    """
    A Caption sitting inside a Table's extent is a real, common layout pattern
    (a caption printed inside a bordered table region), and it is the kind of
    relationship the synthesis step needs in order to answer a question like
    "what is this table about" without hallucinating a connection that is not
    actually there in the boxes.
    """
    outer_table = d("Table", 8, 0.9, x1=0, y1=0, x2=500, y2=500)
    inner_caption = d("Caption", 0, 0.9, x1=10, y1=10, x2=100, y2=50)

    evidence = build_evidence([outer_table, inner_caption], 1025, 1025)

    assert any(
        relation.kind == "contains"
        and relation.subject == "Table"
        and relation.object == "Caption"
        for relation in evidence.relations
    )


def test_two_unrelated_regions_produce_no_relation():
    far_apart_a = d("Text", 9, 0.9, x1=0, y1=0, x2=50, y2=50)
    far_apart_b = d("Text", 9, 0.9, x1=900, y1=900, x2=950, y2=950)

    evidence = build_evidence([far_apart_a, far_apart_b], 1025, 1025)

    assert evidence.relations == []
