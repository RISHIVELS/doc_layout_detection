"""
These tests use stub detector and LLM clients rather than real ones, because
what I actually need to verify here is the *wiring* - does the pipeline call
things in the right order, does it skip the detector when it should, and does
the insufficient_information flag come from my guardrail rather than from
whatever the LLM decided to say. None of that needs a GPU or a real API key.

The one property I care about most is in
test_guardrail_verdict_overrides_whatever_the_llm_says: even if I hand the
pipeline a stub LLM that returns a confident-sounding answer, a triggered
guardrail must still force insufficient_information=True. If that property
ever broke, the guardrail would be decoration rather than a real gate.
"""

from app.reasoning.pipeline import answer_question
from app.schemas import BBox, Detection


class _StubDetector:
    """Records whether predict() was called, so tests can assert the
    pipeline correctly skipped the detector for out-of-scope questions."""

    def __init__(self, detections):
        self._detections = detections
        self.predict_called = False

    def predict(self, image, conf=0.25):
        self.predict_called = True
        return self._detections, 12.3


class _StubGroqResponse:
    def __init__(self, content: str):
        self.choices = [type("Choice", (), {"message": type("Msg", (), {"content": content})()})]


class _StubGroqClient:
    """
    Returns canned JSON for whichever call it receives, keyed by whether the
    request is asking the router's question or the synthesiser's. I keep
    this dumb on purpose - it is not trying to simulate a real model, only to
    let me control exactly what each stage of the pipeline receives.
    """

    def __init__(self, route_json: str, synth_answer: str = "A confident-sounding answer."):
        self._route_json = route_json
        self._synth_answer = synth_answer
        self.calls = []

    @property
    def chat(self):
        outer = self

        class _Completions:
            @staticmethod
            def create(**kwargs):
                outer.calls.append(kwargs)
                is_router_call = "route_decision" in str(kwargs.get("response_format", ""))
                content = outer._route_json if is_router_call else outer._synth_answer
                return _StubGroqResponse(content)

        class _Chat:
            completions = _Completions()

        return _Chat()


def d(name, class_id, confidence, x1=0, y1=0, x2=10, y2=10):
    return Detection(
        class_name=name, class_id=class_id, confidence=confidence,
        bbox=BBox(x1=x1, y1=y1, x2=x2, y2=y2),
    )


ROUTE_TABLE_COUNT = (
    '{"needs_detection": true, "target_classes": ["Table"], '
    '"task_type": "count", "reason": "counting tables"}'
)
ROUTE_OUT_OF_SCOPE = (
    '{"needs_detection": false, "target_classes": [], '
    '"task_type": "out_of_scope", "reason": "asks about text content"}'
)


def test_out_of_scope_question_never_touches_the_detector():
    """
    This is the efficiency half of the routing story, and also a correctness
    one: if the router said no detection is needed, calling the detector
    anyway would waste a GPU call and risk the evidence builder being handed
    detections that have nothing to do with the question being asked.
    """
    detector = _StubDetector([])
    client = _StubGroqClient(route_json=ROUTE_OUT_OF_SCOPE)

    response = answer_question(image=None, question="What is the invoice total?",
                                 detector=detector, client=client)

    assert detector.predict_called is False
    assert response.used_detection is False


def test_confident_detection_produces_a_normal_answer():
    detector = _StubDetector([d("Table", 8, 0.94)])
    client = _StubGroqClient(route_json=ROUTE_TABLE_COUNT, synth_answer="There is 1 table.")

    response = answer_question(image=None, question="How many tables?",
                                 detector=detector, client=client)

    assert detector.predict_called is True
    assert response.insufficient_information is False
    assert response.answer == "There is 1 table."


def test_guardrail_verdict_overrides_whatever_the_llm_says():
    """
    The load-bearing test in this file. I hand the pipeline a stub LLM that
    returns a perfectly confident-sounding sentence with no hedging in it at
    all, on evidence that should trip the guardrail (all detections below the
    confidence floor). insufficient_information must still come out True,
    set by my own code from the guardrail's verdict - never from whether the
    LLM's text happened to sound uncertain.
    """
    detector = _StubDetector([d("Table", 8, 0.31), d("Table", 8, 0.38)])
    client = _StubGroqClient(
        route_json=ROUTE_TABLE_COUNT,
        synth_answer="There are definitely 2 tables on this page.",
    )

    response = answer_question(image=None, question="How many tables?",
                                 detector=detector, client=client)

    assert response.insufficient_information is True
    assert response.reasoning_trace["guardrail"]["triggered"] is True


def test_no_detections_of_the_target_class_is_also_insufficient():
    detector = _StubDetector([d("Text", 9, 0.9)])
    client = _StubGroqClient(route_json=ROUTE_TABLE_COUNT, synth_answer="No tables found.")

    response = answer_question(image=None, question="How many tables?",
                                 detector=detector, client=client)

    assert response.insufficient_information is True
    assert response.reasoning_trace["guardrail"]["rule"] == "no_detections"


def test_reasoning_trace_records_both_stages_for_auditability():
    """
    reasoning_trace exists so a reviewer (or me, six weeks from now) can see
    exactly why the pipeline answered the way it did, without re-running it.
    """
    detector = _StubDetector([d("Table", 8, 0.94)])
    client = _StubGroqClient(route_json=ROUTE_TABLE_COUNT)

    response = answer_question(image=None, question="How many tables?",
                                 detector=detector, client=client)

    assert "router" in response.reasoning_trace
    assert "guardrail" in response.reasoning_trace
    assert response.reasoning_trace["router"]["task_type"] == "count"
