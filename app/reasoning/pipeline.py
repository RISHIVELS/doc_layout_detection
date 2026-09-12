"""
The one function that knows the full order of Part B: route, maybe detect,
build evidence, check the guardrail, synthesise. Every other file in
app/reasoning/ does one job in isolation; this is where the jobs get
sequenced.

I kept the sequencing itself in one small, boring function on purpose. The
interesting decisions - what counts as insufficient, what counts as
out-of-scope, how the answer gets phrased - all live in the modules this
function calls, each with their own tests. This file has almost no logic of
its own to get wrong.
"""

from __future__ import annotations

from app.reasoning.evidence import build_evidence
from app.reasoning.guardrail import evaluate_guardrail
from app.reasoning.router import route as route_question
from app.reasoning.synthesize import synthesize
from app.schemas import AskResponse


def answer_question(image, question: str, detector, client=None) -> AskResponse:
    """
    Runs the full Part B pipeline for one question against one image.

    `detector` and `client` are both injectable - see tests/test_pipeline.py,
    which stubs both to verify the wiring without a GPU or a real Groq key.
    In production, app/main.py supplies the real Detector instance and a real
    Groq client (or lets router.py/synthesize.py construct their own default
    clients when None is passed through).
    """
    route = route_question(question, client=client)

    detections = []
    if route.needs_detection:
        """
        The detector only runs when the router said it is actually needed.
        This matters for more than speed: it also means a question the router
        correctly judged out-of-scope never gets contaminated by detections
        that have nothing to do with what was asked.
        """
        detections, _inference_ms = detector.predict(image)

    evidence = build_evidence(detections, image_width=0, image_height=0)

    """
    Query-budget saturation only means anything for a real detector on a real
    page - a stub in tests has no notion of it, so I check for the attribute
    rather than assuming every detector object exposes it.
    """
    saturated = (
        route.needs_detection
        and hasattr(detector, "query_budget")
        and evidence.total_detections >= detector.query_budget
    )

    verdict = evaluate_guardrail(
        evidence, target_classes=route.target_classes,
        task_type=route.task_type, saturated=saturated,
    ) if route.needs_detection else evaluate_guardrail(evidence, [], route.task_type)

    """
    This is the line the whole design rests on. insufficient_information is
    set directly from the guardrail's verdict - a fact my own deterministic
    code computed from the evidence - never from asking the LLM to self-
    report how confident it feels. See test_pipeline.py's
    test_guardrail_verdict_overrides_whatever_the_llm_says for the check that
    this stays true even when the LLM's own wording sounds perfectly certain.
    """
    if route.needs_detection:
        answer_text = synthesize(question, evidence, verdict, route, client=client)
        insufficient = verdict.triggered
    else:
        """
        Out-of-scope questions never reach the guardrail meaningfully (there
        is no detection evidence to evaluate), but they are still an
        "insufficient information" case in the sense the brief cares about:
        the system is declining to answer rather than guessing. I answer
        directly here without a second LLM call, since the router's `reason`
        field already explains why.
        """
        answer_text = (
            f"I can't answer that from this image's layout: {route.reason}"
        )
        insufficient = True

    return AskResponse(
        answer=answer_text,
        used_detection=route.needs_detection,
        insufficient_information=insufficient,
        detections=detections,
        evidence={
            "class_counts": evidence.class_counts,
            "total_detections": evidence.total_detections,
        } if route.needs_detection else None,
        reasoning_trace={
            "router": {
                "needs_detection": route.needs_detection,
                "target_classes": route.target_classes,
                "task_type": route.task_type,
                "reason": route.reason,
                "source": route.source,
            },
            "guardrail": {
                "triggered": verdict.triggered,
                "rule": verdict.rule,
                "detail": verdict.detail,
            },
        },
    )
