# Wires Part B together: route -> maybe detect -> evidence -> guardrail ->
# synthesize. All the actual decisions live in those modules, each tested
# separately - this file is just the order they run in.
from __future__ import annotations

from app.reasoning.evidence import build_evidence
from app.reasoning.guardrail import evaluate_guardrail
from app.reasoning.router import route as route_question
from app.reasoning.synthesize import synthesize
from app.schemas import AskResponse


def answer_question(image, question: str, detector, client=None) -> AskResponse:
    """Runs the full pipeline for one question against one image.

    detector/client are injectable - tests stub both, no GPU or Groq key
    needed. See test_pipeline.py."""
    route = route_question(question, client=client)

    detections = []
    if route.needs_detection:
        # only run the detector if the router actually says it's needed -
        # keeps out-of-scope questions from getting unrelated detections
        detections, _inference_ms = detector.predict(image)

    evidence = build_evidence(detections, image_width=0, image_height=0)

    # stub detectors in tests don't have a query_budget - check first
    saturated = (
        route.needs_detection
        and hasattr(detector, "query_budget")
        and evidence.total_detections >= detector.query_budget
    )

    verdict = evaluate_guardrail(
        evidence, target_classes=route.target_classes,
        task_type=route.task_type, saturated=saturated,
    ) if route.needs_detection else evaluate_guardrail(evidence, [], route.task_type)

    # insufficient_information comes from the guardrail's verdict, not from
    # the LLM's wording - see test_guardrail_verdict_overrides_whatever_the_llm_says
    if route.needs_detection:
        answer_text = synthesize(question, evidence, verdict, route, client=client)
        insufficient = verdict.triggered
    else:
        # out-of-scope questions skip synthesis entirely - router's reason
        # already explains why, no need for a second LLM call
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
