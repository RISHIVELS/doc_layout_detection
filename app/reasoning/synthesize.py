# Writes the final answer text from evidence only - never sees the image.
# Only job here is phrasing: router already decided whether to detect,
# guardrail already decided if the evidence is trustworthy. Uses the bigger
# Groq model (gpt-oss-120b) since this is open-ended writing, unlike
# routing which is closed classification the small model handles fine.
from __future__ import annotations

import os

from app.reasoning.evidence import Evidence
from app.reasoning.guardrail import GuardrailVerdict
from app.schemas import RouteDecision


def _evidence_to_json(evidence: Evidence) -> dict:
    """Flattens Evidence to plain JSON - the only thing the LLM ever sees
    about the image. No pixels, so it can't describe anything it didn't
    actually detect."""
    return {
        "class_counts": evidence.class_counts,
        "confidence_stats": {
            name: {"max": stats.max, "mean": round(stats.mean, 3), "min": stats.min}
            for name, stats in evidence.confidence_stats.items()
        },
        "total_detections": evidence.total_detections,
        "reading_order": evidence.reading_order,
        "relations": [
            {"kind": r.kind, "subject": r.subject, "object": r.object}
            for r in evidence.relations
        ],
    }


_SYSTEM_PROMPT = """You write short, plain-language answers about a document
page's layout, based ONLY on the structured evidence JSON you are given. You
have not seen the image - do not describe anything beyond what the evidence
contains.

If you are told the evidence is insufficient, say so plainly and explain why
in one sentence, using the guardrail detail provided. Do not hedge around it
or try to answer anyway - an explicit "I can't answer confidently" is the
correct and expected response in that case, not a failure to be worked around.

Otherwise, answer the question directly and concisely using the evidence.
Do not invent counts, classes, or relationships that are not in the evidence.
"""


def synthesize(
    question: str,
    evidence: Evidence,
    verdict: GuardrailVerdict,
    route: RouteDecision,
    client=None,
) -> str:
    """Produces the final answer text. Guardrail's verdict goes in as a
    stated fact, not something the model re-judges - if triggered, it's
    told "explain why this is insufficient", not asked "are you sure?"."""
    context = {
        "question": question,
        "task_type": route.task_type,
        "evidence": _evidence_to_json(evidence),
        "guardrail": {"insufficient": verdict.triggered, "rule": verdict.rule, "detail": verdict.detail},
    }

    if client is None:
        from groq import Groq

        client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

    model = os.environ.get("GROQ_SYNTHESIS_MODEL", "openai/gpt-oss-120b")

    try:
        import json

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(context)},
            ],
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()
    except Exception as error:
        # Groq down/timeout/bad key - fall back to a plain template built
        # from the guardrail/evidence instead of a stack trace, so the API
        # stays honest even when the writing step is unavailable.
        if verdict.triggered:
            return (
                f"I cannot answer confidently: {verdict.rule}. "
                f"(Answer synthesis unavailable: {error})"
            )
        counts = ", ".join(f"{v} {k}" for k, v in evidence.class_counts.items()) or "nothing"
        return f"Detected: {counts}. (Answer synthesis unavailable: {error})"
