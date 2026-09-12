"""
Writes the final answer text, from evidence only - never from the image.

This is the one function in the whole reasoning layer that is allowed to
produce open-ended prose, and I kept its job as narrow as I could: turn a
JSON summary of detections into a sentence a person would want to read. It
does not decide whether to call the detector (router's job) and it does not
decide whether the evidence is good enough to trust (guardrail's job). By the
time this function runs, both of those questions are already answered, and
its only remaining job is phrasing.

I use the larger Groq model here (gpt-oss-120b) rather than the router's
smaller one, because this is genuinely open-ended writing that has to read
naturally, whereas routing is a closed classification problem that the small
model handles fine under strict schema constraints.
"""

from __future__ import annotations

import os

from app.reasoning.evidence import Evidence
from app.reasoning.guardrail import GuardrailVerdict
from app.schemas import RouteDecision


def _evidence_to_json(evidence: Evidence) -> dict:
    """
    Flattens Evidence into plain JSON-serialisable data.

    This is the only thing the LLM ever receives about the image. No pixels,
    no base64, nothing that could let the model describe something it did not
    actually detect - it can only ever talk about what is in this dict.
    """
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
    """
    Produces the final answer text.

    I pass the guardrail's verdict into the prompt as a stated fact rather
    than asking the model to judge confidence itself - the whole point of
    building a separate deterministic guardrail is that this decision does
    not get re-litigated by the LLM. If verdict.triggered is True, the model
    is being told "the evidence is insufficient, explain why", not being
    asked "do you think this is enough to answer".
    """
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
        """
        If synthesis itself fails - Groq down, timeout, bad key - I still owe
        the caller a truthful answer rather than a stack trace. Falling back
        to a template built directly from the guardrail/evidence means the
        API stays honest about what it found even when the writing step that
        was supposed to phrase it nicely is unavailable.
        """
        if verdict.triggered:
            return (
                f"I cannot answer confidently: {verdict.rule}. "
                f"(Answer synthesis unavailable: {error})"
            )
        counts = ", ".join(f"{v} {k}" for k, v in evidence.class_counts.items()) or "nothing"
        return f"Detected: {counts}. (Answer synthesis unavailable: {error})"
