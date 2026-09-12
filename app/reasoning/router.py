"""
Decides whether answering a question requires calling the detector at all.

This is the first stage of Part B, and it is split into two layers on purpose:

1. A cheap, deterministic regex prefilter for questions that obviously do not
   need vision at all - greetings, meta-questions, general knowledge. No
   reason to spend a Groq call and a few hundred milliseconds deciding that
   "what is the capital of France" has nothing to do with the uploaded image.

2. An LLM call, with Groq's strict `json_schema` mode, for everything the
   prefilter is not confident about. Routing genuinely needs language
   understanding - "how many tables" and "is this page well-organised" both
   mention the page, but only one needs the detector - and I do not want a
   hand-rolled keyword list quietly making that call badly.

The one thing I built into the LLM prompt on purpose, rather than leaving it
implicit: the model has to know it is routing for a *layout* detector, not a
general vision model. A question like "what is the invoice total?" mentions
the image and sounds like it needs detection, but my detector cannot read
text - it only knows where regions are, not what they say. Getting this
distinction right in the prompt is what makes the out-of-scope refusal in my
memo a real architectural boundary instead of a coincidence.
"""

from __future__ import annotations

import os
import re

from app.constants import CLASS_NAMES
from app.schemas import RouteDecision

# Patterns for questions the prefilter can answer with confidence, without
# spending an LLM call. I kept this list short and specific rather than
# trying to be clever with it - the moment a pattern here is even slightly
# ambiguous, it belongs with the LLM, not here. A wrong prefilter decision is
# worse than a slow one, because it skips the LLM's judgement entirely.
_GREETING_PATTERN = re.compile(
    r"^\s*(hi|hello|hey|thanks|thank you)[\s!.,]*$", re.IGNORECASE
)
_META_PATTERN = re.compile(
    r"\b(what can you do|who are you|what are you|how do you work)\b", re.IGNORECASE
)
_GENERAL_KNOWLEDGE_PATTERN = re.compile(
    r"\b(capital of|president of|weather|what year is it|who won)\b", re.IGNORECASE
)


def prefilter(question: str) -> RouteDecision | None:
    """
    Returns a RouteDecision for the small set of questions I am confident
    about without asking the LLM, or None to defer to the LLM router.

    Returning None is the important design point here: this function is only
    ever allowed to say "definitely no detection needed", never "definitely
    yes". Deciding that a question *does* need the detector, and which
    classes it needs, requires actually understanding the sentence - that is
    exactly the job I want the LLM doing, not a regex.
    """
    stripped = question.strip()

    if _GREETING_PATTERN.match(stripped):
        return RouteDecision(
            needs_detection=False, target_classes=[], task_type="out_of_scope",
            reason="Greeting, not a question about the image.", source="prefilter",
        )

    if _META_PATTERN.search(stripped) or _GENERAL_KNOWLEDGE_PATTERN.search(stripped):
        return RouteDecision(
            needs_detection=False, target_classes=[], task_type="out_of_scope",
            reason="General-knowledge or meta question, unrelated to image content.",
            source="prefilter",
        )

    return None


_ROUTER_SCHEMA = {
    "name": "route_decision",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "needs_detection": {"type": "boolean"},
            "target_classes": {
                "type": "array",
                "items": {"type": "string", "enum": CLASS_NAMES},
            },
            "task_type": {
                "type": "string",
                "enum": ["count", "presence", "compare", "describe", "out_of_scope"],
            },
            "reason": {"type": "string"},
        },
        "required": ["needs_detection", "target_classes", "task_type", "reason"],
        "additionalProperties": False,
    },
}

_SYSTEM_PROMPT = f"""You route questions for a document layout detection API.

The detector can ONLY report the position and type of structural regions on a
document page. It knows WHERE things are, never WHAT they say. Its full
vocabulary of classes is exactly this list, nothing more:
{", ".join(CLASS_NAMES)}

Decide:
- needs_detection: does answering this actually require knowing what regions
  are on the page? Questions about text CONTENT the detector cannot read
  (amounts, names, dates, what a paragraph says, who signed something) are
  OUT OF SCOPE even though they mention the document - set needs_detection to
  false and task_type to "out_of_scope" for these. Do not guess at content the
  detector cannot see.
- target_classes: which of the classes above the question is actually about.
  Empty if needs_detection is false.
- task_type: "count" (how many), "presence" (is there a/any), "compare"
  (relationships between regions), "describe" (general layout question), or
  "out_of_scope".
- reason: one short sentence explaining the decision.
"""


def route(question: str, client=None) -> RouteDecision:
    """
    Full routing decision: prefilter first, then the LLM if the prefilter
    deferred.

    `client` is injectable so tests and the pipeline can supply a stub instead
    of a real Groq client - see app/reasoning/pipeline.py for how it is wired
    in production.

    On any LLM error - a bad key, a timeout, a schema violation Groq itself
    could not satisfy - I fail closed to out_of_scope rather than guessing
    that detection is needed. Wrongly skipping a real question is a worse
    failure than wrongly declining an answerable one; the second at least
    tells the user something true.
    """
    prefiltered = prefilter(question)
    if prefiltered is not None:
        return prefiltered

    if client is None:
        from groq import Groq

        client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

    model = os.environ.get("GROQ_ROUTER_MODEL", "openai/gpt-oss-20b")

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            response_format={"type": "json_schema", "json_schema": _ROUTER_SCHEMA},
            temperature=0,
        )
        import json

        parsed = json.loads(response.choices[0].message.content)
        return RouteDecision(
            needs_detection=parsed["needs_detection"],
            target_classes=parsed["target_classes"],
            task_type=parsed["task_type"],
            reason=parsed["reason"],
            source="llm",
        )
    except Exception as error:
        return RouteDecision(
            needs_detection=False,
            target_classes=[],
            task_type="out_of_scope",
            reason=f"Router failed closed after an error: {error}",
            source="llm",
        )
