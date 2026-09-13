# Decides if a question needs the detector at all. Two layers:
# 1. Cheap regex prefilter for the obvious cases (greetings, general
#    knowledge) - no point spending a Groq call on "what's the capital of
#    France".
# 2. LLM call (Groq, strict json_schema) for everything else. Routing
#    genuinely needs language understanding - a keyword list can't tell
#    "how many tables" from "is this page well organised".
#
# Key thing baked into the prompt: the model knows it's routing for a
# *layout* detector, not general vision. "What's the invoice total?"
# mentions the image but the detector can't read text - only a prompt that
# says so explicitly catches that and routes it out_of_scope.
from __future__ import annotations

import os
import re

from app.constants import CLASS_NAMES
from app.schemas import RouteDecision

# Kept short and specific on purpose - if a pattern is even slightly
# ambiguous it belongs with the LLM. A wrong prefilter call is worse than a
# slow one since it skips the LLM's judgement entirely.
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
    """Only ever returns "definitely no detection needed", never "yes" -
    deciding detection IS needed requires actually understanding the
    sentence, which is the LLM's job. None means defer to it."""
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
    """Prefilter first, then the LLM if it deferred. `client` is injectable
    for tests/pipeline to swap in a stub - see pipeline.py.

    Fails closed to out_of_scope on any LLM error (bad key, timeout, etc)
    rather than guessing detection is needed - skipping a real question is
    worse than declining an answerable one."""
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
