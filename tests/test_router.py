"""
I split routing into two layers: a cheap regex prefilter for the obvious cases,
and an LLM call for everything else. These tests only cover the prefilter,
because it is the pure, deterministic half - no network, nothing that needs a
real Groq key to exercise. The LLM half (route()) I checked by hand against a
real key and recorded the transcript in the build log, which is the honest way
to verify a prompt's behaviour rather than pretending a mocked test proves
anything about what a real model will do.
"""

from app.reasoning.router import prefilter


def test_prefilter_catches_a_question_with_nothing_to_do_with_the_image():
    decision = prefilter("What is the capital of France?")
    assert decision is not None
    assert decision.needs_detection is False
    assert decision.task_type == "out_of_scope"
    assert decision.source == "prefilter"


def test_prefilter_leaves_visual_questions_for_the_llm_to_route():
    # "How many tables" clearly needs the detector, but deciding that requires
    # understanding the sentence, not just pattern-matching a keyword list -
    # so this is exactly the kind of question the prefilter should NOT try to
    # handle itself.
    assert prefilter("How many tables are on this page?") is None


def test_prefilter_leaves_genuinely_ambiguous_questions_for_the_llm():
    # A prefilter based on keyword lists would be tempted to guess here. I
    # want it to defer instead - a wrong prefilter decision skips the LLM
    # entirely, so false confidence here is worse than just calling the LLM
    # more often than strictly necessary.
    assert prefilter("Is this page complicated?") is None


def test_prefilter_catches_greetings_and_meta_questions():
    for question in ["hello", "what can you do?", "who are you?"]:
        decision = prefilter(question)
        assert decision is not None, f"expected prefilter to catch: {question!r}"
        assert decision.needs_detection is False


class _RaisingClient:
    """A stub Groq client that always fails, to test the fail-closed path."""

    class chat:
        class completions:
            @staticmethod
            def create(**kwargs):
                raise RuntimeError("simulated network failure")


def test_route_fails_closed_to_out_of_scope_when_the_llm_call_errors():
    """
    If Groq is down, times out, or the key is wrong, I would rather the
    router say "I can't determine this" than guess that detection is or
    isn't needed. Wrongly skipping a real question is worse than wrongly
    declining an answerable one - the second at least tells the user
    something true about what happened.
    """
    from app.reasoning.router import route

    decision = route("How many tables are on this page?", client=_RaisingClient())

    assert decision.needs_detection is False
    assert decision.task_type == "out_of_scope"
    assert "simulated network failure" in decision.reason
