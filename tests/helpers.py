"""Shared doubles for the pipeline tests.

`ScriptedLLM` answers by schema rather than by call order, so a test can queue
one classification and not care how many other model calls the pipeline makes
around it.
"""

from __future__ import annotations

from typing import Any, Optional, Type

from cicero.classify import ClassificationOut
from cicero.llm import LLMBackend, LLMError
from cicero.scheduling import RequestedWindows, TimeWindow
from cicero.verify import Rubric

PASSING_RUBRIC = Rubric(factual_grounding=5, tone_fit=5, no_overcommitment=5,
                        answers_the_message=5, scheduling_accuracy=5,
                        would_send_as_is=True, issues=[])

FAILING_RUBRIC = Rubric(factual_grounding=1, tone_fit=2, no_overcommitment=1,
                        answers_the_message=2, scheduling_accuracy=1,
                        would_send_as_is=False, issues=["invents a price"])

# The body used by the pipeline tests. Long enough to clear the classifier's
# "too short to be confident" penalty, and quoted verbatim as the evidence span
# so the grounding check passes -- both are real gates, so a test fixture has to
# satisfy them the same way a real message would.
DEFAULT_BODY = ("Sounds good, happy to keep talking about this whenever it "
                "suits you.")

# Long enough to clear the verifier's minimum word count, bland enough to trip
# none of its content rules.
CLEAN_DRAFT = ("Thanks for coming back to me on this, and for being straight "
               "about where things stand. That makes sense and there is no "
               "pressure at all from our side.\n\nBest,\nAkshat")


def classification_out(**kw: Any) -> ClassificationOut:
    base = dict(
        sender_type="founder", sender_type_evidence=DEFAULT_BODY,
        intent="interested", intent_evidence=DEFAULT_BODY,
        confidence=0.95, summary="scripted",
        reasoning="scripted", wants_call=False, proposed_times_text="",
        accepted_proposed_slot=None, questions_asked=[], red_flags=[])
    base.update(kw)
    return ClassificationOut(**base)


class ScriptedLLM(LLMBackend):
    """Answers by requested schema. Anything not supplied raises LLMError,
    which is what the pipeline sees when a model is unavailable."""

    def __init__(self, *, classification: Optional[ClassificationOut] = None,
                 windows: Optional[RequestedWindows] = None,
                 rubric: Optional[Rubric] = None,
                 draft: Optional[str] = CLEAN_DRAFT):
        self.classification = classification
        self.windows = windows
        self.rubric = rubric if rubric is not None else PASSING_RUBRIC
        self.draft = draft
        self.input_tokens = self.output_tokens = 0
        self.structured_calls: list[str] = []
        self.text_calls: int = 0

    def structured(self, *, model, system, user, schema: Type[Any],
                   max_tokens: int = 4000):
        self.structured_calls.append(schema.__name__)
        supplied = {ClassificationOut: self.classification,
                    RequestedWindows: self.windows,
                    Rubric: self.rubric}.get(schema, None)
        if supplied is None:
            raise LLMError(f"ScriptedLLM: no {schema.__name__} queued")
        return supplied

    def text(self, *, model, system, user, max_tokens=2000, effort="medium"):
        self.text_calls += 1
        if self.draft is None:
            raise LLMError("ScriptedLLM: no draft queued")
        return self.draft


def window(start: str, end: str) -> RequestedWindows:
    return RequestedWindows(resolvable=True, note="scripted",
                            windows=[TimeWindow(start=start, end=end)])
