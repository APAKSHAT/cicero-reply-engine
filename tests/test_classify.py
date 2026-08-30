"""Classification, minus the model.

The model's own output is scripted here on purpose. What is being tested is
everything we wrap around it: that the CRM outranks it on identity, that a
disagreement becomes a flag, and that its self-reported confidence is discounted
when the evidence does not hold up. Those are the parts that decide whether a
reply is allowed to send.
"""

from datetime import datetime, timezone

import pytest

from cicero.classify import build_user_prompt, classify
from cicero.llm import LLMError
from cicero.models import Contact, Email, SenderType
from cicero.redact import Redactor
from helpers import ScriptedLLM, classification_out

BODY = ("I am the owner here and I would be glad to talk this through with "
        "you sometime next month.")


def email(body=BODY, subject="Re: Harborline"):
    return Email(message_id="<m@x>", thread_id="t", from_name="Dana Whitfield",
                 from_email="dana@harborline.com", to=[], subject=subject,
                 body=body, raw_body=body,
                 received_at=datetime(2026, 8, 28, tzinfo=timezone.utc))


def contact(role=SenderType.FOUNDER):
    return Contact(email="dana@harborline.com", name="Dana Whitfield",
                   role=role, company="Harborline Mechanical")


def run(out, mail=None, ct=None):
    return classify(mail or email(), ct, ScriptedLLM(classification=out),
                    Redactor())


# -- identity ---------------------------------------------------------------

def test_the_crm_outranks_the_model_on_role():
    """We sent the original email, so we know who we wrote to."""
    result = run(classification_out(sender_type="broker", intent_evidence=BODY),
                 ct=contact(SenderType.FOUNDER))
    assert result.sender_type is SenderType.FOUNDER
    assert result.sender_type_source == "crm"


def test_a_role_disagreement_becomes_a_flag():
    """The founder-forwards-to-their-broker case must reach a human."""
    result = run(classification_out(sender_type="broker", intent_evidence=BODY),
                 ct=contact(SenderType.FOUNDER))
    assert "identity_uncertain" in result.red_flags


def test_agreement_raises_no_flag():
    result = run(classification_out(sender_type="founder", intent_evidence=BODY),
                 ct=contact(SenderType.FOUNDER))
    assert "identity_uncertain" not in result.red_flags


def test_an_unknown_sender_falls_back_to_the_model():
    result = run(classification_out(sender_type="broker", intent_evidence=BODY),
                 ct=None)
    assert result.sender_type is SenderType.BROKER
    assert result.sender_type_source == "llm"


def test_an_unknown_sender_is_not_flagged_identity_uncertain():
    """It has its own policy rule (r060) which holds a draft for review.
    Flagging it here as well would escalate with no draft and make that rule
    unreachable -- a bug found by running the pipeline."""
    result = run(classification_out(intent_evidence=BODY), ct=None)
    assert "identity_uncertain" not in result.red_flags


# -- confidence discounting -------------------------------------------------

def test_grounded_evidence_keeps_the_models_confidence():
    result = run(classification_out(confidence=0.95, intent_evidence=BODY,
                                    sender_type_evidence=BODY), ct=contact())
    assert result.confidence == pytest.approx(0.95)


def test_evidence_the_model_invented_costs_confidence():
    """A quote that is not in the message means the model is not reading it."""
    result = run(classification_out(
        confidence=0.95, intent_evidence="I would love to sell immediately",
        sender_type_evidence=BODY), ct=contact())
    assert result.confidence == pytest.approx(0.70)
    assert "not found verbatim" in result.reasoning


def test_a_very_short_message_costs_confidence():
    short = email(body="what is this")
    result = run(classification_out(confidence=0.95,
                                    intent_evidence="what is this",
                                    sender_type_evidence="what is this"),
                 mail=short, ct=contact())
    assert result.confidence == pytest.approx(0.75)


def test_an_unknown_sender_costs_confidence():
    result = run(classification_out(confidence=0.95, intent_evidence=BODY,
                                    sender_type_evidence=BODY), ct=None)
    assert result.confidence == pytest.approx(0.85)


def test_a_role_inferred_without_a_grounded_quote_costs_confidence():
    """Only applies when the role came from the model -- with a CRM record we
    are not relying on the quote in the first place."""
    result = run(classification_out(confidence=0.95, intent_evidence=BODY,
                                    sender_type_evidence="not in the message"),
                 ct=None)
    # 0.95 - 0.15 (ungrounded role quote) - 0.10 (unknown sender)
    assert result.confidence == pytest.approx(0.70)


def test_no_role_penalty_when_the_crm_supplied_the_role():
    result = run(classification_out(confidence=0.95, intent_evidence=BODY,
                                    sender_type_evidence="not in the message"),
                 ct=contact())
    assert result.confidence == pytest.approx(0.95)


def test_penalties_stack_and_are_explained():
    short = email(body="what is this")
    result = run(classification_out(confidence=0.95,
                                    intent_evidence="nothing like the message",
                                    sender_type_evidence="also not present"),
                 mail=short, ct=None)
    # 0.95 - 0.25 (ungrounded intent) - 0.15 (ungrounded role) - 0.20 (short)
    #      - 0.10 (unknown sender)
    assert result.confidence == pytest.approx(0.25)
    assert "confidence adjusted down" in result.reasoning


def test_confidence_never_goes_negative():
    short = email(body="hm")
    result = run(classification_out(confidence=0.05, intent_evidence="invented"),
                 mail=short, ct=None)
    assert result.confidence == 0.0


def test_evidence_matching_ignores_whitespace_and_case():
    result = run(classification_out(
        confidence=0.9, intent_evidence="I AM THE OWNER    HERE",
        sender_type_evidence=BODY), ct=contact())
    assert result.confidence == pytest.approx(0.9)


# -- output shape -----------------------------------------------------------

def test_unknown_red_flags_from_the_model_are_discarded():
    result = run(classification_out(
        intent_evidence=BODY, red_flags=["sender_is_upset", "made_up_flag"]),
        ct=contact())
    assert result.red_flags == ["sender_is_upset"]


def test_a_model_failure_propagates_rather_than_returning_a_guess():
    with pytest.raises(LLMError):
        classify(email(), contact(), ScriptedLLM(classification=None),
                 Redactor())


# -- what the model is shown ------------------------------------------------

def test_the_prompt_carries_the_crm_record():
    prompt = build_user_prompt(email(), contact(), Redactor())
    assert "Harborline Mechanical" in prompt
    assert "Role on file: founder" in prompt


def test_the_prompt_says_plainly_when_a_sender_is_unknown():
    prompt = build_user_prompt(email(), None, Redactor())
    assert "NOT in our CRM" in prompt


def test_the_senders_address_is_redacted_before_it_is_sent():
    prompt = build_user_prompt(email(), contact(), Redactor())
    assert "dana@harborline.com" not in prompt
    assert "<EMAIL_1>" in prompt


def test_previously_proposed_slots_are_numbered_for_the_model():
    ct = contact()
    ct.proposed_slots = ["2026-09-01T14:00:00-04:00", "2026-09-02T11:00:00-04:00"]
    prompt = build_user_prompt(email(), ct, Redactor())
    assert "1. 2026-09-01T14:00:00-04:00" in prompt
    assert "2. 2026-09-02T11:00:00-04:00" in prompt
