"""The policy layer is the reason this system is safe to point at a real inbox,
so it is tested as a truth table rather than through the pipeline."""

from datetime import datetime

import pytest

from cicero import config
from cicero.models import (Action, Classification, Contact, Email, Intent,
                           SenderType)
from cicero.policy import RunState, decide, scheduling_warranted


@pytest.fixture
def policy():
    return config.policy()


def email(body="ok"):
    return Email(message_id="<m>", thread_id="t", from_name="A",
                 from_email="a@b.com", to=[], subject="Re: x", body=body,
                 raw_body=body, received_at=datetime.now())


def contact(**kw):
    base = dict(email="a@b.com", name="A", role=SenderType.FOUNDER,
                company="Co")
    base.update(kw)
    return Contact(**base)


def cls(intent=Intent.INTERESTED, conf=0.95, flags=None, wants_call=False,
        slot=None, role=SenderType.FOUNDER):
    return Classification(sender_type=role, sender_type_source="crm",
                          intent=intent, confidence=conf, summary="s",
                          reasoning="r", red_flags=flags or [],
                          wants_call=wants_call, accepted_proposed_slot=slot)


def d(policy, *, c=None, ct=None, run=None, **kw):
    return decide(email(), ct if ct is not None else contact(),
                  c if c is not None else cls(), policy,
                  run or RunState(), **kw)


# -- things that must never be automated -----------------------------------

@pytest.mark.parametrize("intent", [Intent.NEGOTIATION, Intent.HOSTILE,
                                    Intent.REFERRAL, Intent.UNCLEAR])
def test_hard_intents_always_reach_a_human(policy, intent):
    dec = d(policy, c=cls(intent=intent, conf=0.99))
    assert dec.action is Action.ESCALATE


def test_hard_intents_produce_no_draft_to_rubber_stamp(policy):
    # ESCALATE is the signal to pipeline.py that no draft may be generated.
    assert d(policy, c=cls(intent=Intent.NEGOTIATION)).action is Action.ESCALATE


@pytest.mark.parametrize("flag", [
    "mentions_price_or_valuation", "mentions_legal_or_litigation",
    "mentions_nda_or_contract_terms", "sender_is_upset"])
def test_red_flags_override_a_friendly_intent(policy, flag):
    dec = d(policy, c=cls(intent=Intent.INTERESTED, conf=0.99, flags=[flag]))
    assert dec.action is Action.ESCALATE
    assert flag in dec.reason


def test_static_flag_alone_is_enough_to_escalate(policy):
    dec = d(policy, static_flags=["mentions_price_or_valuation"])
    assert dec.action is Action.ESCALATE


def test_opted_out_contact_is_never_replied_to(policy):
    dec = d(policy, ct=contact(opted_out=True),
            c=cls(intent=Intent.SCHEDULE_REQUEST, wants_call=True))
    assert dec.action is Action.SUPPRESS


def test_opt_out_with_legal_threat_reaches_a_person(policy):
    dec = d(policy, prefiltered=(Intent.OPT_OUT, "explicit opt-out"),
            static_flags=["mentions_legal_or_litigation"])
    assert dec.action is Action.ESCALATE


def test_plain_opt_out_is_suppressed_silently(policy):
    dec = d(policy, prefiltered=(Intent.OPT_OUT, "explicit opt-out"))
    assert dec.action is Action.SUPPRESS


def test_classifier_failure_escalates_rather_than_guesses(policy):
    dec = decide(email(), contact(), None, policy, RunState(),
                 classifier_error="timeout")
    assert dec.action is Action.ESCALATE
    assert "timeout" in dec.reason


# -- confidence gates -------------------------------------------------------

def test_below_draft_floor_gets_no_draft(policy):
    assert d(policy, c=cls(conf=0.40)).action is Action.ESCALATE


def test_between_the_floors_gets_a_draft_but_no_send(policy):
    assert d(policy, c=cls(conf=0.70)).action is Action.DRAFT_FOR_REVIEW


def test_above_the_floor_may_send(policy):
    assert d(policy, c=cls(conf=0.95)).action is Action.AUTO_SEND


# -- relationship gates -----------------------------------------------------

def test_unknown_sender_is_held(policy):
    dec = decide(email(), None, cls(conf=0.99), policy, RunState())
    assert dec.action is Action.DRAFT_FOR_REVIEW
    assert dec.rule_id == "r060_unknown_sender"


def test_machine_does_not_talk_over_a_colleague(policy):
    dec = d(policy, ct=contact(human_has_touched_thread=True),
            c=cls(intent=Intent.SCHEDULE_REQUEST, conf=0.99, wants_call=True))
    assert dec.action is Action.DRAFT_FOR_REVIEW
    assert dec.rule_id == "r070_human_in_thread"


def test_consecutive_auto_reply_cap(policy):
    assert d(policy, ct=contact(auto_replies_sent=2)).action \
        is Action.DRAFT_FOR_REVIEW


def test_run_budget_is_a_circuit_breaker(policy):
    run = RunState(auto_sends=policy["rate_limits"]["max_auto_sends_per_run"])
    assert d(policy, run=run).action is Action.DRAFT_FOR_REVIEW


# -- scheduling consent -----------------------------------------------------

def test_enthusiasm_is_not_consent_to_a_calendar_invite():
    assert not scheduling_warranted(cls(intent=Intent.INTERESTED, conf=0.99))


def test_asking_for_a_call_is_consent():
    assert scheduling_warranted(
        cls(intent=Intent.SCHEDULE_REQUEST, wants_call=True))


def test_schedule_request_without_wanting_a_call_is_not_consent():
    # e.g. "someone asked me to set up a call with you" -- reported, not asked.
    assert not scheduling_warranted(
        cls(intent=Intent.SCHEDULE_REQUEST, wants_call=False))


def test_accepting_a_slot_we_offered_is_consent():
    assert scheduling_warranted(cls(intent=Intent.INTERESTED, slot=2))


def test_interested_reply_gets_a_reply_not_an_invite(policy):
    assert d(policy, c=cls(intent=Intent.INTERESTED, conf=0.95)).action \
        is Action.AUTO_SEND


def test_call_request_routes_to_scheduling(policy):
    dec = d(policy, c=cls(intent=Intent.SCHEDULE_REQUEST, conf=0.95,
                          wants_call=True))
    assert dec.action is Action.SCHEDULE_AND_SEND


# -- ordering ---------------------------------------------------------------

def test_suppression_beats_everything(policy):
    """An opted-out contact who asks for a call still gets nothing."""
    dec = d(policy, ct=contact(opted_out=True),
            c=cls(intent=Intent.SCHEDULE_REQUEST, conf=1.0, wants_call=True))
    assert dec.rule_id == "r000_opted_out"


def test_red_flag_beats_confidence(policy):
    dec = d(policy, c=cls(conf=1.0, flags=["mentions_price_or_valuation"]))
    assert dec.rule_id == "r040_red_flag"
