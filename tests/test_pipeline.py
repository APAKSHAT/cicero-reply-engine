"""End-to-end tests over the real pipeline with a scripted model.

These are the tests that prove the wiring: that a decision actually reaches the
mail adapter, that a suppressed message never does, that booking happens after
verification and not before, and that re-running the same inbox sends nothing a
second time.
"""

import json
from datetime import datetime, timezone

import pytest

from cicero import config
from cicero.adapters.mock import MockCalendar, MockGmail
from cicero.models import Action
from cicero.pipeline import load_contacts, process_one, run_pipeline
from cicero.policy import RunState
from cicero.store import Store
from helpers import (CLEAN_DRAFT, DEFAULT_BODY, FAILING_RUBRIC, ScriptedLLM,
                     classification_out, window)

NOW = datetime(2026, 8, 31, 13, 0, tzinfo=timezone.utc)   # Monday


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "test.db")


@pytest.fixture
def calendar():
    return MockCalendar(NOW)


@pytest.fixture
def mail(tmp_path):
    inbox = tmp_path / "inbox.json"
    inbox.write_text("[]")
    return MockGmail(inbox)


@pytest.fixture
def policy():
    return {**config.policy(), "dry_run": False}


def raw(body=DEFAULT_BODY,
        frm="dana@harborlinemech.com", mid="<m1@x>", tid="t-1",
        subject="Re: Harborline"):
    return {"message_id": mid, "thread_id": tid, "from_name": "Dana Whitfield",
            "from_email": frm, "to": ["replies@cicero.example"],
            "subject": subject, "received_at": "2026-08-28T14:12:00-05:00",
            "body": body}


def run_one(*, store, mail, calendar, policy, llm, message=None, contacts=None):
    return process_one(
        raw=message or raw(), contacts=contacts if contacts is not None
        else load_contacts(config.DATA_DIR / "contacts.json", store),
        store=store, llm=llm, mail=mail, calendar=calendar,
        brand=config.brand(), policy=policy, run=RunState(), now=NOW)


# -- the happy path ---------------------------------------------------------

def test_a_clean_reply_is_drafted_verified_and_sent(store, mail, calendar, policy):
    llm = ScriptedLLM(classification=classification_out())
    out = run_one(store=store, mail=mail, calendar=calendar, policy=policy,
                  llm=llm)

    assert out.decision.action is Action.AUTO_SEND
    assert out.sent is True
    assert len(mail.sent) == 1
    assert mail.sent[0]["to"] == "dana@harborlinemech.com"
    assert out.verdict.passed


def test_the_sent_body_is_the_verified_draft(store, mail, calendar, policy):
    """Whatever the verifier approved is byte-for-byte what goes out."""
    llm = ScriptedLLM(classification=classification_out())
    out = run_one(store=store, mail=mail, calendar=calendar, policy=policy,
                  llm=llm)
    assert mail.sent[0]["body"] == out.draft.body == CLEAN_DRAFT


def test_reply_is_threaded_to_the_message_it_answers(store, mail, calendar,
                                                     policy):
    llm = ScriptedLLM(classification=classification_out())
    run_one(store=store, mail=mail, calendar=calendar, policy=policy, llm=llm)
    assert mail.sent[0]["in_reply_to"] == "<m1@x>"
    assert mail.sent[0]["thread_id"] == "t-1"


# -- nothing is sent when it should not be ----------------------------------

def test_suppressed_mail_never_reaches_the_adapter(store, mail, calendar,
                                                   policy):
    bounce = raw(body="Address not found. 550 5.1.1 no such user",
                 frm="mailer-daemon@googlemail.com",
                 subject="Delivery Status Notification (Failure)")
    llm = ScriptedLLM()
    out = run_one(store=store, mail=mail, calendar=calendar, policy=policy,
                  llm=llm, message=bounce)

    assert out.decision.action is Action.SUPPRESS
    assert mail.sent == [] and mail.drafts == []
    assert llm.structured_calls == [], "a pre-filtered message must cost no tokens"


def test_escalated_mail_produces_no_draft_to_rubber_stamp(store, mail, calendar,
                                                          policy):
    llm = ScriptedLLM(classification=classification_out(
        intent="negotiation", confidence=0.99))
    out = run_one(store=store, mail=mail, calendar=calendar, policy=policy,
                  llm=llm)

    assert out.decision.action is Action.ESCALATE
    assert out.draft is None
    assert mail.sent == [] and mail.drafts == []
    assert llm.text_calls == 0, "no drafting should be attempted"


def test_a_failed_verdict_holds_the_draft_instead_of_sending(store, mail,
                                                             calendar, policy):
    llm = ScriptedLLM(classification=classification_out(),
                      rubric=FAILING_RUBRIC)
    out = run_one(store=store, mail=mail, calendar=calendar, policy=policy,
                  llm=llm)

    assert out.decision.rule_id == "r400_failed_verification"
    assert out.sent is False
    assert mail.sent == []
    assert len(mail.drafts) == 1, "it should be waiting in the review queue"


def test_a_classifier_failure_escalates_rather_than_guessing(store, mail,
                                                             calendar, policy):
    llm = ScriptedLLM(classification=None)          # raises LLMError
    out = run_one(store=store, mail=mail, calendar=calendar, policy=policy,
                  llm=llm)

    assert out.decision.rule_id == "r010_classifier_failed"
    assert mail.sent == []


def test_a_generation_failure_escalates(store, mail, calendar, policy):
    llm = ScriptedLLM(classification=classification_out(), draft=None)
    out = run_one(store=store, mail=mail, calendar=calendar, policy=policy,
                  llm=llm)

    assert out.decision.rule_id == "r300_generation_failed"
    assert mail.sent == []


def test_dry_run_does_all_the_work_and_sends_nothing(store, mail, calendar):
    dry = {**config.policy(), "dry_run": True}
    llm = ScriptedLLM(classification=classification_out())
    out = run_one(store=store, mail=mail, calendar=calendar, policy=dry,
                  llm=llm)

    assert out.draft is not None and out.verdict.passed
    assert out.sent is False and mail.sent == []
    assert "DRY RUN" in out.decision.reason


# -- scheduling -------------------------------------------------------------

def test_a_call_request_with_stated_times_books_and_sends(store, mail, calendar,
                                                          policy):
    llm = ScriptedLLM(
        classification=classification_out(
            intent="schedule_request", wants_call=True,
            proposed_times_text="Tuesday afternoon after 1pm central"),
        windows=window("2026-09-01T13:00:00-05:00", "2026-09-01T16:00:00-05:00"))
    out = run_one(store=store, mail=mail, calendar=calendar, policy=policy,
                  llm=llm)

    assert out.decision.action is Action.SCHEDULE_AND_SEND
    assert out.meeting is not None
    assert out.sent is True
    assert out.meeting in calendar.events


def test_a_call_request_without_times_proposes_and_books_nothing(
        store, mail, calendar, policy):
    before = len(calendar.events)
    llm = ScriptedLLM(classification=classification_out(
        intent="schedule_request", wants_call=True, proposed_times_text=""))
    out = run_one(store=store, mail=mail, calendar=calendar, policy=policy,
                  llm=llm)

    assert out.sent is True
    assert out.meeting is None, "consent to a call is not consent to an hour"
    assert len(calendar.events) == before


def test_proposed_slots_are_remembered_so_an_acceptance_can_be_matched(
        store, mail, calendar, policy):
    llm = ScriptedLLM(classification=classification_out(
        intent="schedule_request", wants_call=True, proposed_times_text=""))
    run_one(store=store, mail=mail, calendar=calendar, policy=policy, llm=llm)

    state = store.contact_state("dana@harborlinemech.com")
    assert len(state["proposed_slots"]) == policy["scheduling"]["slots_to_propose"]


def test_the_calendar_is_written_only_after_verification_passes(
        store, mail, calendar, policy):
    """A draft that fails its checks must not leave an orphan invite."""
    before = len(calendar.events)
    llm = ScriptedLLM(
        classification=classification_out(
            intent="schedule_request", wants_call=True,
            proposed_times_text="Tuesday afternoon"),
        windows=window("2026-09-01T13:00:00-05:00", "2026-09-01T16:00:00-05:00"),
        rubric=FAILING_RUBRIC)
    out = run_one(store=store, mail=mail, calendar=calendar, policy=policy,
                  llm=llm)

    assert out.decision.rule_id == "r400_failed_verification"
    assert out.meeting is None
    assert len(calendar.events) == before, "no invite may exist for a held draft"
    assert mail.sent == []


def test_a_calendar_failure_holds_the_email(store, mail, policy):
    """We never promise a meeting that does not exist."""
    class BrokenCalendar(MockCalendar):
        def create_event(self, **kw):
            raise RuntimeError("calendar API down")

    llm = ScriptedLLM(
        classification=classification_out(
            intent="schedule_request", wants_call=True,
            proposed_times_text="Tuesday afternoon"),
        windows=window("2026-09-01T13:00:00-05:00", "2026-09-01T16:00:00-05:00"))
    out = run_one(store=store, mail=mail, calendar=BrokenCalendar(NOW),
                  policy=policy, llm=llm)

    assert out.decision.rule_id == "r500_calendar_failed"
    assert out.sent is False and mail.sent == []


# -- state and replay -------------------------------------------------------

def test_an_opt_out_is_recorded_against_the_contact(store, mail, calendar,
                                                    policy):
    msg = raw(body="Please take me off your list.")
    run_one(store=store, mail=mail, calendar=calendar, policy=policy,
            llm=ScriptedLLM(), message=msg)

    assert store.contact_state("dana@harborlinemech.com")["opted_out"] is True


def test_a_sent_reply_increments_the_auto_reply_counter(store, mail, calendar,
                                                        policy):
    run_one(store=store, mail=mail, calendar=calendar, policy=policy,
            llm=ScriptedLLM(classification=classification_out()))
    assert store.contact_state("dana@harborlinemech.com")["auto_replies_sent"] == 1


def test_the_same_message_is_never_processed_twice(store, mail, calendar,
                                                   policy):
    contacts = load_contacts(config.DATA_DIR / "contacts.json", store)
    first = run_one(store=store, mail=mail, calendar=calendar, policy=policy,
                    llm=ScriptedLLM(classification=classification_out()),
                    contacts=contacts)
    second = run_one(store=store, mail=mail, calendar=calendar, policy=policy,
                     llm=ScriptedLLM(classification=classification_out()),
                     contacts=contacts)

    assert first.sent is True
    assert second is None, "a replay must be a no-op"
    assert len(mail.sent) == 1


def test_a_whole_run_is_replay_safe(store, tmp_path, calendar, policy):
    """Re-running the real fixture inbox must not send a second time."""
    inbox = MockGmail(config.DATA_DIR / "inbox.json")
    kwargs = dict(mail=inbox, calendar=calendar, store=store,
                  llm=ScriptedLLM(classification=classification_out()),
                  contacts_path=config.DATA_DIR / "contacts.json",
                  brand=config.brand(), policy=policy, now=NOW)

    expected = len(json.loads((config.DATA_DIR / "inbox.json").read_text()))
    first = run_pipeline(**kwargs)
    sent_after_first = len(inbox.sent)
    second = run_pipeline(**kwargs)

    assert len(first) == expected
    assert second == []
    assert len(inbox.sent) == sent_after_first


def test_one_bad_message_does_not_stop_the_run(store, tmp_path, calendar,
                                               policy):
    class ExplodingGmail(MockGmail):
        def fetch_replies(self):
            return [raw(mid="<a@x>", tid="t-a"),
                    raw(mid="<b@x>", tid="t-b", frm="tom@prairielabelco.com")]

        def send_reply(self, **kw):
            if kw["in_reply_to"] == "<a@x>":
                raise RuntimeError("transient send failure")
            return super().send_reply(**kw)

    inbox = ExplodingGmail(tmp_path / "unused.json")
    outcomes = run_pipeline(
        mail=inbox, calendar=calendar, store=store,
        llm=ScriptedLLM(classification=classification_out()),
        contacts_path=config.DATA_DIR / "contacts.json",
        brand=config.brand(), policy=policy, now=NOW)

    assert len(outcomes) == 2
    assert outcomes[0].decision.rule_id == "r999_unhandled_error"
    assert outcomes[1].sent is True, "the second message still goes out"
