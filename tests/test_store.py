"""The ledger. Two jobs: make replays impossible, and make every decision
explainable after the fact."""

from datetime import datetime, timezone

import pytest

from cicero.models import (Action, Classification, Contact, Decision, Draft,
                           Email, Intent, Outcome, SenderType, Verdict,
                           VerdictIssue)
from cicero.store import Store


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "ledger.db")


def outcome(message_id="<m1@x>", action=Action.AUTO_SEND, sent=False):
    email = Email(message_id=message_id, thread_id="t-1", from_name="Dana",
                  from_email="dana@x.com", to=["r@c.com"], subject="Re: x",
                  body="hello", raw_body="hello",
                  received_at=datetime(2026, 8, 28, tzinfo=timezone.utc))
    return Outcome(
        email=email,
        contact=Contact(email="dana@x.com", name="Dana",
                        role=SenderType.FOUNDER, company="Co"),
        classification=Classification(
            sender_type=SenderType.FOUNDER, sender_type_source="crm",
            intent=Intent.INTERESTED, confidence=0.91, summary="s",
            reasoning="r", red_flags=["identity_uncertain"]),
        decision=Decision(action, "because", "r210_auto_send"),
        draft=Draft(subject="Re: x", body="body text", model="test"),
        verdict=Verdict(passed=True,
                        issues=[VerdictIssue("warn", "critic_note", "minor")]),
        sent=sent)


# -- idempotency ------------------------------------------------------------

def test_an_unseen_message_is_not_marked_processed(store):
    assert store.already_processed("<never@seen>") is False


def test_recording_marks_a_message_processed(store):
    store.record(outcome())
    assert store.already_processed("<m1@x>") is True


def test_recording_twice_keeps_one_row(store):
    store.record(outcome())
    store.record(outcome())
    assert len(store.all_outcomes()) == 1


# -- what a reviewer needs to see ------------------------------------------

def test_the_row_explains_the_decision(store):
    store.record(outcome())
    row = store.all_outcomes()[0]
    assert row["rule_id"] == "r210_auto_send"
    assert row["reason"] == "because"
    assert row["intent"] == "interested"
    assert row["confidence"] == pytest.approx(0.91)
    assert "identity_uncertain" in row["red_flags"]
    assert row["draft_body"] == "body text"


def test_the_full_payload_round_trips(store):
    import json
    store.record(outcome())
    payload = json.loads(store.all_outcomes()[0]["payload"])
    assert payload["email"]["from_email"] == "dana@x.com"
    assert payload["classification"]["intent"] == "interested"


def test_only_pending_items_appear_in_the_review_queue(store):
    store.record(outcome("<a@x>"), review_status="pending")
    store.record(outcome("<b@x>"), review_status="sent")
    store.record(outcome("<c@x>"), review_status="n/a")
    assert [r["message_id"] for r in store.pending_review()] == ["<a@x>"]


def test_approving_clears_it_from_the_queue(store):
    store.record(outcome("<a@x>"), review_status="pending")
    store.set_review_status("<a@x>", "approved", sent=True)
    assert store.pending_review() == []
    assert store.all_outcomes()[0]["sent"] == 1


# -- contact state ----------------------------------------------------------

def test_an_unknown_contact_has_no_state(store):
    assert store.contact_state("nobody@x.com") == {}


def test_contact_state_is_created_then_merged(store):
    store.update_contact("dana@x.com", auto_replies_sent=1)
    store.update_contact("dana@x.com", opted_out=True)
    state = store.contact_state("dana@x.com")
    assert state["auto_replies_sent"] == 1, "the earlier field must survive"
    assert state["opted_out"] is True


def test_proposed_slots_survive_a_round_trip(store):
    slots = ["2026-09-01T14:00:00-04:00", "2026-09-02T11:00:00-04:00"]
    store.update_contact("dana@x.com", proposed_slots=slots)
    assert store.contact_state("dana@x.com")["proposed_slots"] == slots


def test_state_persists_across_reopening_the_database(tmp_path):
    path = tmp_path / "ledger.db"
    Store(path).update_contact("dana@x.com", opted_out=True)
    assert Store(path).contact_state("dana@x.com")["opted_out"] is True
