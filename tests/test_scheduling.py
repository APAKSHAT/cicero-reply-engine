"""Slot selection and, more importantly, the cases where we refuse to book."""

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from cicero import config, scheduling
from cicero.adapters.mock import MockCalendar
from cicero.llm import StubLLM
from cicero.models import Classification, Contact, Intent, SenderType
from cicero.scheduling import RequestedWindows, TimeWindow

ET = ZoneInfo("America/New_York")
NOW = datetime(2026, 8, 31, 9, 0, tzinfo=ET)      # a Monday


@pytest.fixture
def sched():
    return config.policy()["scheduling"]


@pytest.fixture
def calendar():
    return MockCalendar(NOW)


def contact(tz="America/New_York", slots=None):
    return Contact(email="a@b.com", name="A", role=SenderType.FOUNDER,
                   company="Co", timezone=tz, proposed_slots=slots or [])


def cls(**kw):
    base = dict(sender_type=SenderType.FOUNDER, sender_type_source="crm",
                intent=Intent.SCHEDULE_REQUEST, confidence=0.95, summary="s",
                reasoning="r", wants_call=True)
    base.update(kw)
    return Classification(**base)


# -- slot rules -------------------------------------------------------------

def test_respects_minimum_notice(calendar, sched):
    slots = scheduling.candidate_slots(now=NOW, contact_tz="America/New_York",
                                       calendar=calendar, sched=sched, limit=5)
    assert slots
    assert all(s >= NOW + timedelta(hours=sched["min_notice_hours"])
               for s in slots)


def test_never_proposes_a_weekend(calendar, sched):
    slots = scheduling.candidate_slots(now=NOW, contact_tz="America/New_York",
                                       calendar=calendar, sched=sched, limit=20)
    assert all(s.weekday() < 5 for s in slots)


def test_respects_the_contacts_local_working_hours(calendar, sched):
    """A 9am slot in New York is 6am in Los Angeles. Nobody wants that call."""
    slots = scheduling.candidate_slots(now=NOW, contact_tz="America/Los_Angeles",
                                       calendar=calendar, sched=sched, limit=10)
    lo, hi = sched["working_hours"]
    pt = ZoneInfo("America/Los_Angeles")
    assert slots
    for s in slots:
        local = s.astimezone(pt)
        assert lo <= local.hour < hi


def test_respects_the_organizers_working_hours_too(calendar, sched):
    slots = scheduling.candidate_slots(now=NOW, contact_tz="America/Los_Angeles",
                                       calendar=calendar, sched=sched, limit=10)
    olo, ohi = sched["organizer_working_hours"]
    for s in slots:
        assert olo <= s.astimezone(ET).hour < ohi


def test_avoids_existing_events_including_the_buffer(calendar, sched):
    slots = scheduling.candidate_slots(now=NOW, contact_tz="America/New_York",
                                       calendar=calendar, sched=sched, limit=20)
    pad = timedelta(minutes=sched["buffer_minutes"])
    dur = timedelta(minutes=sched["duration_minutes"])
    for s in slots:
        for b_start, b_end in calendar.busy(s - timedelta(days=1),
                                            s + timedelta(days=1)):
            assert not (b_start < s + dur + pad and b_end > s - pad)


def test_honours_the_daily_meeting_cap(calendar, sched):
    tight = {**sched, "max_meetings_per_day": 1}
    slots = scheduling.candidate_slots(now=NOW, contact_tz="America/New_York",
                                       calendar=calendar, sched=tight, limit=10)
    days = [s.date() for s in slots]
    assert len(days) == len(set(days))


# -- book vs propose --------------------------------------------------------

def test_no_stated_time_means_propose_not_book(calendar, sched):
    mode, slots, _ = scheduling.resolve(
        classification=cls(proposed_times_text=""), contact=contact(),
        received_at=NOW, now=NOW, calendar=calendar, sched=sched,
        llm=StubLLM())
    assert mode == "propose"
    assert len(slots) == sched["slots_to_propose"]


def test_unresolvable_availability_falls_back_to_proposing(calendar, sched):
    llm = StubLLM(structured_responses=[
        RequestedWindows(resolvable=False, windows=[],
                         note="'sometime soon' is not a date")])
    mode, slots, note = scheduling.resolve(
        classification=cls(proposed_times_text="sometime soon"),
        contact=contact(), received_at=NOW, now=NOW, calendar=calendar,
        sched=sched, llm=llm)
    assert mode == "propose"


def test_stated_availability_books_inside_the_window(calendar, sched):
    llm = StubLLM(structured_responses=[RequestedWindows(
        resolvable=True,
        windows=[TimeWindow(start="2026-09-02T13:00:00-04:00",
                            end="2026-09-02T16:00:00-04:00")],
        note="Wednesday afternoon")])
    mode, slots, _ = scheduling.resolve(
        classification=cls(proposed_times_text="Wednesday afternoon"),
        contact=contact(), received_at=NOW, now=NOW, calendar=calendar,
        sched=sched, llm=llm)
    assert mode == "book" and len(slots) == 1
    assert datetime(2026, 9, 2, 13, tzinfo=ET) <= slots[0] \
        <= datetime(2026, 9, 2, 16, tzinfo=ET)


def test_a_window_resolved_into_the_past_is_rejected(calendar, sched):
    llm = StubLLM(structured_responses=[RequestedWindows(
        resolvable=True,
        windows=[TimeWindow(start="2026-08-04T13:00:00-04:00",
                            end="2026-08-04T16:00:00-04:00")],
        note="wrong year/month")])
    mode, _, _ = scheduling.resolve(
        classification=cls(proposed_times_text="the 4th"), contact=contact(),
        received_at=NOW, now=NOW, calendar=calendar, sched=sched, llm=llm)
    assert mode == "propose"       # never books the model's bad date


def test_accepting_a_slot_we_offered_books_that_slot(calendar, sched):
    offered = (NOW + timedelta(days=2)).replace(hour=15, minute=0)
    mode, slots, _ = scheduling.resolve(
        classification=cls(accepted_proposed_slot=1),
        contact=contact(slots=[offered.isoformat()]), received_at=NOW, now=NOW,
        calendar=calendar, sched=sched, llm=StubLLM())
    assert mode == "book" and slots[0] == offered


def test_accepting_a_slot_that_has_since_filled_re_proposes(calendar, sched):
    taken = calendar.events[0].start
    mode, _, note = scheduling.resolve(
        classification=cls(accepted_proposed_slot=1),
        contact=contact(slots=[taken.isoformat()]), received_at=NOW, now=NOW,
        calendar=calendar, sched=sched, llm=StubLLM())
    assert mode == "propose" and "no longer free" in note


def test_referencing_a_slot_we_never_offered_does_nothing(calendar, sched):
    mode, slots, note = scheduling.resolve(
        classification=cls(accepted_proposed_slot=3), contact=contact(slots=[]),
        received_at=NOW, now=NOW, calendar=calendar, sched=sched, llm=StubLLM())
    assert mode == "none" and not slots


# -- double booking ---------------------------------------------------------

def test_booking_twice_with_the_same_key_creates_one_event(calendar, sched):
    brand = config.brand()
    start = NOW + timedelta(days=2, hours=8)
    a = scheduling.book(calendar=calendar, start=start, contact=contact(),
                        thread_id="t-1", sched=sched, brand=brand, summary="s")
    before = len(calendar.events)
    b = scheduling.book(calendar=calendar, start=start, contact=contact(),
                        thread_id="t-1", sched=sched, brand=brand, summary="s")
    assert a.event_id == b.event_id
    assert len(calendar.events) == before


def test_proposals_are_spread_across_different_days(calendar, sched):
    """Three options on one afternoon reads as an empty calendar. Found by
    inspecting a live run."""
    _, slots, _ = scheduling.resolve(
        classification=cls(proposed_times_text=""), contact=contact(),
        received_at=NOW, now=NOW, calendar=calendar, sched=sched,
        llm=StubLLM())
    days = [s.date() for s in slots]
    assert len(days) == len(set(days)), f"clustered on one day: {slots}"


def test_a_full_stated_window_falls_back_to_the_same_days_not_any_day(
        calendar, sched):
    """A live run offered 9:00 AM to a sender who wrote "not mornings". If we
    cannot fit their hours, stay on the days they named."""
    busy_day = (NOW + timedelta(days=2)).date()
    llm = StubLLM(structured_responses=[RequestedWindows(
        resolvable=True, note="Wednesday afternoon",
        windows=[TimeWindow(
            start=f"{busy_day}T13:00:00-04:00",
            end=f"{busy_day}T13:15:00-04:00")])])   # too narrow to fit a call

    mode, slots, note = scheduling.resolve(
        classification=cls(proposed_times_text="Wednesday afternoon"),
        contact=contact(), received_at=NOW, now=NOW, calendar=calendar,
        sched=sched, llm=llm)

    assert mode == "propose" and slots
    assert all(s.astimezone(ET).date() == busy_day for s in slots), \
        "proposals must stay on the day they asked for"
    assert "same days they named" in note


def test_we_never_propose_times_the_sender_already_ruled_out(calendar, sched):
    """The sender said afternoons. If we have no afternoon free on their days,
    a human takes it -- we do not offer mornings instead.

    A live run did offer mornings, and the critic passed it. This is the rule
    moved out of the model's judgement and into code."""
    day = (NOW + timedelta(days=2)).date()
    # A window with no room for a call, on a day whose whole span is also busy.
    llm = StubLLM(structured_responses=[RequestedWindows(
        resolvable=True, note="Wednesday afternoon",
        windows=[TimeWindow(start=f"{day}T13:00:00-04:00",
                            end=f"{day}T13:05:00-04:00")])])

    class FullCalendar(MockCalendar):
        def busy(self, start, end):
            return [(start, end)]              # every minute is taken

    mode, slots, note = scheduling.resolve(
        classification=cls(proposed_times_text="Wednesday afternoon"),
        contact=contact(), received_at=NOW, now=NOW, calendar=FullCalendar(NOW),
        sched=sched, llm=llm)

    assert mode == "none"
    assert slots == []
    assert "already ruled out" in note


def test_no_stated_availability_still_falls_back_to_proposing(calendar, sched):
    """The hard stop applies only when the sender named times. Someone who said
    'send me whatever works' still gets options."""
    mode, slots, _ = scheduling.resolve(
        classification=cls(proposed_times_text=""), contact=contact(),
        received_at=NOW, now=NOW, calendar=calendar, sched=sched,
        llm=StubLLM())
    assert mode == "propose" and slots
