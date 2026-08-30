"""Stage 4 -- picking a time, and refusing to when we shouldn't.

Three separate questions, deliberately kept apart:

1. *May* we book? -- answered in `policy.py` (`scheduling_warranted`). Consent
   only; nothing to do with availability.
2. *When*, if the sender named a time? -- free text like "Tuesday or Wednesday
   afternoon after 1pm central" is turned into concrete windows by a small
   structured LLM call, and then those windows are **validated against our own
   rules** before anything is booked. The model interprets language; it does not
   get to authorize a calendar write.
3. *When*, if they didn't? -- we compute our own slots and propose them. We do
   not book. Someone who said "yes, send me some times" has consented to a
   conversation, not to a specific hour on their calendar.

Double-booking is prevented at the API boundary via a deterministic event id
derived from (thread, contact) -- see `adapters/google.py`. Retrying the whole
pipeline cannot produce a second invite.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone as _tz
from typing import Literal, Optional
from zoneinfo import ZoneInfo

from pydantic import BaseModel, Field

from .adapters.base import CalendarAdapter
from .config import MODEL_CLASSIFY
from .llm import LLMBackend, LLMError
from .models import Classification, Contact, Meeting


class TimeWindow(BaseModel):
    start: str = Field(description="ISO 8601 with offset, e.g. 2026-09-02T13:00:00-05:00")
    end: str = Field(description="ISO 8601 with offset. The latest the call could start.")


class RequestedWindows(BaseModel):
    resolvable: bool = Field(
        description="False if the availability text is too vague to turn into "
                    "concrete dates without guessing.")
    windows: list[TimeWindow] = Field(default_factory=list)
    note: str = Field(description="One line on how you read the text, or why not.")


_WINDOW_SYSTEM = """\
You convert a person's stated availability into concrete date-time windows.

- You are given the date the message was received and the sender's timezone. \
Resolve relative language ("next week", "Tuesday") against that date.
- Emit windows in the SENDER'S timezone with an explicit UTC offset.
- A window is a span in which a call could START, not the call itself.
- "afternoon" = 12:00-17:00. "morning" = 09:00-12:00. "after 1pm" = 13:00-17:00. \
"early next week" = Monday and Tuesday.
- If the text names no usable time, or you would have to guess the week, set \
resolvable=false and return no windows. Guessing here books a real meeting on a \
real person's calendar at the wrong time, so refuse rather than approximate.
- Never invent a window that the text does not support.
"""


def resolve_requested_windows(
    availability_text: str, received_at: datetime, contact_tz: str, llm: LLMBackend,
) -> tuple[list[tuple[datetime, datetime]], str]:
    """Language -> windows. Returns ([], reason) whenever it is not safe."""
    if not availability_text.strip():
        return [], "sender named no times"
    try:
        out = llm.structured(
            model=MODEL_CLASSIFY,
            system=_WINDOW_SYSTEM,
            user=(f"Message received: {received_at.isoformat()}\n"
                  f"Sender timezone: {contact_tz}\n\n"
                  f"Stated availability (verbatim):\n{availability_text}"),
            schema=RequestedWindows,
            max_tokens=4000,
        )
    except LLMError as e:
        return [], f"could not resolve stated times ({e})"

    if not out.resolvable or not out.windows:
        return [], out.note or "stated times were not resolvable"

    windows: list[tuple[datetime, datetime]] = []
    for w in out.windows:
        try:
            s, e = datetime.fromisoformat(w.start), datetime.fromisoformat(w.end)
        except ValueError:
            return [], f"model returned an unparseable datetime: {w.start!r}"
        if s.tzinfo is None or e.tzinfo is None:
            return [], "model returned a datetime without a timezone offset"
        if e <= s:
            return [], "model returned an inverted window"
        # Never trust a window that lands before the message was even sent.
        floor = received_at
        if floor.tzinfo is None:
            floor = floor.replace(tzinfo=_tz.utc)
        if s < floor - timedelta(days=1):
            return [], "model resolved a window in the past"
        windows.append((s, e))
    return windows, out.note


def _widen_to_whole_days(windows: list[tuple[datetime, datetime]],
                         contact_tz: str, sched: dict,
                         ) -> list[tuple[datetime, datetime]]:
    """Turn stated windows into whole working days on those same dates.

    The working-hours rules in `candidate_slots` still apply on top, so this
    widens the hour-of-day only -- never the set of days.
    """
    tz = ZoneInfo(contact_tz)
    lo, hi = sched["working_hours"]
    days = {w[0].astimezone(tz).date() for w in windows}
    out = []
    for day in sorted(days):
        midnight = datetime.combine(day, datetime.min.time(), tzinfo=tz)
        out.append((midnight + timedelta(hours=lo), midnight + timedelta(hours=hi)))
    return out


def _day_meeting_count(calendar: CalendarAdapter, day_start: datetime,
                       day_end: datetime) -> int:
    return len(calendar.busy(day_start, day_end))


def _is_free(calendar: CalendarAdapter, start: datetime, end: datetime,
             buffer_minutes: int) -> bool:
    pad = timedelta(minutes=buffer_minutes)
    for bstart, bend in calendar.busy(start - pad, end + pad):
        if bstart < end + pad and bend > start - pad:
            return False
    return True


def candidate_slots(
    *, now: datetime, contact_tz: str, calendar: CalendarAdapter, sched: dict,
    limit: int, within: Optional[list[tuple[datetime, datetime]]] = None,
) -> list[datetime]:
    """Every rule that makes a slot acceptable, applied in one place.

    A slot must simultaneously be: far enough out, on a working day, inside the
    *contact's* working hours in their timezone, inside the *organizer's*
    working hours in theirs, clear of existing events plus a buffer, and on a
    day that is not already at the meeting cap.
    """
    ctz = ZoneInfo(contact_tz)
    otz = ZoneInfo(sched["organizer_timezone"])
    dur = timedelta(minutes=sched["duration_minutes"])

    earliest = now + timedelta(hours=sched["min_notice_hours"])
    latest = now + timedelta(days=sched["max_days_out"])
    c_lo, c_hi = sched["working_hours"]
    o_lo, o_hi = sched["organizer_working_hours"]

    out: list[datetime] = []
    cursor = earliest.astimezone(ctz).replace(minute=0, second=0, microsecond=0)
    day_counts: dict[str, int] = {}

    while cursor < latest and len(out) < limit:
        local = cursor.astimezone(ctz)
        if local.weekday() not in sched["working_days"]:
            cursor = (local + timedelta(days=1)).replace(hour=0)
            continue

        day_key = local.date().isoformat()
        if day_key not in day_counts:
            day_start = local.replace(hour=0, minute=0)
            day_counts[day_key] = _day_meeting_count(
                calendar, day_start, day_start + timedelta(days=1))
        if day_counts[day_key] >= sched["max_meetings_per_day"]:
            cursor = (local + timedelta(days=1)).replace(hour=0, minute=0)
            continue

        # Both ends of the meeting must fall inside the working window, in
        # both timezones. Comparing hours rather than instants silently allows
        # a 30-minute call that starts at 17:45 and ends after the day does.
        def _within(instant, tz, lo, hi):
            start = instant.astimezone(tz)
            end = (instant + dur).astimezone(tz)
            midnight = start.replace(hour=0, minute=0, second=0, microsecond=0)
            return (start >= midnight + timedelta(hours=lo)
                    and end <= midnight + timedelta(hours=hi))

        in_contact_hours = _within(cursor, ctz, c_lo, c_hi)
        in_org_hours = _within(cursor, otz, o_lo, o_hi)
        in_window = True
        if within is not None:
            in_window = any(ws <= cursor <= we for ws, we in within)

        if in_contact_hours and in_org_hours and in_window and \
                cursor >= earliest and _is_free(calendar, cursor, cursor + dur,
                                                sched["buffer_minutes"]):
            out.append(cursor)
            day_counts[day_key] += 1
            if within is None:
                # Offering three times on one afternoon reads as though we have
                # nothing on. Spread proposals across days; only fall back to
                # same-day options when the caller pinned a window.
                cursor = (local + timedelta(days=1)).replace(
                    hour=0, minute=0, second=0, microsecond=0)
            else:
                cursor += timedelta(minutes=30)
        else:
            cursor += timedelta(minutes=30)

    return out


def resolve(
    *, classification: Classification, contact: Contact, received_at: datetime,
    now: datetime, calendar: CalendarAdapter, sched: dict, llm: LLMBackend,
) -> tuple[Literal["book", "propose", "none"], list[datetime], str]:
    """Decide between booking a specific time and proposing options.

    Returns (mode, slots, note). `book` carries exactly one slot.
    """
    # (a) They picked one of the numbered times we already offered.
    if classification.accepted_proposed_slot is not None:
        idx = classification.accepted_proposed_slot - 1
        if 0 <= idx < len(contact.proposed_slots):
            chosen = datetime.fromisoformat(contact.proposed_slots[idx])
            if chosen < now + timedelta(hours=1):
                return "propose", candidate_slots(
                    now=now, contact_tz=contact.timezone, calendar=calendar,
                    sched=sched, limit=sched["slots_to_propose"]), \
                    "accepted slot has already passed; re-proposing"
            if not _is_free(calendar, chosen,
                            chosen + timedelta(minutes=sched["duration_minutes"]),
                            sched["buffer_minutes"]):
                return "propose", candidate_slots(
                    now=now, contact_tz=contact.timezone, calendar=calendar,
                    sched=sched, limit=sched["slots_to_propose"]), \
                    "accepted slot is no longer free; re-proposing"
            return "book", [chosen], "sender accepted a slot we proposed"
        return "none", [], (f"sender referenced slot "
                            f"{classification.accepted_proposed_slot}, which we "
                            f"never proposed")

    # (b) They named their own availability.
    windows, note = resolve_requested_windows(
        classification.proposed_times_text, received_at, contact.timezone, llm)
    if windows:
        slots = candidate_slots(
            now=now, contact_tz=contact.timezone, calendar=calendar,
            sched=sched, limit=1, within=windows)
        if slots:
            return "book", slots[:1], f"booked inside stated availability ({note})"

        # Nothing free inside exactly what they asked for. Before falling back
        # to "any slot at all", try the same DAYS they named -- someone who says
        # "Tuesday or Wednesday afternoon, not mornings" is far better served by
        # a different hour on Tuesday than by Monday at 9am. A live run produced
        # exactly that: the sender excluded mornings and was offered 9:00 AM.
        same_days = _widen_to_whole_days(windows, contact.timezone, sched)
        slots = candidate_slots(
            now=now, contact_tz=contact.timezone, calendar=calendar,
            sched=sched, limit=sched["slots_to_propose"], within=same_days)
        if slots:
            return "propose", slots, (
                "no opening inside their stated hours; proposing other times on "
                "the same days they named")

        # Still nothing. We do NOT fall through to "any free slot": the sender
        # told us when they can talk, and proposing times that contradict it is
        # the single most irritating thing this system can do. A live run did
        # exactly that -- offered 9:00 AM to someone who wrote "not mornings" --
        # and the critic, which had blocked the identical mistake an hour
        # earlier, waved it through. An LLM judge is not a place to put a rule
        # you can state precisely, so this is now a hard stop and a person picks
        # up the thread.
        return "none", [], (
            "sender gave specific availability and we have nothing free that "
            "respects it; handing to a human rather than proposing times they "
            "already ruled out")

    # (c) They want a call but gave us nothing to work with -> propose, never book.
    slots = candidate_slots(now=now, contact_tz=contact.timezone,
                            calendar=calendar, sched=sched,
                            limit=sched["slots_to_propose"])
    if not slots:
        return "none", [], "no acceptable slots in the booking horizon"
    return "propose", slots, note or "sender consented to a call but named no time"


def book(
    *, calendar: CalendarAdapter, start: datetime, contact: Contact,
    thread_id: str, sched: dict, brand: dict, summary: str,
) -> Meeting:
    dur = timedelta(minutes=sched["duration_minutes"])
    firm = brand["sender"]["firm"]
    return calendar.create_event(
        title=f"{firm} <> {contact.company}",
        start=start, end=start + dur, timezone=contact.timezone,
        attendees=[contact.email, brand["sender"]["email"]],
        description=(f"Intro call.\n\nContext: {summary}\n\n"
                     f"Booked automatically from {contact.name}'s reply."),
        # One key per thread+contact: replays and retries cannot double-book.
        idempotency_key=f"{thread_id}:{contact.email}")
