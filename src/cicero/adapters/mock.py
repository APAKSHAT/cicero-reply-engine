"""In-memory Gmail/Calendar stand-ins backed by the JSON fixtures.

The mock calendar is pre-populated with a realistic-looking week so the slot
picker has something to avoid -- an empty calendar would let a broken
availability check pass.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from ..models import Meeting
from .base import CalendarAdapter, MailAdapter


class MockGmail(MailAdapter):
    def __init__(self, inbox_path: Path):
        self.inbox_path = inbox_path
        self.sent: list[dict] = []
        self.drafts: list[dict] = []

    def fetch_replies(self) -> list[dict]:
        with open(self.inbox_path) as f:
            return json.load(f)

    def send_reply(self, *, thread_id, in_reply_to, to, subject, body) -> str:
        mid = f"<sent-{uuid.uuid4().hex[:10]}@cicerocapital.example>"
        self.sent.append({"message_id": mid, "thread_id": thread_id,
                          "in_reply_to": in_reply_to, "to": to,
                          "subject": subject, "body": body})
        return mid

    def create_draft(self, *, thread_id, in_reply_to, to, subject, body) -> str:
        did = f"draft-{uuid.uuid4().hex[:10]}"
        self.drafts.append({"draft_id": did, "thread_id": thread_id,
                            "in_reply_to": in_reply_to, "to": to,
                            "subject": subject, "body": body})
        return did


class MockCalendar(CalendarAdapter):
    """Organizer calendar with a plausible existing load."""

    def __init__(self, now: datetime, timezone: str = "America/New_York"):
        self.tz = ZoneInfo(timezone)
        self.events: list[Meeting] = []
        self._by_key: dict[str, Meeting] = {}
        self._seed(now)

    def _seed(self, now: datetime) -> None:
        base = now.astimezone(self.tz).replace(hour=0, minute=0, second=0,
                                               microsecond=0)
        # (days from now, start hour, minute, duration minutes, title)
        load = [
            (2, 9, 0, 60, "Portfolio review"),
            (2, 13, 30, 30, "Call: Sunbelt broker intro"),
            (3, 11, 0, 30, "Call: Ohio industrial services"),
            (3, 15, 0, 90, "Diligence: Foster Plumbing"),
            (4, 9, 30, 30, "Weekly pipeline"),
            (4, 14, 0, 60, "Lender call"),
            (5, 10, 0, 30, "Call: seller follow-up"),
            (8, 9, 0, 30, "Call: Keystone"),
            (9, 11, 0, 60, "Site visit prep"),
        ]
        for days, h, m, dur, title in load:
            start = base + timedelta(days=days, hours=h, minutes=m)
            self.events.append(Meeting(
                event_id=f"seed-{len(self.events)}", start=start,
                end=start + timedelta(minutes=dur), timezone=str(self.tz),
                attendees=[], join_url="", title=title))

    def busy(self, start, end):
        return sorted((e.start, e.end) for e in self.events
                      if e.end > start and e.start < end)

    def create_event(self, *, title, start, end, timezone, attendees,
                     description, idempotency_key) -> Meeting:
        if idempotency_key in self._by_key:
            return self._by_key[idempotency_key]     # replay-safe
        eid = f"evt-{uuid.uuid4().hex[:12]}"
        meeting = Meeting(
            event_id=eid, start=start, end=end, timezone=timezone,
            attendees=attendees,
            join_url=f"https://meet.example.com/cicero-{eid[-8:]}",
            title=title)
        self.events.append(meeting)
        self._by_key[idempotency_key] = meeting
        return meeting
