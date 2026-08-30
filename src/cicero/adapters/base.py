"""Ports for the two external systems. The pipeline only ever sees these."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime

from ..models import Meeting


class MailAdapter(ABC):
    @abstractmethod
    def fetch_replies(self) -> list[dict]:
        """Raw inbound messages, newest last."""

    @abstractmethod
    def send_reply(self, *, thread_id: str, in_reply_to: str, to: str,
                   subject: str, body: str) -> str:
        """Send in-thread. Returns the new message id."""

    @abstractmethod
    def create_draft(self, *, thread_id: str, in_reply_to: str, to: str,
                     subject: str, body: str) -> str:
        """Save a draft without sending. Returns the draft id."""


class CalendarAdapter(ABC):
    @abstractmethod
    def busy(self, start: datetime, end: datetime) -> list[tuple[datetime, datetime]]:
        """Busy intervals for the organizer between start and end."""

    @abstractmethod
    def create_event(self, *, title: str, start: datetime, end: datetime,
                     timezone: str, attendees: list[str], description: str,
                     idempotency_key: str) -> Meeting:
        """Create an event. MUST be idempotent on idempotency_key."""
