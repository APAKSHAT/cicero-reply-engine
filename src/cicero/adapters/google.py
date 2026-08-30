"""Real Gmail + Google Calendar adapters.

Not exercised by `make demo` (which runs on the mocks), but this is the code the
prototype would run against a live inbox, and the shape of it is part of the
answer: the pipeline never learns which one it is talking to.

Auth: OAuth installed-app flow, token cached at `.google-token.json`. Scopes are
the narrowest that do the job:

* `gmail.modify` -- read, send, and label. NOT `mail.google.com`, which also
  grants permanent delete.
* `calendar.events` -- create and update events. NOT full `calendar`, which can
  delete entire calendars.
* `calendar.freebusy` -- read busy blocks only. This is deliberately separate:
  `calendar.events` does not cover `freebusy.query`, and the alternative that
  would (`calendar.readonly`) hands over the content of every event on every
  calendar. We only need to know *that* a slot is taken, never what it is.

    pip install google-api-python-client google-auth-oauthlib
"""

from __future__ import annotations

import base64
import hashlib
import re
from datetime import datetime
from email.message import EmailMessage
from pathlib import Path
from typing import Any

from ..models import Meeting
from .base import CalendarAdapter, MailAdapter

SCOPES = [
    "https://www.googleapis.com/auth/gmail.modify",
    "https://www.googleapis.com/auth/calendar.events",
    "https://www.googleapis.com/auth/calendar.freebusy",
]


def build_services(credentials_path: str = "credentials.json",
                   token_path: str = ".google-token.json") -> tuple[Any, Any]:
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build

    creds = None
    if Path(token_path).exists():
        creds = Credentials.from_authorized_user_file(token_path, SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                credentials_path, SCOPES)
            creds = flow.run_local_server(port=0)
        Path(token_path).write_text(creds.to_json())
    return (build("gmail", "v1", credentials=creds),
            build("calendar", "v3", credentials=creds))


def _header(payload: dict, name: str) -> str:
    for h in payload.get("headers", []):
        if h["name"].lower() == name.lower():
            return h["value"]
    return ""


def _plain_body(payload: dict) -> str:
    """Depth-first walk for the first text/plain part; falls back to stripped
    HTML, because plenty of founders reply from Outlook in HTML only."""
    if payload.get("mimeType") == "text/plain" and payload.get("body", {}).get("data"):
        return base64.urlsafe_b64decode(payload["body"]["data"]).decode(
            "utf-8", "replace")
    for part in payload.get("parts", []):
        found = _plain_body(part)
        if found:
            return found
    if payload.get("mimeType") == "text/html" and payload.get("body", {}).get("data"):
        html = base64.urlsafe_b64decode(payload["body"]["data"]).decode(
            "utf-8", "replace")
        return re.sub(r"<[^>]+>", "", re.sub(r"(?is)<(script|style).*?</\1>", "", html))
    return ""


class GmailAdapter(MailAdapter):
    def __init__(self, service, query: str = "in:inbox is:unread -from:me",
                 max_results: int = 50):
        self.svc = service
        self.query = query
        self.max_results = max_results

    def fetch_replies(self) -> list[dict]:
        listing = self.svc.users().messages().list(
            userId="me", q=self.query, maxResults=self.max_results).execute()
        out = []
        for ref in listing.get("messages", []):
            msg = self.svc.users().messages().get(
                userId="me", id=ref["id"], format="full").execute()
            p = msg["payload"]
            frm = _header(p, "From")
            m = re.match(r'^\s*"?([^"<]*)"?\s*<?([^>]+)>?\s*$', frm)
            name, addr = (m.group(1).strip(), m.group(2).strip()) if m else ("", frm)
            out.append({
                "message_id": _header(p, "Message-ID") or ref["id"],
                "gmail_id": ref["id"],
                "thread_id": msg["threadId"],
                "from_name": name,
                "from_email": addr,
                "to": [a.strip() for a in _header(p, "To").split(",") if a.strip()],
                "subject": _header(p, "Subject"),
                "received_at": datetime.fromtimestamp(
                    int(msg["internalDate"]) / 1000).astimezone().isoformat(),
                "headers": {h["name"].lower(): h["value"] for h in p.get("headers", [])},
                "body": _plain_body(p),
            })
        return out

    def _mime(self, *, in_reply_to: str, to: str, subject: str,
              body: str) -> dict:
        msg = EmailMessage()
        msg["To"] = to
        msg["Subject"] = subject if subject.lower().startswith("re:") else f"Re: {subject}"
        if in_reply_to:
            # Both headers: In-Reply-To threads it, References keeps the chain
            # intact for clients that ignore In-Reply-To.
            msg["In-Reply-To"] = in_reply_to
            msg["References"] = in_reply_to
        msg.set_content(body)
        return {"raw": base64.urlsafe_b64encode(msg.as_bytes()).decode()}

    def send_reply(self, *, thread_id, in_reply_to, to, subject, body) -> str:
        payload = self._mime(in_reply_to=in_reply_to, to=to, subject=subject,
                             body=body)
        payload["threadId"] = thread_id
        sent = self.svc.users().messages().send(userId="me", body=payload).execute()
        return sent["id"]

    def create_draft(self, *, thread_id, in_reply_to, to, subject, body) -> str:
        payload = self._mime(in_reply_to=in_reply_to, to=to, subject=subject,
                             body=body)
        payload["threadId"] = thread_id
        draft = self.svc.users().drafts().create(
            userId="me", body={"message": payload}).execute()
        return draft["id"]


class GoogleCalendarAdapter(CalendarAdapter):
    def __init__(self, service, calendar_id: str = "primary"):
        self.svc = service
        self.calendar_id = calendar_id

    def busy(self, start: datetime, end: datetime):
        resp = self.svc.freebusy().query(body={
            "timeMin": start.isoformat(),
            "timeMax": end.isoformat(),
            "items": [{"id": self.calendar_id}],
        }).execute()
        spans = resp["calendars"][self.calendar_id].get("busy", [])
        return sorted((datetime.fromisoformat(b["start"]),
                       datetime.fromisoformat(b["end"])) for b in spans)

    @staticmethod
    def _event_id(idempotency_key: str) -> str:
        """Deterministic, Google-legal event id (base32hex, lowercase).

        This is what makes booking safe to retry: a second insert with the same
        key returns 409 rather than creating a duplicate invite. Idempotency
        belongs at the API boundary, not in a 'did we already do this?' lookup
        that can race with itself.
        """
        digest = hashlib.sha256(idempotency_key.encode()).digest()
        return base64.b32hexencode(digest).decode().lower().rstrip("=")[:32]

    def create_event(self, *, title, start, end, timezone, attendees,
                     description, idempotency_key) -> Meeting:
        from googleapiclient.errors import HttpError

        eid = self._event_id(idempotency_key)
        body = {
            "id": eid,
            "summary": title,
            "description": description,
            "start": {"dateTime": start.isoformat(), "timeZone": timezone},
            "end": {"dateTime": end.isoformat(), "timeZone": timezone},
            "attendees": [{"email": a} for a in attendees],
            "conferenceData": {"createRequest": {
                "requestId": eid,
                "conferenceSolutionKey": {"type": "hangoutsMeet"}}},
            "reminders": {"useDefault": True},
        }
        try:
            ev = self.svc.events().insert(
                calendarId=self.calendar_id, body=body, conferenceDataVersion=1,
                sendUpdates="all").execute()
        except HttpError as e:
            if e.resp.status != 409:
                raise
            ev = self.svc.events().get(calendarId=self.calendar_id,
                                       eventId=eid).execute()
        return Meeting(
            event_id=ev["id"],
            start=datetime.fromisoformat(ev["start"]["dateTime"]),
            end=datetime.fromisoformat(ev["end"]["dateTime"]),
            timezone=ev["start"].get("timeZone", timezone),
            attendees=[a["email"] for a in ev.get("attendees", [])],
            join_url=ev.get("hangoutLink", ""),
            title=ev.get("summary", title))
