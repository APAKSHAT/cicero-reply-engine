"""SQLite ledger. Every message the system sees leaves exactly one row.

Two reasons this exists beyond debugging:

* **Idempotency.** `message_id` is unique. Re-running the pipeline over the same
  inbox cannot send a second reply. Combined with the deterministic calendar
  event id, the whole pipeline is safe to retry after a crash.
* **Accountability.** The ledger stores the classification, the rule id that
  decided the action, the draft, and the verdict. When someone asks "why did we
  send that", the answer is a row, not a re-run of a non-deterministic model.

Contact state (opt-outs, consecutive auto-replies, times we proposed) lives here
too, overlaid on the static CRM fixture -- the fixture is what we knew, the
ledger is what has happened since.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS outcomes (
    message_id     TEXT PRIMARY KEY,
    thread_id      TEXT NOT NULL,
    contact_email  TEXT NOT NULL,
    received_at    TEXT NOT NULL,
    processed_at   TEXT NOT NULL,
    sender_type    TEXT,
    intent         TEXT,
    confidence     REAL,
    action         TEXT NOT NULL,
    rule_id        TEXT NOT NULL,
    reason         TEXT NOT NULL,
    red_flags      TEXT,
    draft_subject  TEXT,
    draft_body     TEXT,
    verdict_passed INTEGER,
    verdict_issues TEXT,
    meeting_id     TEXT,
    meeting_start  TEXT,
    sent           INTEGER NOT NULL DEFAULT 0,
    review_status  TEXT NOT NULL DEFAULT 'n/a',
    error          TEXT,
    payload        TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS contact_state (
    email                    TEXT PRIMARY KEY,
    opted_out                INTEGER NOT NULL DEFAULT 0,
    auto_replies_sent        INTEGER NOT NULL DEFAULT 0,
    human_has_touched_thread INTEGER NOT NULL DEFAULT 0,
    proposed_slots           TEXT NOT NULL DEFAULT '[]',
    meeting_booked           INTEGER NOT NULL DEFAULT 0,
    updated_at               TEXT
);

CREATE INDEX IF NOT EXISTS idx_outcomes_review ON outcomes(review_status);
CREATE INDEX IF NOT EXISTS idx_outcomes_thread ON outcomes(thread_id);
"""


class Store:
    def __init__(self, path: Path):
        self.path = path
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        self.db.commit()

    # -- dedupe -------------------------------------------------------------
    def already_processed(self, message_id: str) -> bool:
        row = self.db.execute(
            "SELECT 1 FROM outcomes WHERE message_id = ?", (message_id,)).fetchone()
        return row is not None

    # -- writes -------------------------------------------------------------
    def record(self, outcome: Any, review_status: str = "n/a") -> None:
        o = outcome
        c, d, v, m = o.classification, o.draft, o.verdict, o.meeting
        self.db.execute(
            """INSERT OR REPLACE INTO outcomes VALUES
               (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                o.email.message_id, o.email.thread_id, o.email.from_email,
                o.email.received_at.isoformat(),
                datetime.now(timezone.utc).isoformat(),
                c.sender_type.value if c else None,
                c.intent.value if c else None,
                c.confidence if c else None,
                o.decision.action.value, o.decision.rule_id, o.decision.reason,
                json.dumps(c.red_flags) if c else "[]",
                d.subject if d else None, d.body if d else None,
                int(v.passed) if v else None,
                json.dumps([i.__dict__ for i in v.issues]) if v else "[]",
                m.event_id if m else None,
                m.start.isoformat() if m else None,
                int(o.sent), review_status, o.error,
                json.dumps(o.to_dict(), default=str),
            ))
        self.db.commit()

    def set_review_status(self, message_id: str, status: str,
                          sent: Optional[bool] = None) -> None:
        if sent is None:
            self.db.execute("UPDATE outcomes SET review_status=? WHERE message_id=?",
                            (status, message_id))
        else:
            self.db.execute(
                "UPDATE outcomes SET review_status=?, sent=? WHERE message_id=?",
                (status, int(sent), message_id))
        self.db.commit()

    # -- contact state ------------------------------------------------------
    def contact_state(self, email: str) -> dict:
        row = self.db.execute("SELECT * FROM contact_state WHERE email=?",
                              (email,)).fetchone()
        if not row:
            return {}
        return {
            "opted_out": bool(row["opted_out"]),
            "auto_replies_sent": row["auto_replies_sent"],
            "human_has_touched_thread": bool(row["human_has_touched_thread"]),
            "proposed_slots": json.loads(row["proposed_slots"]),
            "meeting_booked": bool(row["meeting_booked"]),
        }

    def update_contact(self, email: str, **fields: Any) -> None:
        current = self.contact_state(email) or {
            "opted_out": False, "auto_replies_sent": 0,
            "human_has_touched_thread": False, "proposed_slots": [],
            "meeting_booked": False}
        current.update(fields)
        self.db.execute(
            """INSERT OR REPLACE INTO contact_state
               (email, opted_out, auto_replies_sent, human_has_touched_thread,
                proposed_slots, meeting_booked, updated_at)
               VALUES (?,?,?,?,?,?,?)""",
            (email, int(current["opted_out"]), current["auto_replies_sent"],
             int(current["human_has_touched_thread"]),
             json.dumps(current["proposed_slots"]),
             int(current["meeting_booked"]),
             datetime.now(timezone.utc).isoformat()))
        self.db.commit()

    # -- reads --------------------------------------------------------------
    def pending_review(self) -> list[sqlite3.Row]:
        return self.db.execute(
            "SELECT * FROM outcomes WHERE review_status='pending' "
            "ORDER BY received_at").fetchall()

    def all_outcomes(self) -> list[sqlite3.Row]:
        return self.db.execute(
            "SELECT * FROM outcomes ORDER BY received_at").fetchall()
