"""Core domain types.

Everything that moves between pipeline stages is one of these. They are plain
dataclasses (not Pydantic) except where a model must validate LLM output --
those live in `schemas.py` so the boundary between "our data" and "model
output we do not trust yet" stays visible.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Any, Optional


# --------------------------------------------------------------------------
# Taxonomies
# --------------------------------------------------------------------------

class SenderType(str, Enum):
    FOUNDER = "founder"            # owner/operator of a target business
    BROKER = "broker"              # intermediary representing a seller
    ADVISOR = "advisor"            # banker, lawyer, accountant, exec at target
    UNKNOWN = "unknown"


class Intent(str, Enum):
    SCHEDULE_REQUEST = "schedule_request"   # explicitly wants a call
    INTERESTED = "interested"               # positive, no call asked for yet
    QUESTION = "question"                   # asking something answerable
    NEEDS_INFO = "needs_info"               # wants materials/thesis/proof of funds
    NOT_INTERESTED = "not_interested"       # polite decline
    DEFERRED = "deferred"                   # "not now, circle back in Q3"
    REFERRAL = "referral"                   # "talk to my broker/CFO instead"
    NEGOTIATION = "negotiation"             # price, terms, LOI, exclusivity
    HOSTILE = "hostile"                     # angry, accusatory, legal threat
    OPT_OUT = "opt_out"                     # remove me / unsubscribe
    AUTO_REPLY = "auto_reply"               # OOO, bounce, autoresponder
    UNCLEAR = "unclear"                     # model could not tell


class Action(str, Enum):
    AUTO_SEND = "auto_send"                 # generate + send, no human
    SCHEDULE_AND_SEND = "schedule_and_send" # book calendar + send invite email
    DRAFT_FOR_REVIEW = "draft_for_review"   # generate, hold for a human
    ESCALATE = "escalate"                   # do not draft; flag a human
    SUPPRESS = "suppress"                   # do nothing, record and stop


# --------------------------------------------------------------------------
# Inbound / CRM
# --------------------------------------------------------------------------

@dataclass
class Email:
    """A single inbound message, already normalized by `ingest`."""
    message_id: str
    thread_id: str
    from_name: str
    from_email: str
    to: list[str]
    subject: str
    body: str                      # latest reply only, quotes/signature stripped
    raw_body: str                  # full text as received
    received_at: datetime
    headers: dict[str, str] = field(default_factory=dict)
    signature: str = ""            # stripped signature block, kept for role signals


@dataclass
class Contact:
    """What the CRM already knows. Authoritative over anything the LLM infers."""
    email: str
    name: str
    role: SenderType
    company: str
    timezone: str = "America/New_York"
    deal_stage: str = "cold_outreach"
    notes: str = ""
    opted_out: bool = False
    # Conversation state, updated by the pipeline
    auto_replies_sent: int = 0
    human_has_touched_thread: bool = False
    proposed_slots: list[str] = field(default_factory=list)
    meeting_booked: bool = False


# --------------------------------------------------------------------------
# Stage outputs
# --------------------------------------------------------------------------

@dataclass
class Classification:
    sender_type: SenderType
    sender_type_source: str           # "crm" | "llm" | "heuristic"
    intent: Intent
    confidence: float                 # 0-1, model's own calibrated confidence
    summary: str
    reasoning: str
    red_flags: list[str] = field(default_factory=list)
    wants_call: bool = False
    proposed_times_text: str = ""     # verbatim scheduling language, if any
    accepted_proposed_slot: Optional[int] = None  # index into contact.proposed_slots
    questions_asked: list[str] = field(default_factory=list)


@dataclass
class Decision:
    action: Action
    reason: str                       # human-readable, shown in the review queue
    rule_id: str                      # which policy rule fired


@dataclass
class Draft:
    subject: str
    body: str
    model: str


@dataclass
class VerdictIssue:
    severity: str                     # "block" | "warn"
    code: str
    detail: str


@dataclass
class Verdict:
    passed: bool
    issues: list[VerdictIssue] = field(default_factory=list)

    @property
    def blocking(self) -> list[VerdictIssue]:
        return [i for i in self.issues if i.severity == "block"]


@dataclass
class Meeting:
    event_id: str
    start: datetime
    end: datetime
    timezone: str
    attendees: list[str]
    join_url: str
    title: str


@dataclass
class Outcome:
    """The full record of what happened to one inbound email."""
    email: Email
    contact: Optional[Contact]
    classification: Optional[Classification] = None
    decision: Optional[Decision] = None
    draft: Optional[Draft] = None
    verdict: Optional[Verdict] = None
    meeting: Optional[Meeting] = None
    sent: bool = False
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        def enc(o: Any) -> Any:
            if isinstance(o, Enum):
                return o.value
            if isinstance(o, datetime):
                return o.isoformat()
            if hasattr(o, "__dataclass_fields__"):
                return {k: enc(v) for k, v in asdict(o).items()}
            if isinstance(o, list):
                return [enc(i) for i in o]
            if isinstance(o, dict):
                return {k: enc(v) for k, v in o.items()}
            return o
        return enc(self)
