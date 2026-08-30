"""Turn a raw message into a clean `Email`, and drop the ones no LLM should see.

Two jobs:

1. Normalization -- strip the quoted thread and the signature block so the
   classifier reads only what the person actually typed. Quoted history is the
   single biggest source of misclassification: our own outreach copy sitting
   below the reply is full of scheduling language, and a classifier that sees it
   will happily decide the sender asked for a call when they wrote "no thanks".

2. Deterministic pre-filters -- bounces, autoresponders, and no-reply addresses
   are identified by headers and patterns, never by the model. They are ~15% of
   a real reply inbox and there is no reason to pay for a token or risk a
   hallucinated intent on any of them.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Optional

from .models import Email, Intent

# Lines that begin a quoted block. Ordered roughly by how common they are.
_QUOTE_MARKERS = [
    re.compile(r"^\s*>", re.M),
    re.compile(r"^\s*On .{5,80}\bwrote:\s*$", re.M | re.I),
    re.compile(r"^\s*-{2,}\s*Original Message\s*-{2,}\s*$", re.M | re.I),
    re.compile(r"^\s*_{5,}\s*$", re.M),
    re.compile(r"^\s*From:\s.+\bSent:\s", re.M | re.I),
    re.compile(r"^\s*Sent from my \w+", re.M | re.I),
]

# A signature starts at a `--` delimiter, or at a run of short lines at the end
# that look like contact details.
_SIG_DELIM = re.compile(r"^\s*--\s*$", re.M)
_SIG_HINT = re.compile(
    r"(?i)\b(owner|president|ceo|cfo|coo|founder|partner|managing director|"
    r"principal|vp|director|broker|advisor)\b|\(\d{3}\)\s*\d{3}-\d{4}|"
    r"\b\d{3}[.-]\d{3}[.-]\d{4}\b"
)

_AUTO_HEADERS = {"auto-submitted", "x-autoreply", "x-autorespond", "precedence"}
_AUTO_SUBJECT = re.compile(
    r"(?i)^\s*(automatic reply|auto[- ]?reply|out of (the )?office|"
    r"delivery status notification|undeliverable|mail delivery|returned mail)"
)
_BOUNCE_SENDER = re.compile(
    r"(?i)^(mailer-daemon|postmaster|no-?reply|do-?not-?reply|bounce|notifications?)@"
)
_BOUNCE_BODY = re.compile(
    r"(?i)(address not found|550 5\.\d\.\d|could not be delivered|"
    r"delivery to the following recipient failed|recipient address rejected)"
)
_OOO_BODY = re.compile(
    r"(?i)\b(i am|i'm) (currently )?(out of|away from) the office\b|"
    r"\bon (vacation|annual leave|parental leave)\b|"
    r"\bwill (be )?return(ing)? (to the office )?on\b"
)

# "Take me off your list" is handled deterministically. We do not want a model
# deciding whether an opt-out was really an opt-out.
_OPT_OUT = re.compile(
    r"(?i)\b(unsubscribe|take me off (your |the )?list|remove me from (your |the )?list|"
    r"stop (e-?mailing|contacting) me|do not (contact|email) me( again)?|"
    r"opt me out|no longer wish to receive)\b"
)


def strip_quoted(text: str) -> str:
    """Return only the text above the first quoted-thread marker."""
    cut = len(text)
    for marker in _QUOTE_MARKERS:
        m = marker.search(text)
        if m and m.start() < cut:
            cut = m.start()
    return text[:cut].rstrip()


def split_signature(text: str) -> tuple[str, str]:
    """Split a body into (message, signature). Signature may be empty."""
    m = _SIG_DELIM.search(text)
    if m:
        return text[: m.start()].rstrip(), text[m.end():].strip()

    # No explicit delimiter: consider only the final paragraph, and take it as a
    # signature when every line is short and at least one looks like a contact
    # detail (a job title or a phone number). Anchoring on the paragraph rather
    # than walking back over short lines matters -- otherwise a two-word reply
    # like "Sounds good." is itself short enough to be eaten as a signature.
    paragraphs = re.split(r"\n\s*\n", text.rstrip())
    if len(paragraphs) < 2:
        return text.rstrip(), ""
    tail_lines = [l for l in paragraphs[-1].split("\n") if l.strip()]
    if (tail_lines and len(tail_lines) <= 5
            and all(len(l.strip()) <= 60 for l in tail_lines)
            and _SIG_HINT.search(paragraphs[-1])):
        return "\n\n".join(paragraphs[:-1]).rstrip(), paragraphs[-1].strip()
    return text.rstrip(), ""


def normalize(raw: dict) -> Email:
    body_raw = raw.get("body", "")
    message, signature = split_signature(strip_quoted(body_raw))
    return Email(
        message_id=raw["message_id"],
        thread_id=raw["thread_id"],
        from_name=raw.get("from_name", ""),
        from_email=raw["from_email"].lower().strip(),
        to=raw.get("to", []),
        subject=raw.get("subject", ""),
        body=re.sub(r"\n{3,}", "\n\n", message).strip(),
        raw_body=body_raw,
        received_at=datetime.fromisoformat(raw["received_at"]),
        headers={k.lower(): v for k, v in raw.get("headers", {}).items()},
        signature=signature,
    )


def prefilter(email: Email) -> Optional[tuple[Intent, str]]:
    """Deterministic classification for messages that need no model at all.

    Returns (intent, reason) when the message is handled here, else None.
    """
    if any(h in email.headers for h in _AUTO_HEADERS):
        return Intent.AUTO_REPLY, "auto-submitted header present"
    if _AUTO_SUBJECT.search(email.subject):
        return Intent.AUTO_REPLY, f"autoresponder subject: {email.subject!r}"
    if _BOUNCE_SENDER.match(email.from_email):
        return Intent.AUTO_REPLY, f"system sender: {email.from_email}"
    if _BOUNCE_BODY.search(email.raw_body):
        return Intent.AUTO_REPLY, "body matches a delivery-failure notice"
    if _OOO_BODY.search(email.body):
        return Intent.AUTO_REPLY, "body matches an out-of-office notice"
    if _OPT_OUT.search(email.body):
        # Deliberately not left to the model: an opt-out must be honoured even
        # if it arrives wrapped in an otherwise friendly message.
        return Intent.OPT_OUT, "explicit opt-out language"
    return None


# Static red flags: cheap, high-precision regexes for topics that must reach a
# human even when the model is confident and cheerful about the message. These
# run alongside the model's own red flags -- belt and braces, because these are
# exactly the categories where a wrong automated reply is expensive.
_STATIC_FLAGS = [
    ("mentions_legal_or_litigation",
     r"(?i)\b(attorney|counsel|lawyer|lawsuit|litigation|sue|subpoena|"
     r"cease and desist|can-?spam|gdpr|legal action|report you)\b"),
    ("mentions_price_or_valuation",
     r"(?i)\b(valuation|multiple|purchase price|what would you pay|"
     r"how much (are you|would you) (paying|pay|offer)|\d+\s*x\s*(ebitda|revenue|sde)|"
     r"asking price|earn-?out|seller note|seller-financ)\b"),
    ("mentions_nda_or_contract_terms",
     r"(?i)\b(nda|non-?disclosure|loi|letter of intent|term sheet|exclusivity|"
     r"definitive agreement|indemnif)\b"),
    ("mentions_other_buyer_or_process",
     r"(?i)\b(other (parties|buyers|bidders)|competitive process|data ?room|"
     r"ioi|indications? of (value|interest)|we are running a process|banker)\b"),
]
_STATIC_FLAGS = [(name, re.compile(pat)) for name, pat in _STATIC_FLAGS]


def static_red_flags(email: Email) -> list[str]:
    text = f"{email.subject}\n{email.body}"
    return [name for name, pat in _STATIC_FLAGS if pat.search(text)]
