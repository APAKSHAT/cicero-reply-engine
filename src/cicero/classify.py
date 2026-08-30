"""Stage 1 -- what does this reply actually mean?

Design notes that matter more than the code:

* **The CRM outranks the model on identity.** We sent the original email, so we
  almost always know whether we wrote to a founder or a broker. The model is
  asked for the role anyway -- not to decide it, but so that a *disagreement*
  with the CRM becomes a signal ("this is not the person we mailed") rather
  than passing silently.

* **Evidence must be quoted.** For both role and intent the model has to return
  a verbatim span from the message. We then check that span really occurs in
  the body. A model that cannot point at the words it is reasoning from is a
  model that is inventing, and we discount its confidence accordingly.

* **Confidence is not taken at face value.** Self-reported confidence from an
  LLM is poorly calibrated and skews high. We apply deterministic penalties on
  top of it -- unquoted evidence, very short messages, unknown senders, role
  disagreement -- so that the number the policy layer gates on is ours, not
  the model's.
"""

from __future__ import annotations

import re
from typing import Literal, Optional

from pydantic import BaseModel, Field

from .config import MODEL_CLASSIFY
from .llm import LLMBackend
from .models import Classification, Contact, Email, Intent, SenderType
from .redact import Redactor

RED_FLAG_VALUES = [
    "mentions_price_or_valuation",
    "mentions_legal_or_litigation",
    "mentions_nda_or_contract_terms",
    "mentions_other_buyer_or_process",
    "sender_is_upset",
    "asks_for_something_we_cannot_promise",
    "identity_uncertain",
    "references_unknown_prior_commitment",
]


class ClassificationOut(BaseModel):
    """Exactly what we ask the model for. Constrained enums, no free-form
    action field -- the model never proposes what to *do*, only what it sees."""

    sender_type: Literal["founder", "broker", "advisor", "unknown"]
    sender_type_evidence: str = Field(
        description="Verbatim span from the message that supports the role, or "
                    "empty string if the message contains no role signal.")
    intent: Literal[
        "schedule_request", "interested", "question", "needs_info",
        "not_interested", "deferred", "referral", "negotiation",
        "hostile", "opt_out", "auto_reply", "unclear",
    ]
    intent_evidence: str = Field(
        description="Verbatim span from the message that most directly supports "
                    "the chosen intent.")
    confidence: float = Field(ge=0.0, le=1.0)
    summary: str = Field(description="One sentence, plain English, for a human "
                                     "skimming a queue.")
    reasoning: str = Field(description="Two sentences at most on why this intent "
                                       "over the nearest alternative.")
    wants_call: bool = Field(description="True only if the sender affirmatively "
                                         "asked for, or agreed to, a call.")
    proposed_times_text: str = Field(
        description="Verbatim availability the sender gave, or empty string.")
    accepted_proposed_slot: Optional[int] = Field(
        default=None,
        description="If we previously proposed numbered slots and the sender "
                    "picked one, its 1-based number. Otherwise null.")
    questions_asked: list[str] = Field(
        default_factory=list,
        description="Each distinct question the sender wants answered.")
    red_flags: list[str] = Field(
        default_factory=list,
        description=f"Any that apply, from exactly this set: {RED_FLAG_VALUES}")


SYSTEM = """\
You classify replies to acquisition outreach for Cicero Capital, a firm that \
buys and operates small and mid-sized businesses. Replies come from two main \
populations:

- FOUNDERS: owner-operators of the businesses we contacted. They write in the \
first person about "my company", "my employees", "we do $X in revenue". They \
are often emotional, sometimes blunt, and are deciding about their own life's \
work.
- BROKERS: sell-side intermediaries. They write about "my client", "my seller", \
"the listing", "a teaser", "the CIM". They qualify buyers before sharing \
anything, ask about buy-box and proof of funds, and run processes with \
deadlines. A broker is never the owner of the business being discussed.
- ADVISORS: a CFO, controller, banker, lawyer, or executive at the target who \
is not the owner and is not an intermediary representing a seller for a fee.

Rules you must follow:

1. Classify ONLY the message text given to you. Do not infer facts that are not \
present. If the message is too short or too vague to support a confident \
reading, choose intent "unclear" and set a low confidence -- that is a correct \
answer, not a failure.
2. Evidence fields must be copied VERBATIM from the message, character for \
character. Never paraphrase. If no span supports your choice, return "".
3. `confidence` is the probability that a careful human reviewer would pick the \
same intent. Be honest and use the low end freely. A two-word message is not a \
0.9.
4. Distinguish these carefully, they are the ones that get confused:
   - "interested" vs "schedule_request": schedule_request requires an \
affirmative ask for or agreement to a call/meeting. Warmth alone is "interested".
   - "not_interested" vs "deferred": deferred has a future re-open ("ask me in \
the spring"); not_interested does not.
   - "question" vs "needs_info": question is answerable from what we already \
say publicly about ourselves; needs_info is a request for materials, proof of \
funds, a buy-box document, or anything we would have to produce or commit to.
   - "negotiation" covers ANY message that raises price, valuation, multiples, \
structure, or deal terms -- even casually, even as a throwaway line.
   - "referral" is being pointed at a different person, not a decline.
5. Set red flags generously. A red flag does not mean the message is bad; it \
means a human should look. Under-flagging is far more costly than over-flagging.
6. Placeholders like <EMAIL_1> or <PHONE_2> are redacted identifiers. Treat them \
as opaque and never comment on them.

You are not writing a reply and you are not deciding what happens next. Another \
system does that. Report only what is in the message.
"""


def _fmt_prior_slots(contact: Optional[Contact]) -> str:
    if not contact or not contact.proposed_slots:
        return "We have not proposed any times to this person."
    lines = [f"  {i}. {s}" for i, s in enumerate(contact.proposed_slots, 1)]
    return ("We previously proposed these numbered times to this person:\n"
            + "\n".join(lines))


def build_user_prompt(email: Email, contact: Optional[Contact],
                      redactor: Redactor) -> str:
    names = [n for n in [email.from_name, contact.name if contact else "",
                         contact.company if contact else ""] if n]
    crm = (
        f"Name: {contact.name}\nCompany: {contact.company}\n"
        f"Role on file: {contact.role.value}\nStage: {contact.deal_stage}\n"
        f"Internal note: {contact.notes}"
        if contact else
        "This sender is NOT in our CRM. We have no record of writing to them "
        "directly, so their role must be inferred from the message alone and "
        "identity should be treated as uncertain."
    )
    return f"""\
<crm_record>
{crm}
</crm_record>

<prior_scheduling>
{_fmt_prior_slots(contact)}
</prior_scheduling>

<message>
From: {email.from_name} <{redactor.scrub(email.from_email)}>
Subject: {email.subject}
Received: {email.received_at.isoformat()}

{redactor.scrub(email.body, names)}
</message>

<signature_block>
{redactor.scrub(email.signature, names) or "(none)"}
</signature_block>

Classify this reply."""


def _evidence_is_grounded(span: str, email: Email) -> bool:
    """Is the quoted span actually in the message? Cheap hallucination check."""
    if not span.strip():
        return False
    norm = lambda s: re.sub(r"\s+", " ", s).strip().lower()
    return norm(span) in norm(f"{email.subject} {email.body} {email.signature}")


def classify(email: Email, contact: Optional[Contact], llm: LLMBackend,
             redactor: Redactor) -> Classification:
    """Raises LLMError on failure -- the caller escalates, it never guesses."""
    out = llm.structured(
        model=MODEL_CLASSIFY,
        system=SYSTEM,
        user=build_user_prompt(email, contact, redactor),
        schema=ClassificationOut,
        max_tokens=8000,
    )

    # ---- Identity: CRM wins, disagreement becomes a flag -------------------
    llm_role = SenderType(out.sender_type)
    red_flags = [f for f in out.red_flags if f in RED_FLAG_VALUES]
    unknown_sender = contact is None
    if contact:
        role, source = contact.role, "crm"
        if llm_role != SenderType.UNKNOWN and llm_role != contact.role:
            # The person replying is not the person we wrote to, or is not who
            # our record says they are. That is worth a human's attention.
            red_flags.append("identity_uncertain")
    else:
        # Deliberately NOT flagged `identity_uncertain`: an unknown sender is
        # already caught by its own policy rule (r060), which holds a draft for
        # review. Flagging it here as well would escalate with no draft at all
        # and make that rule unreachable.
        role, source = llm_role, "llm"

    # ---- Confidence: start from the model, then subtract for weak grounding
    confidence = out.confidence
    penalties: list[str] = []

    if not _evidence_is_grounded(out.intent_evidence, email):
        confidence -= 0.25
        penalties.append("intent evidence not found verbatim in the message")
    if role != SenderType.UNKNOWN and source == "llm" and \
            not _evidence_is_grounded(out.sender_type_evidence, email):
        confidence -= 0.15
        penalties.append("role inferred without a grounded quote")
    if len(email.body.split()) < 8:
        confidence -= 0.20
        penalties.append("message too short to be confident about")
    if unknown_sender or "identity_uncertain" in red_flags:
        confidence -= 0.10
        penalties.append("sender identity uncertain")

    confidence = max(0.0, min(1.0, confidence))
    reasoning = out.reasoning
    if penalties:
        reasoning += "  [confidence adjusted down: " + "; ".join(penalties) + "]"

    return Classification(
        sender_type=role,
        sender_type_source=source,
        intent=Intent(out.intent),
        confidence=round(confidence, 3),
        summary=out.summary,
        reasoning=reasoning,
        red_flags=sorted(set(red_flags)),
        wants_call=out.wants_call,
        proposed_times_text=out.proposed_times_text,
        accepted_proposed_slot=out.accepted_proposed_slot,
        questions_asked=out.questions_asked,
    )
