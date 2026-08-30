"""Orchestration. One email in, one row in the ledger out.

    normalize -> dedupe -> prefilter -> classify -> DECIDE -> [schedule] ->
    draft -> verify -> book -> send

Two things about that order are load-bearing. `decide` runs before anything is
written, so no draft exists for a message the policy layer already refused --
there is nothing for a tired reviewer to accidentally approve. And booking
happens *after* verification and immediately before the send, so a draft that
fails its checks can never leave a real invite on a real person's calendar.

`process_one` is the whole flow at a glance; each stage below it does one thing
and returns either a replacement `Decision` (meaning: stop, route it here) or
None (meaning: carry on).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from . import classify as classify_mod
from . import generate as generate_mod
from . import ingest, scheduling
from . import verify as verify_mod
from .adapters.base import CalendarAdapter, MailAdapter
from .llm import LLMBackend, LLMError
from .models import (Action, Classification, Contact, Decision, Email, Intent,
                     Outcome, SenderType)
from .policy import RunState, decide
from .redact import Redactor
from .store import Store


@dataclass
class SchedulingPlan:
    """What we intend to do about a call, before any calendar write happens."""
    mode: str = "none"                                  # book | propose | none
    slots: list[datetime] = field(default_factory=list)
    note: str = ""


def load_contacts(path: Path, store: Store) -> dict[str, Contact]:
    """CRM fixture overlaid with whatever the ledger has learned since."""
    contacts: dict[str, Contact] = {}
    with open(path) as f:
        for row in json.load(f):
            c = Contact(
                email=row["email"].lower(), name=row["name"],
                role=SenderType(row["role"]), company=row["company"],
                timezone=row.get("timezone", "America/New_York"),
                deal_stage=row.get("deal_stage", "cold_outreach"),
                notes=row.get("notes", ""),
                opted_out=row.get("opted_out", False),
                auto_replies_sent=row.get("auto_replies_sent", 0),
                human_has_touched_thread=row.get("human_has_touched_thread", False),
                proposed_slots=row.get("proposed_slots", []),
                meeting_booked=row.get("meeting_booked", False))
            for k, v in store.contact_state(c.email).items():
                setattr(c, k, v)
            contacts[c.email] = c
    return contacts


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------

def _classify(email: Email, contact: Optional[Contact], llm: LLMBackend,
              redactor: Redactor,
              prefiltered: Optional[tuple[Intent, str]],
              ) -> tuple[Optional[Classification], str]:
    """Classify, unless a pre-filter already settled it. Never raises: a failure
    comes back as an error string so the policy layer can escalate it."""
    if prefiltered is not None:
        return None, ""
    try:
        return classify_mod.classify(email, contact, llm, redactor), ""
    except LLMError as e:
        return None, str(e)


def _plan_scheduling(
    *, classification: Classification, contact: Optional[Contact], email: Email,
    calendar: CalendarAdapter, policy: dict, llm: LLMBackend, now: datetime,
) -> tuple[SchedulingPlan, Optional[Decision]]:
    """Choose book vs propose. Writes nothing to any calendar."""
    if contact is None:
        return SchedulingPlan(), None

    mode, slots, note = scheduling.resolve(
        classification=classification, contact=contact,
        received_at=email.received_at, now=now, calendar=calendar,
        sched=policy["scheduling"], llm=llm)

    if mode == "none":
        return SchedulingPlan(), Decision(
            Action.DRAFT_FOR_REVIEW,
            f"They asked for a call but we could not find a time: {note}",
            "r201_no_slot_available")

    if mode == "propose" and not note:
        # Consent to a conversation is not consent to a specific hour.
        note = "no time named; proposing rather than booking"
    return SchedulingPlan(mode, slots, note), None


def _draft(
    *, outcome: Outcome, contact: Optional[Contact], plan: SchedulingPlan,
    brand: dict, policy: dict, llm: LLMBackend, redactor: Redactor,
) -> Optional[Decision]:
    """Write the reply. On failure, escalate rather than send something empty."""
    try:
        outcome.draft = generate_mod.generate(
            email=outcome.email, contact=contact,
            classification=outcome.classification, brand=brand, llm=llm,
            redactor=redactor, mode=plan.mode, slots=plan.slots,
            duration=policy["scheduling"]["duration_minutes"], note=plan.note)
    except LLMError as e:
        outcome.error = f"generation failed: {e}"
        return Decision(Action.ESCALATE, f"Could not draft a reply ({e}).",
                        "r300_generation_failed")
    return None


def _verify(
    *, outcome: Outcome, contact: Optional[Contact], plan: SchedulingPlan,
    brand: dict, policy: dict, llm: LLMBackend,
) -> Optional[Decision]:
    """Check the draft. The writer and the critic are shown the identical
    scheduling statement -- deriving it twice is how a draft comes to claim a
    booking the checker thinks is a proposal."""
    facts = generate_mod.scheduling_brief(
        plan.mode, plan.slots, policy["scheduling"]["duration_minutes"],
        contact.timezone if contact else "America/New_York", plan.note,
        outcome.classification.proposed_times_text)

    outcome.verdict = verify_mod.verify(
        draft=outcome.draft, email=outcome.email,
        classification=outcome.classification, contact=contact, brand=brand,
        policy=policy, llm=llm, scheduling_mode=plan.mode,
        scheduling_facts=facts)

    if outcome.verdict.passed:
        return None
    reasons = "; ".join(f"{i.code}: {i.detail}" for i in outcome.verdict.blocking)
    return Decision(Action.DRAFT_FOR_REVIEW,
                    f"Draft failed verification, held for a person. [{reasons}]",
                    "r400_failed_verification")


def _deliver(
    *, outcome: Outcome, contact: Optional[Contact], plan: SchedulingPlan,
    mail: MailAdapter, calendar: CalendarAdapter, brand: dict, policy: dict,
    run: RunState,
) -> Optional[Decision]:
    """Book, then send. In that order, so a failed calendar write cannot leave
    us having promised a meeting that does not exist."""
    email = outcome.email

    if plan.mode == "book" and contact:
        try:
            outcome.meeting = scheduling.book(
                calendar=calendar, start=plan.slots[0], contact=contact,
                thread_id=email.thread_id, sched=policy["scheduling"],
                brand=brand, summary=outcome.classification.summary)
        except Exception as e:                        # noqa: BLE001 - any failure
            outcome.error = f"calendar write failed: {e}"
            return Decision(
                Action.DRAFT_FOR_REVIEW,
                f"Reply was approved but the calendar write failed ({e}); "
                f"holding the email so we do not promise a meeting that does "
                f"not exist.",
                "r500_calendar_failed")

    mail.send_reply(thread_id=email.thread_id, in_reply_to=email.message_id,
                    to=email.from_email, subject=outcome.draft.subject,
                    body=outcome.draft.body)
    outcome.sent = True
    run.auto_sends += 1
    return None


def _remember(outcome: Outcome, contact: Optional[Contact],
              plan: SchedulingPlan, store: Store) -> None:
    """Advance the contact's state so the next reply is judged in context."""
    if contact is None:
        return
    state: dict = {"auto_replies_sent": contact.auto_replies_sent + 1}
    if plan.mode == "propose":
        state["proposed_slots"] = [s.isoformat() for s in plan.slots]
    if outcome.meeting:
        state["meeting_booked"] = True
        state["proposed_slots"] = []
    store.update_contact(contact.email, **state)


# ---------------------------------------------------------------------------
# The flow
# ---------------------------------------------------------------------------

def process_one(
    *, raw: dict, contacts: dict[str, Contact], store: Store, llm: LLMBackend,
    mail: MailAdapter, calendar: CalendarAdapter, brand: dict, policy: dict,
    run: RunState, now: datetime,
) -> Optional[Outcome]:
    email = ingest.normalize(raw)
    if store.already_processed(email.message_id):
        return None                                     # replay-safe

    contact = contacts.get(email.from_email)
    redactor = Redactor(redact_names=policy.get("redact_names", False))
    outcome = Outcome(email=email, contact=contact)

    prefiltered = ingest.prefilter(email)
    static_flags = ingest.static_red_flags(email)
    outcome.classification, classifier_error = _classify(
        email, contact, llm, redactor, prefiltered)

    outcome.decision = decide(
        email, contact, outcome.classification, policy, run,
        prefiltered=prefiltered, static_flags=static_flags,
        classifier_error=classifier_error)

    # Record every flag that was actually weighed, not just the model's. An
    # escalation fired by a static regex flag used to show a reviewer an empty
    # list, which made the ledger row unable to explain its own decision.
    if outcome.classification and static_flags:
        outcome.classification.red_flags = sorted(
            set(outcome.classification.red_flags) | set(static_flags))

    # Honour an opt-out immediately, whichever branch produced it.
    if prefiltered and prefiltered[0] is Intent.OPT_OUT:
        store.update_contact(email.from_email, opted_out=True)

    action = outcome.decision.action
    if action is Action.SUPPRESS:
        store.record(outcome, review_status="n/a")
        return outcome
    if action is Action.ESCALATE:
        store.record(outcome, review_status="pending")
        return outcome

    plan = SchedulingPlan()
    if action is Action.SCHEDULE_AND_SEND:
        plan, override = _plan_scheduling(
            classification=outcome.classification, contact=contact, email=email,
            calendar=calendar, policy=policy, llm=llm, now=now)
        if override:
            outcome.decision, action = override, override.action

    # Draft. A generation failure escalates -- we do not send an empty reply.
    override = _draft(outcome=outcome, contact=contact, plan=plan, brand=brand,
                      policy=policy, llm=llm, redactor=redactor)
    if override:
        outcome.decision = override
        store.record(outcome, review_status="pending")
        return outcome

    # Verify. A failed check never rewrites the draft, it hands it to a person.
    override = _verify(outcome=outcome, contact=contact, plan=plan, brand=brand,
                       policy=policy, llm=llm)
    if override:
        outcome.decision, action = override, override.action

    if action is Action.DRAFT_FOR_REVIEW:
        mail.create_draft(thread_id=email.thread_id,
                          in_reply_to=email.message_id, to=email.from_email,
                          subject=outcome.draft.subject,
                          body=outcome.draft.body)
        store.record(outcome, review_status="pending")
        return outcome

    if policy["dry_run"]:
        outcome.decision = Decision(
            action,
            outcome.decision.reason + "  [DRY RUN: nothing sent or booked]",
            outcome.decision.rule_id)
        store.record(outcome, review_status="pending")
        return outcome

    override = _deliver(outcome=outcome, contact=contact, plan=plan, mail=mail,
                        calendar=calendar, brand=brand, policy=policy, run=run)
    if override:
        outcome.decision = override
        store.record(outcome, review_status="pending")
        return outcome

    _remember(outcome, contact, plan, store)
    store.record(outcome, review_status="sent")
    return outcome


def run_pipeline(
    *, mail: MailAdapter, calendar: CalendarAdapter, store: Store,
    llm: LLMBackend, contacts_path: Path, brand: dict, policy: dict,
    now: Optional[datetime] = None,
) -> list[Outcome]:
    now = now or datetime.now(timezone.utc)
    contacts = load_contacts(contacts_path, store)
    run = RunState()
    outcomes: list[Outcome] = []
    for raw in mail.fetch_replies():
        try:
            o = process_one(raw=raw, contacts=contacts, store=store, llm=llm,
                            mail=mail, calendar=calendar, brand=brand,
                            policy=policy, run=run, now=now)
        except Exception as e:                        # noqa: BLE001
            # An unhandled error must never take down the run or, worse, get
            # retried into a duplicate send. Record it and move on.
            email = ingest.normalize(raw)
            o = Outcome(email=email, contact=contacts.get(email.from_email),
                        decision=Decision(Action.ESCALATE,
                                          f"Unhandled error: {e}",
                                          "r999_unhandled_error"),
                        error=repr(e))
            store.record(o, review_status="pending")
        if o:
            outcomes.append(o)
    return outcomes
