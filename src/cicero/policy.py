"""Stage 2 -- the decision layer. No LLM touches this file.

This is the part of the system I would defend hardest in review. The model is
allowed to *read* and to *write*, but the decision to put an email in someone's
inbox is made by ordered, named rules over a small set of facts. Every decision
carries the id of the rule that produced it, so any outcome in the ledger can be
traced to a line in `config/policy.yaml` rather than to a prompt.

Rules are evaluated top to bottom and the first match wins. They are ordered
most-restrictive-first on purpose: a message can only reach `AUTO_SEND` by
falling through every gate above it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .models import (Action, Classification, Contact, Decision, Email,
                     Intent)


@dataclass
class RunState:
    """Cross-message state for one pipeline run -- the circuit breaker."""
    auto_sends: int = 0

    def budget_left(self, cap: int) -> bool:
        return self.auto_sends < cap


def scheduling_warranted(c: Classification) -> bool:
    """Consent, defined narrowly and in one place.

    A calendar invite is an intrusion. We require the sender to have actually
    asked for a call or accepted one we offered. Enthusiasm is not consent:
    "this sounds interesting, tell me more" gets a reply, not an invite.
    """
    if c.accepted_proposed_slot is not None:
        return True
    return c.intent == Intent.SCHEDULE_REQUEST and c.wants_call


def decide(
    email: Email,
    contact: Optional[Contact],
    classification: Optional[Classification],
    policy: dict,
    run: RunState,
    *,
    prefiltered: Optional[tuple[Intent, str]] = None,
    static_flags: Optional[list[str]] = None,
    classifier_error: str = "",
) -> Decision:
    lim = policy["rate_limits"]
    conf = policy["confidence"]
    static_flags = static_flags or []

    # -- 0. Suppression list. Nothing gets past an opt-out, ever. ------------
    if contact and contact.opted_out:
        return Decision(Action.SUPPRESS, "Contact is on the suppression list.",
                        "r000_opted_out")

    # -- 1. Deterministic pre-filters, decided before any model ran ----------
    if prefiltered:
        intent, why = prefiltered
        if intent == Intent.OPT_OUT:
            # Honour it silently and tell a human, because opt-outs that arrive
            # with a legal threat attached are a person's problem, not a bot's.
            if any(f in static_flags for f in
                   ("mentions_legal_or_litigation",)):
                return Decision(
                    Action.ESCALATE,
                    f"Opt-out with a legal reference ({why}). Suppressed and "
                    "flagged for a person to acknowledge by hand.",
                    "r002_opt_out_with_legal_flag")
            return Decision(Action.SUPPRESS,
                            f"Opt-out honoured, contact suppressed ({why}).",
                            "r001_opt_out")
        return Decision(Action.SUPPRESS, f"Machine-generated mail ({why}).",
                        "r003_auto_generated")

    # -- 2. We could not classify at all -------------------------------------
    if classification is None:
        return Decision(Action.ESCALATE,
                        f"Classifier failed, so we have no idea what this says. "
                        f"({classifier_error or 'no classification'})",
                        "r010_classifier_failed")

    c = classification
    flags = sorted(set(c.red_flags) | set(static_flags))

    # -- 3. Intents that end the conversation --------------------------------
    if c.intent.value in policy["suppress_intents"]:
        return Decision(Action.SUPPRESS,
                        f"Intent '{c.intent.value}' needs no reply.",
                        "r020_suppress_intent")

    # -- 4. Intents a machine may never answer -------------------------------
    # Note: we escalate WITHOUT generating a draft. A plausible draft sitting
    # next to a hard case is how a reviewer ends up rubber-stamping the exact
    # message we built this layer to prevent.
    if c.intent.value in policy["always_human_intents"]:
        return Decision(Action.ESCALATE,
                        f"Intent '{c.intent.value}' is always handled by a "
                        f"person. {c.summary}",
                        "r030_always_human_intent")

    # -- 5. Topic tripwires ---------------------------------------------------
    hits = [f for f in flags if f in policy["escalate_on_red_flags"]]
    if hits:
        return Decision(Action.ESCALATE,
                        f"Red flag(s): {', '.join(hits)}.",
                        "r040_red_flag")

    # -- 6. Confidence floors -------------------------------------------------
    if c.confidence < conf["draft_floor"]:
        return Decision(Action.ESCALATE,
                        f"Confidence {c.confidence:.2f} is below the drafting "
                        f"floor of {conf['draft_floor']}; a draft would be a "
                        f"guess dressed up as an answer.",
                        "r050_below_draft_floor")

    # -- 7. Relationship-level gates -----------------------------------------
    if contact is None and lim["require_human_for_unknown_senders"]:
        return Decision(Action.DRAFT_FOR_REVIEW,
                        "Sender is not in the CRM, so we cannot confirm who "
                        "they are or what we told them.",
                        "r060_unknown_sender")

    if lim["never_auto_reply_after_human_touch"] and contact and \
            contact.human_has_touched_thread:
        return Decision(Action.DRAFT_FOR_REVIEW,
                        "A colleague has already written on this thread by "
                        "hand; the machine must not talk over them.",
                        "r070_human_in_thread")

    if contact and contact.auto_replies_sent >= lim["max_consecutive_auto_replies"]:
        return Decision(Action.DRAFT_FOR_REVIEW,
                        f"{contact.auto_replies_sent} consecutive automated "
                        f"replies already sent to this contact; a person takes "
                        f"it from here.",
                        "r080_auto_reply_cap")

    if c.confidence < conf["auto_send_floor"]:
        return Decision(Action.DRAFT_FOR_REVIEW,
                        f"Confidence {c.confidence:.2f} is below the auto-send "
                        f"floor of {conf['auto_send_floor']}.",
                        "r090_below_auto_send_floor")

    if c.intent.value not in policy["auto_send_intents"]:
        return Decision(Action.DRAFT_FOR_REVIEW,
                        f"Intent '{c.intent.value}' is not on the auto-send "
                        f"allowlist.",
                        "r100_intent_not_allowlisted")

    if not run.budget_left(lim["max_auto_sends_per_run"]):
        return Decision(Action.DRAFT_FOR_REVIEW,
                        f"Run cap of {lim['max_auto_sends_per_run']} automated "
                        f"sends reached; the rest queue for review.",
                        "r110_run_budget_exhausted")

    # -- 8. Cleared to act ----------------------------------------------------
    if scheduling_warranted(c):
        return Decision(Action.SCHEDULE_AND_SEND,
                        "Sender asked for or accepted a call and every gate "
                        "passed.",
                        "r200_schedule")

    return Decision(Action.AUTO_SEND,
                    f"'{c.intent.value}' from a known {c.sender_type.value} at "
                    f"confidence {c.confidence:.2f}; all gates passed.",
                    "r210_auto_send")
