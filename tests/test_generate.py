"""Prompt assembly and the scheduling brief.

The drafting prompt is not testable for taste, but it is testable for the things
that make a reply dangerous: whether the model was told what it may assert,
whether it was conditioned on the right recipient, and above all whether it was
told the truth about the calendar. `scheduling_brief` is shown to both the
writer and the critic, so it gets the most attention here.
"""

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pytest

from cicero import config
from cicero.generate import (AUDIENCE, OBJECTIVE, build_system, build_user,
                             generate, scheduling_brief)
from cicero.models import (Classification, Contact, Email, Intent, SenderType)
from cicero.redact import Redactor
from helpers import CLEAN_DRAFT, ScriptedLLM

def flat(text: str) -> str:
    """Collapse wrapping. These prompts are hard-wrapped for readability, so
    asserting on raw substrings breaks the moment a line reflows -- which tests
    a formatting accident rather than the instruction."""
    return " ".join(text.split())


CT = "America/Chicago"
SLOTS = [datetime(2026, 9, 1, 13, 30, tzinfo=ZoneInfo(CT)),
         datetime(2026, 9, 2, 15, 0, tzinfo=ZoneInfo(CT))]


@pytest.fixture
def brand():
    return config.brand()


def email(body="Happy to talk. Call me on (312) 555-0142."):
    return Email(message_id="<m@x>", thread_id="t", from_name="Dana Whitfield",
                 from_email="dana@harborline.com", to=[], subject="Re: x",
                 body=body, raw_body=body,
                 received_at=datetime(2026, 8, 28, tzinfo=timezone.utc))


def contact():
    return Contact(email="dana@harborline.com", name="Dana Whitfield",
                   role=SenderType.FOUNDER, company="Harborline Mechanical",
                   timezone=CT, notes="Internal: price-sensitive, go gently")


def cls(intent=Intent.INTERESTED, role=SenderType.FOUNDER, **kw):
    base = dict(sender_type=role, sender_type_source="crm", intent=intent,
                confidence=0.9, summary="wants to keep talking",
                reasoning="r")
    base.update(kw)
    return Classification(**base)


# -- the scheduling brief: the highest-consequence text in the prompt -------

def test_book_mode_states_the_time_as_settled():
    brief = flat(scheduling_brief("book", SLOTS[:1], 30, CT, ""))
    assert "Tuesday, September 1 at 1:30 PM" in brief
    assert "30 minutes" in brief
    assert "invite" in brief.lower()


def test_book_mode_forbids_writing_a_link_into_the_email():
    """The link belongs on the invite; a link in the body is one the model made
    up."""
    assert "do NOT write a link" in flat(scheduling_brief("book", SLOTS[:1], 30,
                                                          CT, ""))


def test_propose_mode_says_plainly_that_nothing_is_booked():
    brief = flat(scheduling_brief("propose", SLOTS, 30, CT, ""))
    assert "NO invite has been created and NO meeting is booked" in brief


def test_propose_mode_denies_a_cancellation_ever_happened():
    """A live run produced 'apologies, I see a conflict' on a first contact.
    The brief now rules that out explicitly."""
    brief = flat(scheduling_brief("propose", SLOTS, 30, CT, ""))
    assert "Nothing has been cancelled, moved, or rescheduled" in brief


def test_propose_mode_numbers_every_option():
    brief = scheduling_brief("propose", SLOTS, 30, CT, "")
    assert "1. Tuesday, September 1 at 1:30 PM" in brief
    assert "2. Wednesday, September 2 at 3:00 PM" in brief


def test_stated_availability_we_could_not_meet_is_surfaced():
    """Offering times that ignore what they asked for reads as not listening."""
    brief = flat(scheduling_brief(
        "propose", SLOTS, 30, CT, "",
        stated_availability="afternoons after 1pm central"))
    assert "afternoons after 1pm central" in brief
    assert "The times above do not match it" in brief
    assert "Acknowledge what they asked for FIRST" in brief


def test_no_constraint_text_when_they_named_no_times():
    brief = flat(scheduling_brief("propose", SLOTS, 30, CT, "",
                                  stated_availability=""))
    assert "do not match it" not in brief


def test_none_mode_forbids_proposing_any_time():
    brief = flat(scheduling_brief("none", [], 30, CT, ""))
    assert "Do not propose a specific time" in brief
    assert "do not imply anything is booked" in brief


def test_propose_with_no_slots_degrades_to_none_mode():
    """An empty slot list must never render as an empty numbered list."""
    assert "No scheduling is happening" in flat(scheduling_brief(
        "propose", [], 30, CT, ""))


# -- the system prompt ------------------------------------------------------

def test_every_approved_fact_reaches_the_model(brand):
    system = flat(build_system(brand, SenderType.FOUNDER))
    for fact in brand["approved_facts"]:
        assert flat(fact["text"]) in system, f"missing fact: {fact['id']}"
    assert "Treat this as a closed world" in system


def test_the_never_say_list_reaches_the_model(brand):
    system = flat(build_system(brand, SenderType.FOUNDER))
    assert "WHAT YOU MAY NEVER SAY" in system
    for never in brand["never_say"]:
        assert flat(never) in system


def test_refusing_is_given_a_script_rather_than_forbidden(brand):
    """Models overcommit when declining has no sanctioned form."""
    system = flat(build_system(brand, SenderType.FOUNDER))
    assert "a colleague will follow up on that point" in system
    assert "That is a good answer, not an evasion." in system


def test_the_model_is_told_it_has_no_calendar(brand):
    assert "no access to a calendar" in flat(build_system(brand,
                                                          SenderType.FOUNDER))


@pytest.mark.parametrize("role", list(SenderType))
def test_every_role_has_recipient_conditioning(brand, role):
    assert flat(AUDIENCE[role]) in flat(build_system(brand, role))


def test_founder_and_broker_prompts_actually_differ(brand):
    founder = flat(build_system(brand, SenderType.FOUNDER))
    broker = flat(build_system(brand, SenderType.BROKER))
    assert founder != broker
    assert "most of their net worth is in it" in founder
    assert "most of their net worth is in it" not in broker
    assert "They are qualifying you" in broker
    assert "They are qualifying you" not in founder


# -- the user prompt --------------------------------------------------------

def test_internal_notes_are_marked_never_to_repeat():
    prompt = flat(build_user(email(), contact(), cls(), Redactor(), "none"))
    assert "price-sensitive" in prompt
    assert "never repeat it" in prompt


def test_the_senders_phone_number_is_redacted():
    prompt = build_user(email(), contact(), cls(), Redactor(), "none")
    assert "555-0142" not in prompt
    assert "<PHONE_1>" in prompt


def test_each_intent_gets_exactly_one_stated_job():
    for intent, objective in OBJECTIVE.items():
        prompt = flat(build_user(email(), contact(), cls(intent=intent),
                                 Redactor(), "none"))
        assert flat(objective) in prompt


def test_the_decline_objective_forbids_re_opening():
    prompt = flat(build_user(email(), contact(),
                             cls(intent=Intent.NOT_INTERESTED), Redactor(),
                             "none"))
    assert "do not ask why" in prompt


def test_questions_are_listed_for_the_model():
    prompt = build_user(email(), contact(),
                        cls(questions_asked=["Do you keep the team?"]),
                        Redactor(), "none")
    assert "- Do you keep the team?" in prompt


# -- the generated draft ----------------------------------------------------

def test_redacted_identifiers_are_restored_in_the_final_draft(brand):
    """The model works on tokens; the recipient must never see one."""
    redactor = Redactor()
    llm = ScriptedLLM(draft="Call me back on <PHONE_1>.\n\nBest,\nAkshat")
    draft = generate(email=email(), contact=contact(), classification=cls(),
                     brand=brand, llm=llm, redactor=redactor)
    assert "(312) 555-0142" in draft.body
    assert "<PHONE_1>" not in draft.body


def test_the_subject_stays_on_the_thread(brand):
    llm = ScriptedLLM(draft=CLEAN_DRAFT)
    draft = generate(email=email(), contact=contact(), classification=cls(),
                     brand=brand, llm=llm, redactor=Redactor())
    assert draft.subject == "Re: x"


def test_a_subject_without_re_gets_one(brand):
    mail = email()
    mail.subject = "Harborline Mechanical"
    llm = ScriptedLLM(draft=CLEAN_DRAFT)
    draft = generate(email=mail, contact=contact(), classification=cls(),
                     brand=brand, llm=llm, redactor=Redactor())
    assert draft.subject == "Re: Harborline Mechanical"
