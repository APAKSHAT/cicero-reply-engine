"""The pre-filter and normalizer are the cheapest guardrails in the system, so
they get the most direct tests."""

from cicero import ingest
from cicero.models import Intent


def mk(body, subject="Re: Test", frm="a@b.com", headers=None):
    return ingest.normalize({
        "message_id": "<x@y>", "thread_id": "t", "from_name": "A",
        "from_email": frm, "to": ["r@c.com"], "subject": subject,
        "received_at": "2026-08-28T10:00:00-04:00",
        "headers": headers or {}, "body": body})


def test_quoted_thread_is_removed():
    e = mk("No thanks, not selling.\n\n> On Aug 24, 2026, Akshat Pandey wrote:\n"
           "> Would you be open to a call next week?")
    assert "call next week" not in e.body
    assert e.body == "No thanks, not selling."


def test_outlook_style_quote_is_removed():
    e = mk("Not interested.\n\n-----Original Message-----\nFrom: Akshat\n"
           "Are you free Tuesday?")
    assert "Tuesday" not in e.body


def test_signature_split():
    e = mk("Happy to talk.\n\n--\nDana Whitfield\nOwner, Harborline\n"
           "(312) 555-0142")
    assert e.body == "Happy to talk."
    assert "555-0142" in e.signature


def test_signature_without_delimiter():
    e = mk("Sounds good.\n\nAngela Sunderland\nCFO, Clearwater Machining")
    assert e.body == "Sounds good."
    assert "CFO" in e.signature


def test_autoreply_header_prefiltered():
    assert ingest.prefilter(mk("I am away.", headers={"Auto-Submitted": "auto-replied"}))[0] \
        is Intent.AUTO_REPLY


def test_bounce_prefiltered():
    e = mk("Address not found. 550 5.1.1 no such user",
           subject="Delivery Status Notification (Failure)",
           frm="mailer-daemon@googlemail.com")
    assert ingest.prefilter(e)[0] is Intent.AUTO_REPLY


def test_ooo_prefiltered():
    assert ingest.prefilter(mk("I'm out of the office until Sept 8."))[0] \
        is Intent.AUTO_REPLY


def test_opt_out_prefiltered_even_when_polite():
    e = mk("Thanks for thinking of us, but please take me off your list.")
    assert ingest.prefilter(e)[0] is Intent.OPT_OUT


def test_static_flags_catch_price_and_legal():
    e = mk("What multiple are you paying? I've copied our attorney.")
    flags = ingest.static_red_flags(e)
    assert "mentions_price_or_valuation" in flags
    assert "mentions_legal_or_litigation" in flags


def test_static_flags_quiet_on_a_plain_message():
    assert ingest.static_red_flags(mk("Sure, happy to chat next week.")) == []
