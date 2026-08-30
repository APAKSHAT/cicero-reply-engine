"""Command line: `run`, `review`, `ledger`, `show`.

`review` is the human-in-the-loop surface. It is a terminal queue on purpose --
the interesting question is *what a reviewer is shown*, not what it is rendered
in. Each item shows the classification, the confidence, the exact policy rule
that held it, and the verifier's objections, so the reviewer is deciding with
the same information the machine had.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from . import config
from .adapters.mock import MockCalendar, MockGmail
from .llm import default_llm
from .models import Action
from .pipeline import run_pipeline
from .store import Store

ROOT = config.ROOT
DB_PATH = ROOT / "cicero.db"

C = {
    "reset": "\033[0m", "dim": "\033[2m", "bold": "\033[1m",
    "green": "\033[32m", "yellow": "\033[33m", "red": "\033[31m",
    "blue": "\033[34m", "cyan": "\033[36m", "mag": "\033[35m",
}
if not sys.stdout.isatty() or os.getenv("NO_COLOR"):
    C = {k: "" for k in C}

ACTION_COLOR = {
    Action.AUTO_SEND.value: C["green"],
    Action.SCHEDULE_AND_SEND.value: C["cyan"],
    Action.DRAFT_FOR_REVIEW.value: C["yellow"],
    Action.ESCALATE.value: C["red"],
    Action.SUPPRESS.value: C["dim"],
}


def _adapters(args, now, policy):
    """Mock by default. `--real` swaps in Gmail and Google Calendar behind the
    identical interfaces, so nothing else in the pipeline changes."""
    if not getattr(args, "real", False):
        return (MockGmail(config.DATA_DIR / "inbox.json"),
                MockCalendar(now, policy["scheduling"]["organizer_timezone"]))

    from .adapters.google import (GmailAdapter, GoogleCalendarAdapter,
                                  build_services)
    gmail_svc, cal_svc = build_services()
    return (GmailAdapter(gmail_svc, query=args.query),
            GoogleCalendarAdapter(cal_svc))


def _build(args) -> tuple:
    brand, policy = config.brand(), config.policy()
    if args.live:
        policy = {**policy, "dry_run": False}
    now = (datetime.fromisoformat(args.now) if args.now
           else datetime.now(timezone.utc))
    store = Store(DB_PATH)
    mail, calendar = _adapters(args, now, policy)
    return brand, policy, store, mail, calendar, now


def cmd_run(args) -> int:
    brand, policy, store, mail, calendar, now = _build(args)
    target = "REAL Gmail + Google Calendar" if args.real else "mock adapters"
    mode = ("LIVE -- replies will be sent and invites created"
            if not policy["dry_run"] else
            "DRY RUN -- full pipeline, nothing sent or booked")
    print(f"\n{C['bold']}Cicero reply engine{C['reset']}  {C['dim']}({mode}){C['reset']}")
    banner = C["red"] if (args.real and not policy["dry_run"]) else C["dim"]
    print(f"{banner}target: {target}{C['reset']}")
    print(f"{C['dim']}now = {now.isoformat()}{C['reset']}\n")

    llm = default_llm()
    print(f"{C['dim']}provider = {config.PROVIDER}   classify={config.MODEL_CLASSIFY}"
          f"  generate={config.MODEL_GENERATE}  verify={config.MODEL_VERIFY}"
          f"{C['reset']}\n")
    if not (os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN")
            or os.getenv("OPENROUTER_API_KEY")
            or Path.home().joinpath(".config/anthropic").exists()):
        print(f"{C['yellow']}No Anthropic credentials found. The run will still "
              f"complete -- every message will fail closed and escalate to a "
              f"human, which is the designed behaviour when the model is "
              f"unavailable. Set ANTHROPIC_API_KEY to see it actually "
              f"classify, draft, and book.{C['reset']}\n")
    outcomes = run_pipeline(
        mail=mail, calendar=calendar, store=store, llm=llm,
        contacts_path=Path(args.contacts) if args.contacts
        else config.DATA_DIR / "contacts.json",
        brand=brand, policy=policy, now=now)

    raw_by_id = ({} if args.real
                 else {r["message_id"]: r for r in mail.fetch_replies()})
    hdr = f"{'FROM':<22} {'ROLE':<8} {'INTENT':<17} {'CONF':>5}  {'ACTION':<18} RULE"
    print(f"{C['bold']}{hdr}{C['reset']}")
    print(C["dim"] + "-" * len(hdr) + C["reset"])

    misses = []
    for o in outcomes:
        c, d = o.classification, o.decision
        colour = ACTION_COLOR.get(d.action.value, "")
        name = (o.email.from_name or o.email.from_email)[:21]
        role = c.sender_type.value if c else "-"
        prefiltered = d.rule_id.startswith(("r001", "r002", "r003"))
        intent = c.intent.value if c else (
            "(prefiltered)" if prefiltered else "(unclassified)")
        conf = f"{c.confidence:.2f}" if c else "  - "
        print(f"{name:<22} {role:<8} {intent:<17} {conf:>5}  "
              f"{colour}{d.action.value:<18}{C['reset']} {C['dim']}{d.rule_id}{C['reset']}")
        expected = raw_by_id.get(o.email.message_id, {}).get("_expected_intent")
        if expected and c and c.intent.value != expected:
            misses.append((name, expected, c.intent.value))

    n = len(outcomes)
    by = {}
    for o in outcomes:
        by[o.decision.action.value] = by.get(o.decision.action.value, 0) + 1
    print(f"\n{C['bold']}{n} messages{C['reset']}  " +
          "  ".join(f"{ACTION_COLOR.get(k,'')}{k}={v}{C['reset']}"
                    for k, v in sorted(by.items())))
    booked = [o for o in outcomes if o.meeting]
    if booked:
        print(f"{C['cyan']}{len(booked)} invite(s) created{C['reset']}")
        for o in booked:
            print(f"  {o.email.from_name}: "
                  f"{o.meeting.start.strftime('%a %b %-d, %-I:%M %p %Z')}")
    if misses:
        print(f"\n{C['yellow']}Intent differed from the fixture label on "
              f"{len(misses)}:{C['reset']}")
        for name, exp, got in misses:
            print(f"  {name}: labelled {exp}, classified {got}")
    served = getattr(llm, "models_used", None)
    if served:
        print(f"{C['dim']}models that answered: "
              + ", ".join(f"{m} x{n}" for m, n in sorted(served.items()))
              + C["reset"])
    print(f"\n{C['dim']}tokens in/out: {llm.input_tokens}/{llm.output_tokens}"
          f"   ledger: {DB_PATH}{C['reset']}")
    print(f"{C['dim']}next: python -m cicero.cli review{C['reset']}\n")
    return 0


def _print_item(row, index: int, total: int) -> None:
    payload = json.loads(row["payload"])
    print(f"\n{C['bold']}{'=' * 78}{C['reset']}")
    print(f"{C['bold']}[{index}/{total}] {row['contact_email']}{C['reset']}  "
          f"{C['dim']}{row['received_at']}{C['reset']}")
    print(f"{C['dim']}{'-' * 78}{C['reset']}")

    print(f"{C['bold']}Their message{C['reset']}")
    for line in payload["email"]["body"].splitlines():
        print(f"  {line}")

    print(f"\n{C['bold']}How it was read{C['reset']}")
    if row["intent"]:
        print(f"  {row['sender_type']} / {row['intent']} "
              f"(confidence {row['confidence']:.2f})")
        if payload.get("classification"):
            print(f"  {C['dim']}{payload['classification']['summary']}{C['reset']}")
            print(f"  {C['dim']}{payload['classification']['reasoning']}{C['reset']}")
    else:
        print(f"  {C['dim']}handled before classification{C['reset']}")
    flags = json.loads(row["red_flags"] or "[]")
    if flags:
        print(f"  {C['red']}flags: {', '.join(flags)}{C['reset']}")

    colour = ACTION_COLOR.get(row["action"], "")
    print(f"\n{C['bold']}Why it is here{C['reset']}")
    print(f"  {colour}{row['action']}{C['reset']} "
          f"{C['dim']}({row['rule_id']}){C['reset']}")
    print(f"  {row['reason']}")

    issues = json.loads(row["verdict_issues"] or "[]")
    if issues:
        print(f"\n{C['bold']}Verifier{C['reset']}")
        for i in issues:
            col = C["red"] if i["severity"] == "block" else C["yellow"]
            print(f"  {col}{i['severity']:<5}{C['reset']} {i['code']}: {i['detail']}")

    if row["draft_body"]:
        print(f"\n{C['bold']}Proposed reply{C['reset']}  "
              f"{C['dim']}({row['draft_subject']}){C['reset']}")
        for line in row["draft_body"].splitlines():
            print(f"  {C['green']}| {line}{C['reset']}")
    else:
        print(f"\n{C['dim']}No draft was generated -- this intent is not one a "
              f"machine may answer.{C['reset']}")


def cmd_auth(args) -> int:
    """One-time OAuth, then prove we can actually read mail and the calendar."""
    from .adapters.google import build_services

    print(f"\n{C['bold']}Google authorisation{C['reset']}")
    print(f"{C['dim']}A browser window will open. Approve access for the "
          f"account whose inbox this should read.{C['reset']}\n")
    gmail_svc, cal_svc = build_services()

    profile = gmail_svc.users().getProfile(userId="me").execute()
    print(f"  {C['green']}Gmail   {C['reset']} {profile['emailAddress']}  "
          f"{C['dim']}({profile['messagesTotal']} messages){C['reset']}")

    # Verify with the calls we actually make. `calendars().get()` needs a
    # broader scope than we request, so checking with it would either fail here
    # or tempt us into asking for more access than the job needs.
    from datetime import timedelta
    window_start = datetime.now(timezone.utc)
    fb = cal_svc.freebusy().query(body={
        "timeMin": window_start.isoformat(),
        "timeMax": (window_start + timedelta(days=7)).isoformat(),
        "items": [{"id": "primary"}]}).execute()
    busy = fb["calendars"]["primary"].get("busy", [])
    print(f"  {C['green']}Calendar{C['reset']} freebusy readable  "
          f"{C['dim']}({len(busy)} busy blocks in the next 7 days){C['reset']}")

    print(f"\n{C['dim']}Token cached at .google-token.json. "
          f"Next: python -m cicero.cli run --real{C['reset']}\n")
    return 0


def cmd_review(args) -> int:
    store = Store(DB_PATH)
    rows = store.pending_review()
    if not rows:
        print("Nothing pending. Run `python -m cicero.cli run` first.")
        return 0

    mail = MockGmail(config.DATA_DIR / "inbox.json")
    for n, row in enumerate(rows, 1):
        _print_item(row, n, len(rows))
        if args.list:
            continue
        prompt = (f"\n{C['bold']}[a]pprove & send  [s]kip  [r]eject  "
                  f"[q]uit >{C['reset']} ")
        try:
            choice = input(prompt).strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nstopping.")
            return 0
        if choice == "q":
            return 0
        if choice == "a":
            if not row["draft_body"]:
                print(f"  {C['red']}nothing to send -- write this one by hand"
                      f"{C['reset']}")
                continue
            mail.send_reply(thread_id=row["thread_id"],
                            in_reply_to=row["message_id"],
                            to=row["contact_email"],
                            subject=row["draft_subject"],
                            body=row["draft_body"])
            store.set_review_status(row["message_id"], "approved", sent=True)
            print(f"  {C['green']}sent{C['reset']}")
        elif choice == "r":
            store.set_review_status(row["message_id"], "rejected", sent=False)
            print(f"  {C['red']}rejected{C['reset']}")
        else:
            print(f"  {C['dim']}skipped{C['reset']}")
    return 0


def cmd_ledger(args) -> int:
    store = Store(DB_PATH)
    rows = store.all_outcomes()
    if not rows:
        print("Ledger is empty.")
        return 0
    hdr = f"{'RECEIVED':<20} {'FROM':<30} {'ACTION':<18} {'RULE':<28} SENT"
    print(f"{C['bold']}{hdr}{C['reset']}")
    print(C["dim"] + "-" * len(hdr) + C["reset"])
    for r in rows:
        col = ACTION_COLOR.get(r["action"], "")
        print(f"{r['received_at'][:19]:<20} {r['contact_email'][:29]:<30} "
              f"{col}{r['action']:<18}{C['reset']} {r['rule_id']:<28} "
              f"{'yes' if r['sent'] else '-'}")
    return 0


def cmd_show(args) -> int:
    store = Store(DB_PATH)
    for r in store.all_outcomes():
        if args.message_id in r["message_id"]:
            print(json.dumps(json.loads(r["payload"]), indent=2))
            return 0
    print(f"No ledger entry matching {args.message_id!r}")
    return 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="cicero", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("auth", help="run the Google OAuth flow and verify access")
    a.set_defaults(func=cmd_auth, live=False, now=None, real=True,
                   query="in:inbox", contacts=None)

    r = sub.add_parser("run", help="process the inbox")
    r.add_argument("--real", action="store_true",
                   help="use the real Gmail + Google Calendar APIs")
    r.add_argument("--query", default="in:inbox is:unread -from:me",
                   help="Gmail search query for --real")
    r.add_argument("--contacts", default=None,
                   help="CRM file to use (default data/contacts.json)")
    r.add_argument("--live", action="store_true",
                   help="actually send and book (default is a dry run)")
    r.add_argument("--now", help="override 'now' (ISO 8601) for reproducible runs")
    r.set_defaults(func=cmd_run)

    v = sub.add_parser("review", help="work the human review queue")
    v.add_argument("--list", action="store_true", help="print only, do not prompt")
    v.set_defaults(func=cmd_review, live=False, now=None, real=False, query="", contacts=None)

    l = sub.add_parser("ledger", help="every decision this system has made")
    l.set_defaults(func=cmd_ledger, live=False, now=None, real=False, query="", contacts=None)

    s = sub.add_parser("show", help="full JSON for one message")
    s.add_argument("message_id")
    s.set_defaults(func=cmd_show, live=False, now=None, real=False, query="", contacts=None)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
