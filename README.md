# Cicero reply engine

Handles replies to acquisition outreach: works out who wrote in and what they
want, drafts a reply in the firm's voice, and books a call when — and only when —
the person actually asked for one.

The organising idea is that **the model proposes and deterministic code
disposes.** An LLM reads the email and writes the draft. Nothing an LLM produces
decides whether that draft is sent. That is decided by ordered, named rules in
`config/policy.yaml`, and every row in the ledger carries the id of the rule that
produced it, so any outcome can be traced to a line of config rather than to a
prompt.

## Run it

```bash
make install
export ANTHROPIC_API_KEY=sk-ant-...     # or OPENROUTER_API_KEY
make test                               # 158 tests, no API key needed
make demo                               # full pipeline on the sample inbox
make review                             # work the human review queue
make ledger                             # every decision the system has made
```

`make demo` is a dry run: it classifies, drafts, verifies and picks times, then
stops before sending or booking. `make live` does the same against the mock
adapters with sending switched on.

Against real accounts:

```bash
python -m cicero.cli auth               # one-time Google OAuth
python -m cicero.cli run --real         # dry run over a real Gmail inbox
python -m cicero.cli run --real --live  # sends and books for real
```

## Flow

```
normalize ─→ dedupe ─→ prefilter ─→ classify ─→ DECIDE ─→ [schedule] ─→
                                                   │
generate ─→ verify ─→ book ─→ send                 └─→ escalate / suppress
```

Two things about that ordering are load-bearing. `DECIDE` runs before anything is
written, so no draft exists for a message the policy layer already refused —
there is nothing for a tired reviewer to rubber-stamp. And the calendar write
happens *after* verification and immediately before the send, so a draft that
fails its checks can never leave a real invite on a real person's calendar.

## Mocks and real accounts

Both external systems sit behind adapter interfaces
(`src/cicero/adapters/base.py`), so the pipeline never learns which
implementation it has. The sample inbox is mocked, as the brief allows; `--real`
swaps in Gmail and Google Calendar with no other change.

| | Default | `--real` |
|---|---|---|
| Mail | `MockGmail` — reads `data/inbox.json` | Gmail API, threaded via `In-Reply-To`/`References` |
| Calendar | `MockCalendar` — pre-seeded with a working week | Google Calendar — freebusy + `events.insert` |

OAuth scopes are the narrowest that do the job: `gmail.modify` (not
`mail.google.com`, which grants permanent delete), `calendar.events` (not full
`calendar`, which can delete calendars), and `calendar.freebusy` — busy blocks
only, never the content of anyone's meetings.

Two provider backends sit behind one seam in `llm.py`: Anthropic
(`ANTHROPIC_API_KEY`) and any OpenAI-compatible endpoint via OpenRouter
(`OPENROUTER_API_KEY`, which wins if both are set). Models are set per stage —
`CICERO_MODEL_CLASSIFY`, `CICERO_MODEL_GENERATE`, `CICERO_MODEL_VERIFY` — because
classification is schema-constrained and cheap while drafting is the one stage
whose output a person reads.

## Layout

```
config/policy.yaml   every threshold and gate. Data, not code, so it can be
                     tightened without a deploy and read by a non-engineer.
config/brand.yaml    voice, the closed set of facts the model may assert, the
                     never-say list, and regex tripwires.
data/inbox.json      20 sample replies, including adversarial ones
data/contacts.json   CRM fixture

src/cicero/
  ingest.py          quote/signature stripping, deterministic pre-filters
  classify.py        intent and role, with confidence discounted for weak
                     evidence rather than taken from the model
  policy.py          the decision state machine. No LLM touches this file.
  scheduling.py      book vs propose, slot rules, double-booking prevention
  generate.py        the four-layer drafting prompt
  verify.py          regex tripwires plus a critic pass. Fails closed.
  redact.py          reversible pseudonymization at the LLM boundary
  store.py           SQLite ledger: idempotency and accountability
  pipeline.py        orchestration
  cli.py             auth / run / review / ledger / show

tools/seed_inbox.py  inserts test replies into a real mailbox via
                     messages.insert, so nothing is sent to anyone

tests/               158 tests, no network or API key required
```

## State of the prototype

**Runs end to end against real Google APIs.** Verified live: a reply delivered
into the Gmail thread it answered; a calendar invite created inside the sender's
stated hours with a Meet link; and, after clearing the ledger and re-running the
whole booking, the *same* event id returned rather than a duplicate — a retry
cannot double-book someone.

**Guardrails hold under adversarial input.** Prompt-injection attempts and price
demands escalate without a draft being written. On one live run over a real
inbox, 50 unrelated messages (newsletters, notifications, auto-replies) were
suppressed by the deterministic pre-filters using two model calls in total.

**Accuracy is unmeasured.** 20 self-written fixtures cannot support an accuracy
claim. `WRITEUP.md` §1 sets out what would be needed and where classification is
known to misfire.

**Reply quality tracks model quality.** Drafts written by small models were
noticeably worse and were caught by the verifier rather than sent.
`CICERO_MODEL_GENERATE` is the dial that matters.

**The numbers in `config/brand.yaml` are placeholders** and must be replaced
before this points at a real inbox.

The reasoning behind the design, and the six questions from the brief, are in
[WRITEUP.md](WRITEUP.md).

---

Written as a take-home assignment and shared for evaluation only. See
[LICENSE](LICENSE) — all rights reserved.
