# Cicero reply engine — design notes

The system rests on one line: **the model proposes, deterministic code disposes.**
An LLM reads each reply and writes each draft, because those are language
problems. Nothing an LLM produces decides whether a draft is *sent*. That is
decided by ordered, named rules over a small set of facts, and every outcome in
the ledger carries the id of the rule that produced it — so "why did we send
that" is answered by a row pointing at a line of YAML, not by re-running a
non-deterministic model.

*(Detail beyond what fits here is in the module docstrings, which is where I put
the reasoning that a reviewer reading the code would want next to the code.)*

## 1. Intent classification

Three layers, cheapest first. **Deterministic pre-filters** handle autoresponders,
bounces, out-of-office, and opt-out language — ~15% of a real reply inbox, and no
model ever sees them. "Thanks, but take me off your list" is a hard stop even
though it's friendly; that is far too important to leave to a probabilistic
reader.

**Identity comes from the CRM, not the model.** We sent the original email, so we
know who we wrote to. The model is asked for the role anyway — not to decide it,
but so a *disagreement* with the CRM becomes a signal rather than passing
silently. That's the founder-forwards-to-their-broker case. Beyond the CRM, the
prompt separates the populations behaviourally: founders write in the first
person about "my company" and "my people"; brokers write about "my client", "the
teaser", qualify buyers before sharing anything, and run processes with
deadlines.

**The LLM classifies intent** into 12 enum values under a constrained schema, so
it physically cannot return one outside the set, and it never proposes an
*action* — only a reading. The confusions that actually happen are disambiguated
as explicit pairs: `schedule_request` needs an affirmative ask, warmth alone is
`interested`; `deferred` needs a future re-open where `not_interested` has none;
`question` is answerable from what we already say publicly where `needs_info`
requires us to produce or commit to something.

**Confidence is not taken at face value.** Self-reported LLM confidence skews
high. Two corrections: the model must quote a **verbatim span** from the message
supporting its intent and role, and we check that span really occurs in the body
— a model that can't point at the words it reasoned from is inventing (−0.25).
Then deterministic penalties: −0.20 under eight words, −0.10 uncertain sender,
−0.15 role inferred without a quote. The number the policy layer gates on is
ours, not the model's. Below 0.55 we don't even draft; below 0.85 we draft but
never send.

**Where it misfires.** *Quoted-thread contamination* is the worst: our own
outreach copy sits below the reply full of scheduling language, and a classifier
that sees it decides they asked for a call when they wrote "no thanks". I strip
six quote-marker patterns, but interleaved top-posting still defeats it.
*Multi-intent collapses to one label* — "Not selling, but what's your typical
structure?" is both `not_interested` and `negotiation`; the red-flag layer
catches this example, which is a patch, not a fix. *Indirect declines read as
interest* — "certainly something to think about at some stage" is a no, and
models are agreeable readers; the confidence penalties don't help because the
message is long and the quote is real. Also: the `deferred`/`not_interested`
boundary with no date named, politeness masking hostility, and the signature
heuristic guessing when there's no `--` delimiter.

**What the live runs actually showed.** Running the real pipeline over the 17
fixtures surfaced things I had guessed at and one I hadn't:

- The model's **self-reported confidence is near-useless**: a small model
  returned 0.90 for almost every message regardless of difficulty. The only
  scores that discriminated were the ones the *deterministic penalties* pushed
  down — an unknown sender with a weak quote landed at 0.55, a two-word message
  at 0.50. The penalty layer is doing all of the calibration work, which is an
  argument for keeping it and against ever gating on a raw model score.
- **Two live misfires**, both on the boundaries I flagged: "Ray forwarded me
  your email, I'm the CFO" was read as `question` rather than `referral`, and
  the two-word "what is this" as `question` rather than `unclear`. Neither
  reached a recipient — both were escalated by red flags and confidence floors
  anyway. That is defence in depth doing its job, not the classifier being
  right.
- A weaker model classified a hard no ("we're not selling, my daughter takes
  over next year") as `deferred` with 0.99 confidence. Raising the token ceiling
  and pinning a better classifier fixed it. Confident and wrong is the failure
  mode to design around.

**I still have not measured accuracy.** The fixtures carry labels and `make demo`
prints divergence, but that's a smoke check — 20 examples I wrote myself against
a prompt I wrote myself would produce a flattering, meaningless number. A real
claim needs a few hundred human-labelled production replies, two labellers, an
inter-rater score, held out from prompt development. That's also the only honest
way to set the thresholds, which are currently reasoned defaults.

## 2. Reply quality and tone control

Four prompt layers, in order. **Voice** from `config/brand.yaml`, including a
banned-word list. **Closed-world facts** — the complete set of claims the model
may make about us, plus an explicit *escape hatch*: "acknowledge it and say a
colleague will follow up", framed as a good answer rather than an evasion. Models
overcommit when refusing has no sanctioned form; giving refusal a script beats
forbidding invention. **Recipient conditioning**, where most of the quality lives:
for a founder, lead by acknowledging what they actually said, avoid transaction
vocabulary unless they used it first, take the question about their people
seriously because it's usually the real question, and never imply urgency — they
aren't on a clock and pretending they are reads as a tactic. For a broker, lead
with the answer, address qualifying questions in the order asked, and if they
want a document say a colleague will send it without describing its contents.
**One job per intent** — replies get vague when the model decides what the email
is for. Three full **exemplars** carry tone; "warm but direct" in a prompt does
almost nothing next to a paragraph that is.

**Two independent gates before sending.** *Deterministic checks*: banned patterns
(guarantees, valuation multiples, "attached is", `[Name]` placeholders),
word-count bounds, leftover redaction tokens, missing sign-off, any URL or address
we didn't supply, and **any number appearing in neither the approved facts nor
their email** — that's the one that catches a confident, well-written, entirely
fabricated revenue figure. Plus a scheduling-consistency check in both
directions. *A critic pass* catches what regexes can't — tone that's off, an
answer that overcommits without a banned word. Five rubric axes plus
`would_send_as_is`; below 4 blocks. It gets the approved facts but **not** the
drafting prompt: it should judge the email as a recipient would, not check the
writer's homework. If the critic errors, that's a block — it fails closed.

**The critic is not reliable enough to be a sole gate, and I have the evidence.**
On one live run it caught a draft offering 9:00 AM to a founder who wrote "after
1pm, not mornings" and scored it 1/5. An hour later, on the same class of error,
it passed the draft and the email went out. Same rule, same system, opposite
verdict — that is what a non-deterministic judge is. The lesson I took is not
"write a better critic prompt": it is that **any rule you can state precisely
does not belong in a model's judgement.** Respecting stated availability is such
a rule, so it moved into `scheduling.py` as a hard stop — if we have nothing
free that honours what the sender said, we hand the thread to a person rather
than propose times they already ruled out. That stop is verified against the
live calendar rather than only in tests: blocking out both days the sender named
and re-running produced `r201_no_slot_available`, no email, and no invite, where
the same input had previously produced an offer of 9:00 AM. The critic stays, but as a net for
the fuzzy things regexes cannot express, never as the only thing between a draft
and a send.

**A failure never rewrites the draft**, it demotes to human review. An automatic
repair loop is a second chance to make the same class of mistake with the
evidence discarded.

**This layer earned its place on the first live run.** Of 17 messages then in the
fixture, six drafts
were blocked, and every block was correct: a reply that **invented a colleague**
("my colleague Brian will be in touch"); two replies that **invented meeting
times** on threads where nothing had been checked against a calendar; one that
said "I will send an invite shortly" when the pipeline was in propose-mode and no
invite existed; and one that **promised a proof-of-funds letter** we had not
agreed to send. The invented-time check did not exist before that run — the first
live pass is what revealed the gap, and it is now a deterministic rule with
tests.

## 3. Scheduling trigger and logic

Consent is defined narrowly, in one function, and has nothing to do with
availability: an affirmative ask for a call, or acceptance of a slot we offered.
**Enthusiasm is not consent** — "sounds interesting, tell me more" gets a reply,
not an invite.

Even with consent we only *book* when a time is resolvable. Accepted one of our
slots → book it, after re-checking it's still free. Named their own availability
→ resolve, then book inside it. Said yes but named no time → **propose three and
book nothing.** Consent to a conversation isn't consent to a specific hour.

Turning "Tuesday or Wednesday afternoon after 1pm central" into a datetime is a
language problem, so a structured LLM call does it — but the model interprets
language, it doesn't authorise a calendar write. Every window comes back through
our validation (parseable, timezone-aware, not inverted, not in the past). If
anything fails, we fall back to proposing. Guessing books a real meeting on a real
calendar at the wrong time, so refusing is the better failure.

A candidate slot must simultaneously clear: 24-hour notice, a 10-day horizon, a
working day, the *contact's* working hours in *their* timezone, the organiser's in
theirs, a 15-minute buffer around existing events, and a four-per-day cap. The
two-timezone check matters — 9am in New York is 6am in LA. (A unit test caught a
real bug here: the original check compared hours rather than instants, allowing a
call starting 17:45 to end after the day did.)

**Double-booking is prevented at the API boundary**, not in application logic: the
Google event id is `sha256(thread:contact)` in base32hex, so a duplicate insert
returns 409. This is verified against the live API, not just reasoned about — I
cleared the ledger so deduplication could not help, re-ran the entire booking,
and Google returned the same event id; the calendar still held one event. With a unique `message_id` in the ledger, the pipeline is safe to
retry — a "did we already do this?" lookup can race with itself, a deterministic
key cannot. Ordering is generate → verify → **book** → send, so a draft that fails
checks never leaves an orphan invite; and if the calendar write fails after the
draft passed, the email is held rather than sent.

## 4. Guardrails and failure handling

A message reaches `AUTO_SEND` only by falling through every gate: opted out →
suppress; autoresponder/bounce/opt-out → suppress (opt-out *with a legal
reference* → escalate); classifier failed → escalate; `negotiation`, `hostile`,
`referral`, `unclear` → escalate; red flag (price, legal, NDA/terms, competing
process, upset sender, identity mismatch) → escalate; confidence <0.55 →
escalate; unknown sender, colleague already on the thread, ≥2 consecutive auto
replies, confidence <0.85, failed verification, run cap of 25 → draft and hold.

**The line follows cost of being wrong, not model confidence.** Price and terms
are excluded regardless of confidence because a casual number is the one thing
here that creates a real obligation, and the model is most fluent exactly where it
should be most cautious. `unclear` is excluded because a plausible reply to a
message we didn't understand is worse than a slow one.

Two rules worth calling out. **Escalations produce no draft** — a plausible draft
next to a hard case is precisely how a reviewer approves the message the guardrail
existed to prevent. And the gate is on **unknown senders, not first replies**: "a
human reads the first reply to everyone" sounds safe and would make the system
pointless, since for contacts we chose to mail an automated first reply is the
entire point.

Every failure has an explicit destination and the default direction is toward a
human: classifier fails → escalate, never guess; generation fails → escalate;
critic unavailable → block, since an unavailable check isn't a pass; calendar
fails → hold; unhandled exception → that message escalates and the run continues,
so one bad email can't take down the batch or get retried into a duplicate send.
`dry_run: true` is the default and `CICERO_DRY_RUN` can only force it *on* —
there is deliberately no variable that turns sending on. The run cap is a circuit
breaker: if the classifier is wrong this morning it'll be wrong 25 times, not 400.

**Prompt injection, and the finding I did not expect.** Every inbound email is
untrusted text going straight into a model, so I fired three attacks at the live
pipeline from *known CRM contacts* (the full auto-send path): a direct override
demanding confirmation of a $14M binding offer, an exfiltration request asking
for the approved facts and never-say list verbatim, and a buried override
demanding an unwanted invite plus a fabricated signed NDA and a $250,000
deposit.

All three were stopped. But the interesting part is *how*. The writer held the
line and refused to disclose anything — and then **the critic read the same
injected instruction, believed it, and marked the refusal down** for "failing to
answer the message", scoring it 2/5. The block was therefore safe by accident:
had the draft complied and leaked, the critic would have scored it well and
approved it. **My last line of defence was itself an injection surface.**

Four changes came out of that. The critic and the writer are both now told
explicitly that the message is untrusted data and that refusing an embedded
instruction is *correct*, never a failure to answer. Two deterministic checks
were added that cannot be argued with — `prompt_leak` (code fences, "my
instructions", "never-say list") and `fabricated_commitment` (claimed NDAs,
wired deposits, exclusivity, binding offers). And the ledger was recording only
the model's red flags, so an escalation fired by a *static* regex flag showed a
reviewer an empty list — the row could not explain its own decision. It now
records every flag weighed.

The deterministic half is covered by tests, including one asserting that a
correct refusal passes cleanly: a check that punished refusals would push the
model toward complying. The prompt-level hardening has not been re-validated
against a live model — the free-tier quota ran out — so I am claiming the
regexes, not the prompts.

**A note on failing closed.** Across these runs the infrastructure failed three
different ways — no credentials, exhausted credits, and a rate limit. Every time,
every affected message escalated to a human and nothing was sent. That was
designed, but it is worth more having watched it happen than having asserted it.

Before real volume I'd add a daily global send cap with alerting, a weekly random
sample of *sent* replies routed to a human (the gates only inspect what they
already suspect), and a short send delay with a recall window.

## 5. Architecture and reasoning

`normalize → dedupe → prefilter → classify → POLICY → [schedule] → generate →
verify → book → send`, with a SQLite ledger and review queue alongside.

**Python, plain modules, no workflow framework.** The whole value is the policy
layer being auditable, unit-testable, and readable by someone who doesn't write
code — ~150 lines of ordered rules with a truth table against them. Each stage
is a function that takes facts and returns either a replacement decision or
nothing, so `process_one` reads as the flow itself and every stage is testable
alone. 158 tests run with no API key, the pipeline ones against a scripted
model. As a DAG in
n8n the one part that must be inspectable becomes the least inspectable, and
diffing it becomes impossible.

**Two providers behind one seam.** `llm.py` exposes exactly two methods
(`structured`, `text`), so the Anthropic SDK and an OpenAI-compatible backend
(OpenRouter) are interchangeable and nothing downstream changes. Structured
output via Pydantic means an out-of-enum intent is not a possible bug either way;
on the OpenAI-compatible path the schema is narrowed to the strict subset those
APIs accept and the response is re-validated locally, so a constraint the wire
schema cannot express still fails closed.

**The stage split is visible on purpose.** Classification and verification are
schema-constrained and high-volume; drafting has to sound human. Different
cost/quality problems, so the three models are separate env vars rather than one
buried constant. This stopped being theoretical on the first live run: pointing
all three at OpenRouter's *autorouter* sent the drafting stage to a cheap model
optimised for cost, and the replies it produced ignored a founder's stated
availability and invented a meeting time. Routing on price is correct for
classification and wrong for the one stage whose output a human reads.

The deeper objection to a router is **variance, not quality**. Across two
identical autorouter runs the same inbox produced different outcomes, because
different models answered: one run resolved "Tuesday or Wednesday afternoon
after 1pm central" into a correct 1:30pm Tuesday booking, the other ran a
reasoning model out of tokens mid-draft. For a system whose entire premise is
that a decision can be explained after the fact, "the router chose differently
today" is not an acceptable entry in the ledger. Classification and verification
tolerate this — they are schema-constrained and their failures fail closed.
Drafting does not. `CICERO_MODEL_GENERATE` is the one dial that matters.

**Adapters** for Gmail and Calendar, so the pipeline never learns which
implementation it has. The mock calendar is pre-seeded with a realistic week — an
empty one would let a broken availability check pass.

**Deliberately left out:** Gmail push via Pub/Sub (polling instead), a real queue
with backoff, multi-turn thread memory beyond the CRM row (each reply is judged on
its own message, which will misread the third message of a long negotiation),
attachments and HTML mail, auth/multi-tenancy, a web review UI, and threshold
calibration.

**At hundreds a day**, four things change. *Ingestion* moves to push with a worker
queue and a per-contact lock — two replies landing together on one thread is a
live double-send risk the message-id key alone doesn't solve. *Cost* is ~3–4 calls
per reply, an estimated $0.05–0.08 each on Opus-tier, so ~$20/day at 300
(estimated from token counts, not measured); classification would move to Haiku
via the Batch API. *Review becomes the bottleneck long before the pipeline does* —
at ~35% held that's 100 items a day, needing prioritisation by deal value and bulk
approval, and at that point the escalate-without-a-draft rule needs revisiting
because reviewer fatigue overtakes rubber-stamping as the bigger risk. *Quality*
needs the labelled eval set from §1 running in CI, so a prompt change that shifts
the `deferred`/`not_interested` boundary is caught before it reaches an inbox.

## 6. Data handling

**The inbound email is the untrusted boundary, in both directions.** Data
handling here is not only about what leaves us — it is about what an outside
party can make the model do. See §4: an email is capable of carrying
instructions, and the defence is that the sender's text is framed as data to
every model that sees it, with deterministic checks behind that in case the
framing fails.

**Reversible pseudonymization at the boundary.** Emails, phones, URLs, and
account/EIN/SSN-shaped numbers become stable tokens (`<EMAIL_1>`) before text
leaves the process, restored in the draft afterwards. Tokens are per-email, so a
leaked log can't be joined across contacts.

**Names are the deliberate exception** — the model needs a first name to open a
letter and the company name to sound like it read the message. I'd rather state
that plainly than claim protection the system doesn't have. `Redactor(redact_names=True)`
exists for a counterparty who requires it, at a real cost to reply quality.

**Never reaches the model:** attachments, the quoted thread history (stripped at
ingest, so our own prior copy is never re-sent), and any CRM field not explicitly
selected into the prompt.

**Storage** is local SQLite, no third-party datastore. The ledger *does* hold the
message body and draft in plaintext, because a reviewer can't review a redacted
email — a conscious trade whose mitigation is access control, not obscurity. In
production: encrypted volume, access logging, and retention that drops bodies once
a deal closes or goes cold. Anthropic doesn't train on API traffic by default; for
deal correspondence I'd also configure the org for zero data retention before this
touches a live inbox. Keys come from the environment and never enter the ledger.

**Google scopes** are the narrowest that work: `gmail.modify` not
`mail.google.com` (which includes permanent delete), `calendar.events` not
`calendar` (which can delete calendars).

**Still open:** contact records have no deletion path, so a GDPR/CCPA erasure
request is currently a manual SQL statement; and suppression lives on the contact
rather than in a durable list that survives a CRM re-import — re-contacting
someone who opted out is both the legal and the reputational exposure.
