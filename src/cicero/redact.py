"""Reversible pseudonymization at the LLM boundary.

Direct identifiers are swapped for stable tokens before the text leaves the
process, and swapped back in the generated draft. The model still sees enough
to write a good reply -- the business context, the tone, the questions -- but
the payload that reaches a third party is stripped of the things that are
useful to whoever might end up holding a log of it.

Names are handled separately (`redact_names`): the model needs a first name to
open a letter, so by default we keep it and note the tradeoff, but the plumbing
to tokenize names is here because some counterparties will require it.
"""

from __future__ import annotations

import re

_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    ("PHONE", re.compile(
        r"(?:\+?1[\s.-]?)?(?:\(\d{3}\)|\d{3})[\s.-]?\d{3}[\s.-]?\d{4}\b")),
    ("URL", re.compile(r"\bhttps?://\S+")),
    ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("EIN", re.compile(r"\b\d{2}-\d{7}\b")),
    ("ACCOUNT", re.compile(r"\b\d{9,18}\b")),
]


class Redactor:
    """Stable within one email, fresh for the next. Tokens are not global IDs,
    so a leaked log cannot be joined across contacts."""

    def __init__(self, redact_names: bool = False):
        self.redact_names = redact_names
        self._fwd: dict[str, str] = {}   # original -> token
        self._rev: dict[str, str] = {}   # token -> original
        self._n: dict[str, int] = {}

    def _token(self, kind: str, value: str) -> str:
        if value in self._fwd:
            return self._fwd[value]
        self._n[kind] = self._n.get(kind, 0) + 1
        tok = f"<{kind}_{self._n[kind]}>"
        self._fwd[value] = tok
        self._rev[tok] = value
        return tok

    def scrub(self, text: str, names: list[str] | None = None) -> str:
        if not text:
            return text
        out = text
        if self.redact_names:
            for name in sorted(names or [], key=len, reverse=True):
                for part in [name] + name.split():
                    if len(part) > 2:
                        out = re.sub(rf"\b{re.escape(part)}\b",
                                     self._token("NAME", part), out)
        for kind, pat in _PATTERNS:
            out = pat.sub(lambda m: self._token(kind, m.group(0)), out)
        return out

    def restore(self, text: str) -> str:
        out = text
        for tok, original in self._rev.items():
            out = out.replace(tok, original)
        return out

    @property
    def token_count(self) -> int:
        return len(self._rev)
