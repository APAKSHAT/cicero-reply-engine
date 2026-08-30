"""Loads config/*.yaml once and exposes it as plain dicts."""

from __future__ import annotations

import os
from functools import lru_cache

import yaml

from ._env import ROOT

CONFIG_DIR = ROOT / "config"
DATA_DIR = ROOT / "data"



# Which provider answers. OpenRouter is used when its key is present, so the
# same codebase runs against either without a flag.
PROVIDER = "openrouter" if os.getenv("OPENROUTER_API_KEY") else "anthropic"

# Per-stage model defaults. Classification and verification are
# schema-constrained and high-volume; drafting is the one stage that has to
# sound human, so it is the one worth pinning to a strong model rather than
# leaving to a router optimising for cost.
_DEFAULTS = {
    "anthropic": {"classify": "claude-opus-5",
                  "generate": "claude-opus-5",
                  "verify": "claude-opus-5"},
    # OpenRouter's autorouter picks a model per call, optimising for cost.
    # Caveat worth knowing: on the first live run it sent the DRAFTING stage to
    # a cheap model, which ignored a founder's stated availability and invented
    # a meeting time. Classification and verification tolerate this; drafting is
    # the one stage whose output a human reads. Pin it when there is credit:
    #   CICERO_MODEL_GENERATE=anthropic/claude-sonnet-4.5
    "openrouter": {"classify": "openrouter/auto",
                   "generate": "openrouter/auto",
                   "verify": "openrouter/auto"},
}[PROVIDER]

MODEL_CLASSIFY = os.getenv("CICERO_MODEL_CLASSIFY", _DEFAULTS["classify"])
MODEL_GENERATE = os.getenv("CICERO_MODEL_GENERATE", _DEFAULTS["generate"])
MODEL_VERIFY = os.getenv("CICERO_MODEL_VERIFY", _DEFAULTS["verify"])


@lru_cache(maxsize=None)
def _load(name: str) -> dict:
    with open(CONFIG_DIR / name) as f:
        return yaml.safe_load(f)


def policy() -> dict:
    p = _load("policy.yaml")
    # Env var can force dry_run on, never off -- safe direction only.
    if os.getenv("CICERO_DRY_RUN", "").lower() in {"1", "true", "yes"}:
        p = {**p, "dry_run": True}
    return p


def brand() -> dict:
    return _load("brand.yaml")
