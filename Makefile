VENV ?= .venv
PY   := $(VENV)/bin/python
export PYTHONPATH := src

.PHONY: install install-google demo live review ledger test clean

install:
	python3 -m venv $(VENV)
	$(VENV)/bin/pip install -q --upgrade pip
	$(VENV)/bin/pip install -q -e ".[dev]"

## Adds the Gmail and Google Calendar clients, for `cicero run --real`.
install-google:
	$(VENV)/bin/pip install -q -e ".[google,dev]"

## Full pipeline over the sample inbox. Nothing is sent or booked.
demo:
	$(PY) -m cicero.cli run

## Same, but actually sends (to the mock Gmail) and books (on the mock calendar).
live:
	$(PY) -m cicero.cli run --live

review:
	$(PY) -m cicero.cli review

ledger:
	$(PY) -m cicero.cli ledger

## Guardrail tests. No API key needed -- these cover the deterministic layers.
test:
	$(PY) -m pytest -q

clean:
	rm -f cicero.db
