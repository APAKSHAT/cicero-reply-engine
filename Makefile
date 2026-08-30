VENV ?= .venv
PY   := $(VENV)/bin/python
export PYTHONPATH := src

.PHONY: install demo live review ledger test clean

install:
	python3 -m venv $(VENV)
	$(VENV)/bin/pip install -q -r requirements.txt

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
