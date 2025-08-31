PY=python3
VENV=.venv
ACT=. $(VENV)/bin/activate

.PHONY: dev run lint fmt test clean
dev:
	$(PY) -m venv $(VENV); \
	$(ACT); pip install --upgrade pip; \
	pip install -r backend/requirements.txt -r backend/requirements-dev.txt || true; \
	pre-commit install

run:
	$(ACT); uvicorn backend.app:app --reload --port $${PORT:-8000}

lint:
	$(ACT); ruff check backend && black --check backend

fmt:
	$(ACT); ruff check --fix backend; black backend

test:
	$(ACT); pytest -q

clean:
	rm -rf $(VENV) .pytest_cache
