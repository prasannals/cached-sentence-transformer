.PHONY: help install lint test-unit test-integration test-all build distcheck publish publish-test clean

PYTHON ?= python
PIP ?= $(PYTHON) -m pip
PUBLISH_REPOSITORY_URL ?= https://upload.pypi.org/legacy/
PUBLISH_TEST_REPOSITORY_URL ?= https://test.pypi.org/legacy/

help:
	@echo "Targets:"
	@echo "  install           Install package in editable mode with dev deps"
	@echo "  lint              Run ruff"
	@echo "  test-unit         Run unit tests with coverage (skips integration)"
	@echo "  test-integration  Run integration tests with coverage (requires Postgres; set PG_DSN)"
	@echo "  test-all          Run unit + integration tests with coverage"
	@echo "  build             Build sdist/wheel"
	@echo "  distcheck         Validate dist metadata (twine check)"
	@echo "  publish           Build + upload to PyPI (requires TWINE_USERNAME/TWINE_PASSWORD + CONFIRM_PUBLISH=1)"
	@echo "  publish-test      Build + upload to TestPyPI (requires TWINE_USERNAME/TWINE_PASSWORD + CONFIRM_PUBLISH=1)"
	@echo "  clean             Remove build artifacts"
	@echo ""
	@echo "Integration env:"
	@echo "  RUN_PG_INTEGRATION=1 PG_DSN='host=... port=... dbname=... user=... password=...'"
	@echo ""
	@echo "Publish env:"
	@echo "  CONFIRM_PUBLISH=1 TWINE_USERNAME=__token__ TWINE_PASSWORD='pypi-...'"
	@echo "  Optional override: PUBLISH_REPOSITORY_URL=$(PUBLISH_REPOSITORY_URL)"
	@echo "  Optional override: PUBLISH_TEST_REPOSITORY_URL=$(PUBLISH_TEST_REPOSITORY_URL)"

install:
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"

lint:
	ruff check .

test-unit:
	pytest -m "not integration" --cov=cached_sentence_transformer --cov-report=term-missing --cov-fail-under=90

test-integration:
	@test -n "$$PG_DSN" || (echo "PG_DSN must be set for integration tests. Example: PG_DSN='host=localhost port=5432 dbname=postgres user=postgres password=postgres'"; exit 2)
	RUN_PG_INTEGRATION=1 pytest -m integration --cov=cached_sentence_transformer --cov-report=term-missing

test-all:
	@test -n "$$PG_DSN" || (echo "PG_DSN must be set for integration tests. Example: PG_DSN='host=localhost port=5432 dbname=postgres user=postgres password=postgres'"; exit 2)
	RUN_PG_INTEGRATION=1 pytest --cov=cached_sentence_transformer --cov-report=term-missing --cov-fail-under=90

build:
	$(PIP) install --upgrade build
	$(PYTHON) -m build

distcheck:
	$(PIP) install --upgrade twine
	twine check dist/*

publish:
	@test "$$CONFIRM_PUBLISH" = "1" || (echo "Refusing to publish: set CONFIRM_PUBLISH=1"; exit 2)
	@test -n "$$TWINE_USERNAME" || (echo "Missing TWINE_USERNAME (use __token__ for API tokens)"; exit 2)
	@test -n "$$TWINE_PASSWORD" || (echo "Missing TWINE_PASSWORD (paste your PyPI API token here)"; exit 2)
	$(PIP) install --upgrade build twine
	rm -rf dist
	$(PYTHON) -m build
	twine check dist/*
	twine upload --repository-url "$(PUBLISH_REPOSITORY_URL)" dist/*

publish-test:
	@test "$$CONFIRM_PUBLISH" = "1" || (echo "Refusing to publish: set CONFIRM_PUBLISH=1"; exit 2)
	@test -n "$$TWINE_USERNAME" || (echo "Missing TWINE_USERNAME (use __token__ for API tokens)"; exit 2)
	@test -n "$$TWINE_PASSWORD" || (echo "Missing TWINE_PASSWORD (paste your TestPyPI API token here)"; exit 2)
	$(PIP) install --upgrade build twine
	rm -rf dist
	$(PYTHON) -m build
	twine check dist/*
	twine upload --repository-url "$(PUBLISH_TEST_REPOSITORY_URL)" dist/*

clean:
	rm -rf dist build .pytest_cache .ruff_cache .mypy_cache *.egg-info .coverage coverage.xml


