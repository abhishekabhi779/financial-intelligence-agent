# Financial Intelligence Agent — Makefile
# Common development tasks

.PHONY: help install install-dev test lint type-check format clean build run dev notebook docs

# Default target
help:
	@echo "Financial Intelligence Agent — Development Commands"
	@echo ""
	@echo "Setup:"
	@echo "  install        Install production dependencies"
	@echo "  install-dev    Install all dependencies (including dev)"
	@echo ""
	@echo "Development:"
	@echo "  dev            Start development server with hot reload"
	@echo "  notebook       Start Jupyter Lab"
	@echo "  run            Run CLI research command"
	@echo ""
	@echo "Quality:"
	@echo "  test           Run tests with pytest"
	@echo "  test-cov       Run tests with coverage"
	@echo "  lint           Run ruff linter"
	@echo "  format         Format code with ruff"
	@echo "  type-check     Run mypy type checking"
	@echo "  quality        Run all quality checks"
	@echo ""
	@echo "Docker:"
	@echo "  build          Build production Docker image"
	@echo "  build-dev      Build development Docker image"
	@echo "  up             Start all services with docker-compose"
	@echo "  down           Stop all services"
	@echo "  logs           View docker-compose logs"
	@echo ""
	@echo "Data Ingestion:"
	@echo "  ingest-filings  Ingest SEC filings (10-K, 10-Q, 8-K)"
	@echo "  ingest-earnings Ingest earnings transcripts"
	@echo "  ingest-13f      Ingest 13F institutional holdings"
	@echo ""
	@echo "Maintenance:"
	@echo "  clean          Clean build artifacts"
	@echo "  reset-db       Reset ChromaDB"

# ============================================================================
# Setup
# ============================================================================

install:
	poetry install --only=main

install-dev:
	poetry install

# ============================================================================
# Development
# ============================================================================

dev:
	poetry run uvicorn financial_intel.api.main:app --reload --host 0.0.0.0 --port 8000

notebook:
	poetry run jupyter lab --ip=0.0.0.0 --port=8888 --no-browser

run:
	@if [ -z "$(QUERY)" ]; then \
		echo "Usage: make run QUERY=\"your research query\""; \
		exit 1; \
	fi
	poetry run fi research "$(QUERY)"

# ============================================================================
# Quality
# ============================================================================

test:
	poetry run pytest tests/ -v

test-cov:
	poetry run pytest tests/ --cov=financial_intel --cov-report=term-missing --cov-report=html

lint:
	poetry run ruff check .

format:
	poetry run ruff check --fix .
	poetry run ruff format .

type-check:
	poetry run mypy src/financial_intel

quality: lint type-check test

# ============================================================================
# Docker
# ============================================================================

build:
	docker build -f docker/Dockerfile -t financial-intel:latest ..

build-dev:
	docker build -f docker/Dockerfile.dev -t financial-intel:dev ..

up:
	docker-compose -f docker/docker-compose.yml up -d

down:
	docker-compose -f docker/docker-compose.yml down

logs:
	docker-compose -f docker/docker-compose.yml logs -f

ps:
	docker-compose -f docker/docker-compose.yml ps

# ============================================================================
# Data Ingestion
# ============================================================================

ingest-filings:
	@if [ -z "$(TICKER)" ] || [ -z "$(FORMS)" ]; then \
		echo "Usage: make ingest-filings TICKER=NVDA FORMS=10-K,10-Q"; \
		exit 1; \
	fi
	poetry run fi ingest filings "$(TICKER)" --forms "$(FORMS)"

ingest-earnings:
	@if [ -z "$(TICKER)" ] || [ -z "$(QUARTERS)" ]; then \
		echo "Usage: make ingest-earnings TICKER=NVDA QUARTERS=4"; \
		exit 1; \
	fi
	poetry run fi ingest earnings "$(TICKER)" --quarters "$(QUARTERS)"

ingest-13f:
	@if [ -z "$(TICKER)" ] || [ -z "$(QUARTERS)" ]; then \
		echo "Usage: make ingest-13f TICKER=NVDA QUARTERS=4"; \
		exit 1; \
	fi
	poetry run fi ingest 13f "$(TICKER)" --quarters "$(QUARTERS)"

# ============================================================================
# Maintenance
# ============================================================================

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .coverage htmlcov dist build *.egg-info

reset-db:
	rm -rf data/chromadb/*
	@echo "ChromaDB reset. Restart services to reinitialize."

# ============================================================================
# Documentation
# ============================================================================

docs:
	@echo "Generating API documentation..."
	poetry run python -m financial_intel.cli.main --help > docs/cli_help.txt
	@echo "CLI help saved to docs/cli_help.txt"
