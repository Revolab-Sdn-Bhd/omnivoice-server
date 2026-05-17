.PHONY: help install dev test test-cov lint format type-check clean build publish

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install with auto-detected PyTorch CUDA variant
	bash scripts/install.sh

dev: ## Install with dev dependencies and auto-detected PyTorch CUDA variant
	bash scripts/install.sh
	uv sync --extra dev

test: ## Run tests
	pytest tests/ -v

test-e2e: ## Run e2e benchmark against live server (requires server running)
	uv run python scripts/e2e_benchmark.py

test-cov: ## Run tests with coverage report
	pytest tests/ -v --cov=omnivoice_server --cov-report=term-missing --cov-report=html

lint: ## Run linting with ruff
	ruff check omnivoice_server/ tests/

format: ## Format code with ruff
	ruff format omnivoice_server/ tests/

type-check: ## Run type checking with mypy
	mypy omnivoice_server/

clean: ## Clean build artifacts
	rm -rf build/ dist/ *.egg-info/ .pytest_cache/ .mypy_cache/ .ruff_cache/ htmlcov/ .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

build: ## Build package wheel and sdist
	python -m build

publish-test: ## Publish to TestPyPI (for testing)
	python -m twine upload --repository testpypi dist/*

publish: ## Publish to PyPI (requires authentication)
	python -m twine upload dist/*

pre-commit-install: ## Install pre-commit hooks
	pre-commit install

pre-commit-run: ## Run pre-commit hooks on all files
	pre-commit run --all-files
