.PHONY: venv install setup upgrade-deps clean

venv: ## Create local virtual environment (.venv)
	@test -d "$(VENV_DIR)" || $(PYTHON) -m venv $(VENV_DIR)

install: venv ## Install project dependencies
	$(PIP) install -r $(REQUIREMENTS)

setup: install ## Full local setup (venv + deps)

upgrade-deps: venv ## Upgrade pip/setuptools/wheel
	$(PIP) install --upgrade pip setuptools wheel

clean: ## Remove caches and local artifacts
	rm -rf __pycache__ .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type d -name ".ipynb_checkpoints" -prune -exec rm -rf {} +

