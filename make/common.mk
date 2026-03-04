VENV_DIR ?= .venv
PYTHON ?= python3
PIP := $(VENV_DIR)/bin/pip
PY := $(VENV_DIR)/bin/python
STREAMLIT := $(VENV_DIR)/bin/streamlit
JUPYTER := $(VENV_DIR)/bin/jupyter

REQUIREMENTS ?= requirements.txt
DATA_DIR ?= data/raw
REPORTS_DIR ?= reports

export PIP_DISABLE_PIP_VERSION_CHECK := 1

.DEFAULT_GOAL := help

.PHONY: help vars

help: ## Show available make targets
	@echo "Usage: make <target>"
	@echo ""
	@awk 'BEGIN {FS = ":.*## "}; /^[a-zA-Z0-9_.-]+:.*## / {printf "  %-18s %s\n", $$1, $$2}' $(MAKEFILE_LIST) | sort

vars: ## Print important make variables
	@echo "VENV_DIR=$(VENV_DIR)"
	@echo "PYTHON=$(PYTHON)"
	@echo "REQUIREMENTS=$(REQUIREMENTS)"
	@echo "DATA_DIR=$(DATA_DIR)"
	@echo "REPORTS_DIR=$(REPORTS_DIR)"

