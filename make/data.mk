.PHONY: data-dir list-data validate-data download-data

data-dir: ## Create expected data folders
	mkdir -p $(DATA_DIR)
	mkdir -p $(REPORTS_DIR)

list-data: data-dir ## List CSV files found in data/raw
	@echo "CSV files in $(DATA_DIR):"
	@ls -1 $(DATA_DIR)/*.csv 2>/dev/null || echo "  (none found)"

validate-data: data-dir ## Check that at least one CSV exists in data/raw
	@count=$$(ls -1 $(DATA_DIR)/*.csv 2>/dev/null | wc -l | tr -d ' '); \
	if [ "$$count" -eq 0 ]; then \
		echo "No CSV files found in $(DATA_DIR)."; \
		echo "Download from Kaggle and place CSV files there."; \
		exit 1; \
	fi; \
	echo "Found $$count CSV file(s) in $(DATA_DIR)."

download-data: setup data-dir ## Download Kaggle dataset via kagglehub and copy CSVs to data/raw
	$(PY) scripts/download_data.py
