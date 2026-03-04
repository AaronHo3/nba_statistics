.PHONY: run run-dev check

run: setup validate-data ## Start Streamlit dashboard
	$(STREAMLIT) run app.py

run-dev: setup validate-data ## Start Streamlit with file watcher
	$(STREAMLIT) run app.py --server.runOnSave=true

check: setup ## Syntax check app
	$(PY) -m py_compile app.py

