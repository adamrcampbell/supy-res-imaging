VENV = venv
PYTHON = $(VENV)/bin/python3
PIP = $(VENV)/bin/pip

.PHONY: run clean freeze

run: $(VENV)/bin/activate
	$(PYTHON) supy-res-imaging.py

$(VENV)/bin/activate: requirements.txt
	python3 -m venv $(VENV)
	source $(VENV)/bin/activate
	$(PIP) install -r requirements.txt

clean:
	rm -rf __pycache__
	rm -rf $(VENV)

freeze:
	pip3 freeze > requirements.txt