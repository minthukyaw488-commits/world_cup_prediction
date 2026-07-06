.PHONY: install test train predict simulate knockout download-data

PYTHON ?= python
RUNS ?= 10000

install:
	pip install -e . pytest

test:
	$(PYTHON) -m pytest -q

train:
	$(PYTHON) -m wc_predictor.train

# usage: make predict HOME=Argentina AWAY=France
predict:
	$(PYTHON) -m wc_predictor.predict "$(HOME)" "$(AWAY)"

simulate:
	mkdir -p output
	$(PYTHON) -m wc_predictor.simulate --runs $(RUNS) --out output/wc2026_forecast.csv

# usage: make knockout BRACKET=data/example_r16_bracket.csv
BRACKET ?= data/example_r16_bracket.csv
knockout:
	$(PYTHON) -m wc_predictor.simulate --knockout $(BRACKET) --runs $(RUNS)

download-data:
	$(PYTHON) scripts/download_full_data.py
