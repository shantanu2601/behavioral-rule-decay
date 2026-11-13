.PHONY: setup matrix parse drift prune mine survival confound rules figures reproduce clean

PY := python

## install pinned dependencies
setup:
	$(PY) -m pip install -r requirements.txt

## family x detection-quarter matrix -> reports/
matrix:
	$(PY) src/family_quarter_matrix.py

## reports zip -> typed, class-tagged predicates (data/predicates.parquet), ~1 hour
parse:
	$(PY) src/build_predicates.py

## drift check: LightGBM family classifier, train 2018Q3-Q4, test 2019 quarters
drift:
	$(PY) src/drift_gate.py

## prune predicates on the training window
prune:
	$(PY) src/pruning.py

## cluster each family and mine rules -> data/rules.parquet
mine:
	$(PY) src/mining.py

## per-predicate and per-rule survival over 2019 quarters
survival:
	$(PY) src/survival.py

## family- and size-controlled rule-level comparison
confound:
	$(PY) src/rule_confound.py

## human-readable rule exports -> reports/rules.{csv,txt}
rules:
	$(PY) src/export_rules.py

## paper figures -> reports/fig_*.png
figures:
	$(PY) src/figures.py

## full pipeline, in order
reproduce: matrix parse drift prune mine survival confound rules figures

clean:
	rm -rf data/*.parquet
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
