# verdict-decay

How long do frozen behavioral malware-detection rules keep working, and does it depend on what they key on?

We mine family-identification rules from sandbox behavior reports in an early time window, freeze them, and measure each rule's recall on its own family over the following quarters. Every condition (predicate) in a rule is labeled as either:

- **artifact**: a value the attacker picks freely, such as a mutex name, dropped filename, or C2 address
- **mechanism**: a step the technique requires, such as an injection API sequence, an auto-run persistence location, or a defense-evasion command

The labels come from explicit rules in `src/parser.py`, not from a learned model. The main result: mechanism predicates keep far more of their original recall over time than artifact predicates, because families reuse techniques across campaigns but rotate the values they choose.

## Data

The [Avast-CTU Public CAPEv2 dataset](https://github.com/avast/avast-ctu-cape-dataset): 48,976 full sandbox reports, 10 families. The data is not included in this repo. Download the full reports and labels from the dataset page and place them next to this repo:

```
<parent>/
  public_full_reports.zip    # full reports (17 GB); read on demand, do not extract
  public_labels.csv          # sha256, family, type, date
  verdict-decay/             # this repo
```

Paths are set in `src/config.py`. Samples are ordered by detection date (mostly 2017 to 2019), not by sandbox execution time.

## Setup

Python 3.11.

```bash
make setup                   # pip install -r requirements.txt
brew install libomp          # macOS only: OpenMP runtime for LightGBM
```

## Pipeline

```bash
make reproduce               # runs every step below, in order
```

| Step | Command | Script | Output |
|---|---|---|---|
| Family by quarter counts | `make matrix` | `family_quarter_matrix.py` | `reports/family_quarter_*.csv` |
| Reports to predicates (about 1 hour) | `make parse` | `build_predicates.py`, `parser.py` | `data/predicates.parquet` |
| Drift check | `make drift` | `drift_gate.py` | `reports/drift_gate*` |
| Prune predicates (training window only) | `make prune` | `pruning.py` | `data/predicates_pruned.parquet`, `data/predicate_stats.parquet` |
| Cluster families and mine rules | `make mine` | `mining.py` | `data/rules.parquet` |
| Survival over test quarters | `make survival` | `survival.py` | `data/survival_*.parquet` |
| Rule-level comparison controlled for family and size | `make confound` | `rule_confound.py` | printed |
| Readable rule exports | `make rules` | `export_rules.py` | `reports/rules.csv`, `reports/rules.txt` |
| Figures | `make figures` | `figures.py` | `reports/fig_*.png` |

Training window: 2018Q3 to 2018Q4. Test quarters: 2019Q1 to 2019Q4. All predicate statistics and rule selection use the training window only.

`retag.py`, `refilter.py`, `renorm_drops.py`, and `renorm_mech.py` update an existing `predicates.parquet` after a labeling or filtering change, without re-parsing. They call the same functions as `parser.py`, so a fresh `make parse` already includes their changes and does not need them.

Fixed seed: `src/config.py:SEED`. Pinned dependencies: `requirements.txt`.
