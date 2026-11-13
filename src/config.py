"""Central paths and constants for the verdict-decay pipeline.

Data lives OUTSIDE the repo (in the Projects parent) because the extracted
reports zip is 17 GB. Read reports on demand from the zip; do not extract it
(the loose JSON totals ~564 GiB).
"""
from __future__ import annotations

from pathlib import Path

# --- reproducibility -------------------------------------------------------
SEED = 20260802

# --- repo-relative dirs ----------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"        # gitignored intermediate artifacts (parquet, etc.)
REPORTS_DIR = REPO_ROOT / "reports"  # committed figures/tables/matrices

# --- external raw data (parent of the repo) --------------------------------
PROJECTS_DIR = REPO_ROOT.parent
REPORTS_ZIP = PROJECTS_DIR / "public_full_reports.zip"      # 17 GB, 48,976 json reports
LABELS_CSV = PROJECTS_DIR / "public_labels.csv"            # sha256,family,type,date

# 10 families in the Avast-CTU CAPEv2 dataset
FAMILIES = [
    "Emotet", "Swisyn", "Qakbot", "Trickbot", "Lokibot",
    "njRAT", "Zeus", "Ursnif", "Adload", "HarHar",
]
