"""Family x detection-quarter matrix — the C2 (TESSERACT) sanity check.

Answers, straight from the labels (no report parsing needed):
  - how many quarters do we actually have, and how many samples per quarter?
  - which families are present in which quarters (does a family enter/leave)?
  - is "rule decay" going to be confounded with a family disappearing from the
    dataset (a family-attribution reinterpretation of TESSERACT's C2)?

Sorts on the DETECTION date (2017-2019 bulk), never the 2021 sandbox timestamp.

Outputs (written to reports/):
  - family_quarter_matrix.csv   counts, families x quarter
  - family_quarter_share.csv    each family's share of the dataset per quarter
  - printed summary to stdout
"""
from __future__ import annotations

import sys

import pandas as pd

from config import FAMILIES, LABELS_CSV, REPORTS_DIR


def load_labels() -> pd.DataFrame:
    df = pd.read_csv(LABELS_CSV, parse_dates=["date"])
    # column names from the file header: sha256, classification_family,
    # classification_type, date
    df = df.rename(columns={"classification_family": "family"})
    df["quarter"] = df["date"].dt.to_period("Q").astype(str)
    return df


def main() -> None:
    if not LABELS_CSV.exists():
        sys.exit(f"labels not found at {LABELS_CSV}")
    REPORTS_DIR.mkdir(exist_ok=True)

    df = load_labels()
    n = len(df)

    print(f"samples: {n:,}   date span: {df['date'].min().date()} -> {df['date'].max().date()}")
    print(f"distinct quarters: {df['quarter'].nunique()}")

    # --- counts: family x quarter ---
    mat = (
        df.pivot_table(index="family", columns="quarter", values="sha256",
                       aggfunc="count", fill_value=0)
        .reindex(FAMILIES)
    )
    mat["TOTAL"] = mat.sum(axis=1)
    mat.loc["TOTAL"] = mat.sum(axis=0)
    mat.to_csv(REPORTS_DIR / "family_quarter_matrix.csv")

    # --- share of each quarter (base-rate drift the fixed-mix precision must control) ---
    counts = mat.drop(index="TOTAL", columns="TOTAL")
    share = counts.div(counts.sum(axis=0), axis=1).round(3)
    share.to_csv(REPORTS_DIR / "family_quarter_share.csv")

    # --- quarter totals + how many families present per quarter ---
    q_total = counts.sum(axis=0).astype(int)
    q_fams = (counts > 0).sum(axis=0).astype(int)

    print("\nquarter        samples   #families present")
    for q in q_total.index:
        print(f"  {q:<10} {q_total[q]:>8,}   {q_fams[q]:>2}")

    # --- family churn: first/last quarter each family appears ---
    print("\nfamily        first_q   last_q    n_quarters   total")
    for fam in FAMILIES:
        present = counts.columns[counts.loc[fam] > 0]
        if len(present) == 0:
            print(f"  {fam:<12} (never appears?)")
            continue
        print(f"  {fam:<12} {present.min():<9} {present.max():<9} "
              f"{len(present):>6}      {int(counts.loc[fam].sum()):>6,}")

    print(f"\nwrote reports/family_quarter_matrix.csv and reports/family_quarter_share.csv")


if __name__ == "__main__":
    main()
