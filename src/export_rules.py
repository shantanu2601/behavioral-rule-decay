"""Human-readable exports of data/rules.parquet.

The parquet `predicates` column is a nested list-of-structs, which table viewers
show as `[object Object]`. This flattens it into:
  reports/rules.csv  one row per rule, predicates joined into one string column
  reports/rules.txt  grouped by family, easy to skim
"""
from __future__ import annotations

import pandas as pd

from config import DATA_DIR, REPORTS_DIR


def rule_str(preds) -> str:
    return "  AND  ".join(f"[{p['klass']}/{p['field']}] {p['value']}" for p in preds)


def main() -> None:
    r = pd.read_parquet(DATA_DIR / "rules.parquet")

    flat = r.drop(columns=["predicates"]).copy()
    flat["rule"] = r["predicates"].map(rule_str)
    flat.to_csv(REPORTS_DIR / "rules.csv", index=False)

    lines = []
    for fam, g in r.groupby("family", observed=True):
        lines.append(f"\n{'='*80}\n{fam}  ({len(g)} rules)\n{'='*80}")
        for _, row in g.iterrows():
            lines.append(
                f"{row.rule_id:14} {row.composition:15} "
                f"prec={row.train_precision} cov={row.train_coverage}")
            for p in row.predicates:
                lines.append(f"      [{p['klass']:9}/{p['field']:8}] {p['value']}")
    (REPORTS_DIR / "rules.txt").write_text("\n".join(lines))
    print(f"wrote reports/rules.csv ({len(flat)} rows) and reports/rules.txt")


if __name__ == "__main__":
    main()
