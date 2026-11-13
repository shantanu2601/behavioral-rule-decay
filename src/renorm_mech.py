"""Apply the uniform mechanism-value normalization to an existing
predicates.parquet WITHOUT re-parsing.

A mechanism predicate must key on the technique SKELETON, not on the volatile
campaign tokens embedded in its value (random Run value-names, scheduled-task
GUIDs, user-writable drop paths, self-relaunch exe names). This runs every
mechanism-class value through the SAME `canon_mech_value()` the parser now uses
(single source of truth), so the result equals a fresh parse. Artifacts are left
untouched. Rows are de-duplicated after collapsing.

    python src/renorm_mech.py
"""
from __future__ import annotations

import pandas as pd

from config import DATA_DIR
from parser import canon_mech_value

PATH = DATA_DIR / "predicates.parquet"
FIELDS = ("registry", "command", "process")   # file drop-path done by renorm_drops.py


def main() -> None:
    df = pd.read_parquet(PATH)
    n0 = len(df)

    mask = (df["field"].isin(FIELDS)) & (df["klass"] == "mechanism")
    sub = df.loc[mask, ["field", "ptype", "value"]]
    # cache per distinct (field, ptype, value)
    uniq = {t: canon_mech_value(*t)
            for t in sub.drop_duplicates().itertuples(index=False, name=None)}
    before_distinct = sub["value"].nunique()
    df.loc[mask, "value"] = [uniq[(f, p, v)] for f, p, v in
                             zip(sub["field"], sub["ptype"], sub["value"])]
    after_distinct = df.loc[mask, "value"].nunique()

    df = df.drop_duplicates(subset=["sha", "field", "klass", "ptype", "value"]) \
           .reset_index(drop=True)

    print(f"normalized {mask.sum():,} mechanism rows in {FIELDS}")
    print(f"distinct mechanism values: {before_distinct:,} -> {after_distinct:,}")
    print(f"rows: {n0:,} -> {len(df):,}")
    df.to_parquet(PATH, index=False)
    print(f"rewrote {PATH}")


if __name__ == "__main__":
    main()
