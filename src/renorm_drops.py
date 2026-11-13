"""Re-normalize drop-persist predicates in an existing predicates.parquet WITHOUT
re-parsing.

Fixes the granularity bug: a persistence drop was tagged `mechanism` but keyed on
the FULL directory incl. the campaign-specific sub-folder (`%LOCALAPPDATA%\\iproppass\\`),
so it decayed like the artifact inside it. The mechanism should key on the auto-run
LOCATION CLASS (`%LOCALAPPDATA%\\*\\`); the specific sub-folder becomes an artifact.

This reproduces exactly what a fresh parse of the updated _files() would emit, since
the old `drop_persist_dir` value IS the specific directory we need. Each such row ->
  * mechanism `drop_persist_loc` = persist_location_class(value)
  * artifact  `drop_dir`        = original value        (if different)
then de-duplicated within (sha, field, klass, ptype, value).

    python src/renorm_drops.py
"""
from __future__ import annotations

import pandas as pd

from config import DATA_DIR
from parser import persist_location_class

PATH = DATA_DIR / "predicates.parquet"


def main() -> None:
    df = pd.read_parquet(PATH)
    n0 = len(df)

    mask = (df["field"] == "file") & (df["ptype"] == "drop_persist_dir")
    old = df.loc[mask].copy()
    print(f"drop_persist_dir rows to renormalize: {len(old):,} "
          f"({old['value'].nunique():,} distinct dirs)")

    keep = df.loc[~mask].copy()

    # mechanism: location class
    loc = old.copy()
    loc["ptype"] = "drop_persist_loc"
    loc["value"] = loc["value"].map(persist_location_class)

    # artifact: the specific sub-folder (only where it differs from the class)
    art = old.copy()
    art["klass"] = "artifact"
    art["ptype"] = "drop_dir"
    art = art[art["value"] != art["value"].map(persist_location_class)]

    out = pd.concat([keep, loc, art], ignore_index=True)
    out = out.drop_duplicates(subset=["sha", "field", "klass", "ptype", "value"]) \
             .reset_index(drop=True)

    print(f"location classes produced: {sorted(loc['value'].unique())}")
    print(f"rows: {n0:,} -> {len(out):,}")
    out.to_parquet(PATH, index=False)
    print(f"rewrote {PATH}")


if __name__ == "__main__":
    main()
