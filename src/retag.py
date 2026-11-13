"""Re-apply klass/ptype to an existing predicates.parquet WITHOUT re-parsing.

`klass` is a pure function of (field, ptype, value), so a tagging-rule change only
needs the value re-run through the same classifier -- not a 55-min re-parse. This
imports the SAME functions the parser uses (single source of truth), so the result
is identical to a fresh parse. Only `command` and `registry` tagging can change;
all other fields are left untouched.

    python src/retag.py
"""
from __future__ import annotations

import pandas as pd

from config import DATA_DIR
from parser import classify_command, classify_registry

PATH = DATA_DIR / "predicates.parquet"


def main() -> None:
    df = pd.read_parquet(PATH)
    before = df.groupby("klass", observed=True).size().to_dict()
    # columns come back dictionary-encoded; make them plain strings so new
    # ptype categories (e.g. reg_defense_tamper) can be assigned
    df["ptype"] = df["ptype"].astype("object")
    df["klass"] = df["klass"].astype("object")

    for field, fn in [("command", classify_command), ("registry", classify_registry)]:
        mask = df["field"] == field
        vals = df.loc[mask, "value"]
        # cache per distinct value (few distinct commands/keys relative to rows)
        uniq = {v: fn(v) for v in vals.unique()}
        df.loc[mask, "ptype"] = vals.map(lambda v: uniq[v][0]).to_numpy()
        df.loc[mask, "klass"] = vals.map(lambda v: uniq[v][1]).to_numpy()

    after = df.groupby("klass", observed=True).size().to_dict()
    df.to_parquet(PATH, index=False)
    print(f"klass before: {before}")
    print(f"klass after : {after}")
    moved = after.get("mechanism", 0) - before.get("mechanism", 0)
    print(f"re-tagged {moved:+,} predicate rows artifact -> mechanism")
    print(f"rewrote {PATH}")


if __name__ == "__main__":
    main()
