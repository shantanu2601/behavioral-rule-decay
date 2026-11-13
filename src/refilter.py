"""Drop OS/sandbox background predicates from an existing predicates.parquet
WITHOUT re-parsing.

Whether a value is benign background is a pure function of (field, value), so a
noise-filter change only needs the values re-run through the SAME predicates the
parser uses -- not a 55-min re-parse. This imports those predicates directly
(single source of truth), so the result is identical to a fresh parse. Idempotent:
re-applying the old filters drops nothing new; only the newly added noise classes
(cert/web cache, CDN/cert-infra network, OS registry/mutex objects) are removed.

    python src/refilter.py
"""
from __future__ import annotations

import pandas as pd

from config import DATA_DIR
from parser import (
    BENIGN_FILE,
    BENIGN_IP,
    BENIGN_MUTEX,
    BENIGN_MUTEX_SUB,
    BENIGN_NET_DOMAIN,
    BENIGN_REG,
    PERSIST_REG,
    is_benign_ip,
)

PATH = DATA_DIR / "predicates.parquet"


def _mutex_noise(v: str) -> bool:
    return v in BENIGN_MUTEX or bool(BENIGN_MUTEX_SUB.search(v))


def _registry_noise(v: str) -> bool:
    return bool(BENIGN_REG.search(v)) and not PERSIST_REG.search(v)


def _file_noise(v: str) -> bool:
    return bool(BENIGN_FILE.search(v) or BENIGN_FILE.search(v.rsplit("\\", 1)[-1]))


def _network_noise(ptype: str, v: str) -> bool:
    if ptype == "host_ip":
        return is_benign_ip(v)
    if ptype == "domain":
        return bool(BENIGN_NET_DOMAIN.search(v))
    return False


def main() -> None:
    df = pd.read_parquet(PATH)
    n0 = len(df)
    before = df.groupby("klass", observed=True).size().to_dict()

    drop = pd.Series(False, index=df.index)

    for field, fn in [("mutex", _mutex_noise), ("registry", _registry_noise),
                      ("file", _file_noise)]:
        mask = df["field"] == field
        vals = df.loc[mask, "value"]
        uniq = {v: fn(v) for v in vals.unique()}
        drop.loc[mask] = vals.map(uniq).to_numpy()

    nmask = df["field"] == "network"
    nsub = df.loc[nmask, ["ptype", "value"]]
    uniq_net = {t: _network_noise(t[0], t[1])
                for t in nsub.itertuples(index=False, name=None)}
    drop.loc[nmask] = [uniq_net[(p, v)] for p, v in
                       zip(nsub["ptype"], nsub["value"])]

    dropped_by_field = df.loc[drop, "field"].value_counts().to_dict()
    df = df.loc[~drop].reset_index(drop=True)

    after = df.groupby("klass", observed=True).size().to_dict()
    df.to_parquet(PATH, index=False)
    print(f"rows: {n0:,} -> {len(df):,}  (dropped {drop.sum():,})")
    print(f"dropped by field: {dropped_by_field}")
    print(f"klass before: {before}")
    print(f"klass after : {after}")
    print(f"rewrote {PATH}")


if __name__ == "__main__":
    main()
