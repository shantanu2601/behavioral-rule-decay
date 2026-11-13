"""Pruning pass — reduce ~412k candidate predicates to a discriminative set.

Runs BEFORE mining. All statistics are computed on the TRAIN (freeze) window only
(2018Q3-Q4, same slice the rules are mined+frozen on) so nothing about the later
test quarters leaks into which predicates exist.

Two dead zones are cut, nothing in between (conjunctions are decided later by the
miner, so we must NOT drop predicates that are only useful in combination):
  * ultra-rare  : train df < MIN_DF  -> per-run noise, can't support a rule
  * ubiquitous  : supported in > FAMILY_SPREAD_MAX families -> background behavior
                  (stall loops, DLL loading) with no discriminating power
A predicate is kept only if it also covers >= MIN_BEST_SUPPORT of some family
(so it can actually anchor / participate in a rule).

Outputs:
  data/predicate_stats.parquet   one row per surviving predicate + train stats
  data/predicates_pruned.parquet membership rows (sha x surviving predicate) over
                                 the usable window (train + eval) -> miner+survival input
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import DATA_DIR, LABELS_CSV

TRAIN_Q = ["2018Q3", "2018Q4"]
EVAL_Q = ["2019Q1", "2019Q2", "2019Q3", "2019Q4"]

MIN_DF = 20              # >= this many TRAIN samples contain it (kills rare noise)
SUPP_MIN = 0.05          # support at/above which a family "counts" as supported
FAMILY_SPREAD_MAX = 5    # supported in more families than this -> background, cut
MIN_BEST_SUPPORT = 0.05  # must cover >=5% of its best family (anchorable)


def load_samples() -> pd.DataFrame:
    df = pd.read_csv(LABELS_CSV, parse_dates=["date"]).rename(
        columns={"classification_family": "family"})
    df["quarter"] = df["date"].dt.to_period("Q").astype(str)
    df = df[df["quarter"].isin(TRAIN_Q + EVAL_Q)]
    return df[["sha256", "family", "quarter"]].rename(columns={"sha256": "sha"})


def main() -> None:
    samples = load_samples()
    sha2fam = dict(zip(samples["sha"], samples["family"]))
    sha2q = dict(zip(samples["sha"], samples["quarter"]))
    train_shas = set(samples.loc[samples.quarter.isin(TRAIN_Q), "sha"])
    n_f = samples[samples.quarter.isin(TRAIN_Q)].groupby("family").size()  # train family sizes
    print("train family sizes:\n" + n_f.to_string())

    preds = pd.read_parquet(DATA_DIR / "predicates.parquet",
                            columns=["sha", "field", "klass", "ptype", "value"])
    preds = preds[preds["sha"].isin(sha2fam)].copy()
    # unique int pid per (field, value) without a giant string concat
    fcodes = preds["field"].astype("category").cat.codes.to_numpy()
    vcodes, _ = pd.factorize(preds["value"])
    preds["pid"] = vcodes.astype(np.int64) * (int(fcodes.max()) + 1) + fcodes
    preds["family"] = preds["sha"].map(sha2fam)
    preds["quarter"] = preds["sha"].map(sha2q)
    print(f"usable-window predicate rows: {len(preds):,}  "
          f"distinct predicates: {preds['pid'].nunique():,}")

    # ---- train-window per (predicate, family) counts -> support ----
    tr = preds[preds["sha"].isin(train_shas)]
    tc = tr.groupby(["pid", "family"], observed=True).size().rename("c").reset_index()
    tc["supp"] = tc["c"] / tc["family"].map(n_f).to_numpy()

    # per-predicate aggregates
    stats = tc.groupby("pid").agg(
        df_train=("c", "sum"),
        best_supp=("supp", "max"),
        n_fams_supp=("supp", lambda s: int((s >= SUPP_MIN).sum())),
    )
    # best family + its precision  P(best family | predicate) on train
    best = tc.loc[tc.groupby("pid")["supp"].idxmax(), ["pid", "family", "c"]]
    best = best.rename(columns={"family": "best_family", "c": "best_c"}).set_index("pid")
    stats = stats.join(best)
    stats["precision_train"] = stats["best_c"] / stats["df_train"]

    # ---- keep / cut ----
    keep = (
        (stats["df_train"] >= MIN_DF)
        & (stats["best_supp"] >= MIN_BEST_SUPPORT)
        & (stats["n_fams_supp"] <= FAMILY_SPREAD_MAX)
        & (stats["n_fams_supp"] >= 1)
    )
    survivors = stats[keep].copy()

    # attach predicate metadata (field/klass/ptype/value) — one row per pid
    meta = preds.drop_duplicates("pid").set_index("pid")[["field", "klass", "ptype", "value"]]
    survivors = survivors.join(meta)

    _report(stats, survivors)

    survivors.reset_index().to_parquet(DATA_DIR / "predicate_stats.parquet", index=False)

    # membership table for survivors over the full usable window (mining + survival input)
    surv_ids = set(survivors.index)
    pruned = preds[preds["pid"].isin(surv_ids)][
        ["sha", "pid", "field", "klass", "ptype", "value", "family", "quarter"]]
    pruned.to_parquet(DATA_DIR / "predicates_pruned.parquet", index=False)
    print(f"\nwrote data/predicate_stats.parquet ({len(survivors):,} predicates)")
    print(f"wrote data/predicates_pruned.parquet ({len(pruned):,} membership rows)")


def _report(stats: pd.DataFrame, survivors: pd.DataFrame) -> None:
    print(f"\ncandidate predicates (train df>=1): {len(stats):,}")
    print(f"survivors after prune            : {len(survivors):,} "
          f"({100*len(survivors)/len(stats):.1f}%)")
    print("\nsurvivors by field:")
    print(survivors.groupby("field", observed=True).size().sort_values(ascending=False).to_string())
    print("\nsurvivors by klass:")
    print(survivors.groupby("klass", observed=True).size().to_string())
    print("\nsurvivors by best_family:")
    print(survivors.groupby("best_family", observed=True).size().sort_values(ascending=False).to_string())


if __name__ == "__main__":
    main()
