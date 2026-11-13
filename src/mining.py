"""Rule mining (build step 5).

Per family: HDBSCAN into variant clusters -> per-cluster frequent-itemset mining
-> 3-5 predicate conjunctions that fire for the family and (almost) never for the
other 9. Support is measured INSIDE a cluster (recovers campaign-bound artifacts);
api_seq 3-grams are capped per cluster so they don't drown discrete predicates.
Rules are frozen on the 2018Q3-Q4 window; survival.py measures them on 2019 later.

Signed-off knobs (2026-08-03): train precision >= 0.95, cap api_seq per cluster.

Output: data/rules.parquet (one row per rule, with predicate-class composition).
"""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd
import hdbscan
from mlxtend.frequent_patterns import fpgrowth

from config import DATA_DIR, FAMILIES, LABELS_CSV, SEED

TRAIN_Q = ["2018Q3", "2018Q4"]

MIN_TRAIN_PRECISION = 0.95   # rule must fire >=95% on-family over the train window
MIN_COV_SAMPLES = 10         # ... and cover at least this many family train samples
MIN_CLUSTER_SUPPORT = 0.40   # predicate/itemset frequency within its cluster
MAX_PREDICATES = 4           # verdict rules are 3-5; cap at 4 to bound the lattice
MIN_PREDICATES = 2
API_SEQ_TOPK = 25            # most-discriminative api_seq kept per cluster
POOL_CAP = 30                # total candidate items per cluster (bounds fp-growth)
ITEMSET_CAP = 4000           # keep at most this many frequent itemsets per cluster
MIN_CLUSTER_MINE = 15        # don't mine clusters smaller than this
CLUSTER_FEAT_SUPPORT = 0.05  # family support for a pid to be a clustering feature
RULES_PER_FAMILY = 60        # greedy set-cover cap


def load_train():
    lab = pd.read_csv(LABELS_CSV, parse_dates=["date"]).rename(
        columns={"classification_family": "family"})
    lab["q"] = lab["date"].dt.to_period("Q").astype(str)
    lab = lab[lab.q.isin(TRAIN_Q)]
    pruned = pd.read_parquet(DATA_DIR / "predicates_pruned.parquet",
                             columns=["sha", "pid", "field", "klass", "quarter"])
    pruned = pruned[pruned.quarter.isin(TRAIN_Q)]
    stats = pd.read_parquet(DATA_DIR / "predicate_stats.parquet")
    meta = stats.set_index("pid")[["field", "klass", "ptype", "value", "precision_train"]]
    return lab, pruned, meta


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    lab, pruned, meta = load_train()
    sha2fam = dict(zip(lab.sha256, lab.family))
    fam_shas = {f: set(g.sha256) for f, g in lab.groupby("family")}
    n_f = {f: len(s) for f, s in fam_shas.items()}
    klass_of = meta["klass"].to_dict()

    # inverted index over the whole train window: pid -> set of shas (all families)
    pid2shas = {pid: set(g.sha.values) for pid, g in pruned.groupby("pid")}
    # per-sample transaction: sha -> set of pids
    sha2pids = {sha: set(g.pid.values) for sha, g in pruned.groupby("sha")}
    is_api = meta["field"] == "api_seq"

    rules: list[dict] = []
    rid = 0
    print(f"{'family':10} {'train':>6} {'clusters':>9} {'cands':>7} {'kept':>6}")
    for fam in FAMILIES:
        fshas = [s for s in fam_shas[fam] if s in sha2pids]
        if not fshas:
            print(f"{fam:10} {0:>6}  (no train samples)")
            continue
        t0 = time.time()
        labels = cluster_family(fam, fshas, sha2pids, meta)
        cand, kept = 0, 0
        fam_rules: list[dict] = []
        for cl in sorted(set(labels)):
            members = [s for s, c in zip(fshas, labels) if c == cl]
            if len(members) < MIN_CLUSTER_MINE:
                continue
            print(f"  {fam} cluster {cl}: {len(members)} samples, mining...", flush=True)
            itemsets = mine_cluster(members, sha2pids, meta, is_api)
            for pids in itemsets:
                cand += 1
                match = set.intersection(*(pid2shas[p] for p in pids))
                on = match & fam_shas[fam]
                if len(on) < MIN_COV_SAMPLES:
                    continue
                prec = len(on) / len(match)
                if prec < MIN_TRAIN_PRECISION:
                    continue
                fam_rules.append(dict(pids=frozenset(pids), match_on=on,
                                      precision=prec, coverage=len(on)))
        # dedup identical predicate-sets, annotate composition
        seen: set = set()
        uniq: list[dict] = []
        for r in fam_rules:
            if r["pids"] in seen:
                continue
            seen.add(r["pids"])
            r["n_art"] = sum(1 for p in r["pids"] if klass_of.get(p) == "artifact")
            uniq.append(r)
        # stratified selection: reserve up to half the budget for artifact-containing
        # rules so they aren't outranked by higher-coverage api_seq rules
        key = lambda r: (r["coverage"] * r["precision"], -len(r["pids"]))
        art = sorted((r for r in uniq if r["n_art"] > 0), key=key, reverse=True)
        mech = sorted((r for r in uniq if r["n_art"] == 0), key=key, reverse=True)
        quota = RULES_PER_FAMILY // 2
        chosen = art[:quota] + mech[:RULES_PER_FAMILY - min(quota, len(art))]
        if len(chosen) < RULES_PER_FAMILY:  # fill leftover from whichever remains
            extra = art[quota:] + mech[RULES_PER_FAMILY - min(quota, len(art)):]
            chosen += sorted(extra, key=key, reverse=True)[:RULES_PER_FAMILY - len(chosen)]
        for r in chosen:
            rules.append(_finalize(rid, fam, r, meta))
            rid += 1
            kept += 1
        n_clusters = len({c for c in labels if c >= 0}) or 1
        print(f"{fam:10} {len(fshas):>6} {n_clusters:>9} {cand:>7} {kept:>6}")

    rdf = pd.DataFrame(rules)
    rdf.to_parquet(DATA_DIR / "rules.parquet", index=False)
    _report(rdf)


def cluster_family(fam, fshas, sha2pids, meta) -> np.ndarray:
    """HDBSCAN (Jaccard) over family train samples. Fallback: single cluster."""
    # clustering features = pids reaching CLUSTER_FEAT_SUPPORT within the family
    from collections import Counter
    c: Counter = Counter()
    for s in fshas:
        c.update(sha2pids[s])
    thr = CLUSTER_FEAT_SUPPORT * len(fshas)
    feats = [p for p, k in c.items() if k >= thr]
    if len(feats) < 3 or len(fshas) < 2 * MIN_CLUSTER_MINE:
        return np.zeros(len(fshas), dtype=int)
    col = {p: i for i, p in enumerate(feats)}
    X = np.zeros((len(fshas), len(feats)), dtype=bool)
    for i, s in enumerate(fshas):
        for p in sha2pids[s] & set(feats):
            X[i, col[p]] = True
    mcs = max(MIN_CLUSTER_MINE, int(0.03 * len(fshas)))
    try:
        lab = hdbscan.HDBSCAN(min_cluster_size=mcs, metric="jaccard").fit_predict(X)
    except Exception:
        return np.zeros(len(fshas), dtype=int)
    # noise (-1) -> its own catch-all cluster id
    lab = np.where(lab < 0, lab.max() + 1 if lab.max() >= 0 else 0, lab)
    return lab


def mine_cluster(members, sha2pids, meta, is_api) -> list[frozenset]:
    """fp-growth on a cluster's transactions, api_seq capped. Returns pid-itemsets."""
    n = len(members)
    from collections import Counter
    supp: Counter = Counter()
    for s in members:
        supp.update(sha2pids[s])
    # candidate pool: cluster support >= threshold
    cand = [p for p, k in supp.items() if k >= MIN_CLUSTER_SUPPORT * n]
    if not cand:
        return []

    def disc_score(p):  # discrimination: cluster support * global train precision
        return (supp[p] / n) * float(meta["precision_train"].get(p, 0.0))

    api = sorted((p for p in cand if bool(is_api.get(p, False))), key=disc_score, reverse=True)
    disc = sorted((p for p in cand if not bool(is_api.get(p, False))), key=disc_score, reverse=True)
    # keep ALL discrete (few, and the artifacts we care about); fill rest with top api
    pool = list(disc) + api[:API_SEQ_TOPK]
    if len(pool) > POOL_CAP:  # keep the most discriminative overall, bound the lattice
        pool = sorted(pool, key=disc_score, reverse=True)[:POOL_CAP]
    pool = set(pool)
    if len(pool) < MIN_PREDICATES:
        return []
    cols = sorted(pool)
    cidx = {p: i for i, p in enumerate(cols)}
    M = np.zeros((n, len(cols)), dtype=bool)
    for i, s in enumerate(members):
        for p in sha2pids[s] & pool:
            M[i, cidx[p]] = True
    df = pd.DataFrame(M, columns=cols)
    fi = fpgrowth(df, min_support=MIN_CLUSTER_SUPPORT, max_len=MAX_PREDICATES,
                  use_colnames=True)
    fi = fi[fi["itemsets"].map(len) >= MIN_PREDICATES]
    if fi.empty:
        return []
    # keep ALL itemsets that contain a discrete (artifact-capable) predicate — they
    # are few and are exactly what mustn't be cut; bound only the api-only ones
    has_disc = fi["itemsets"].map(lambda s: any(not bool(is_api.get(p, False)) for p in s))
    disc_sets = fi[has_disc]
    api_sets = fi[~has_disc]
    budget = max(0, ITEMSET_CAP - len(disc_sets))
    if len(api_sets) > budget:
        api_sets = api_sets.nlargest(budget, "support")
    out = pd.concat([disc_sets, api_sets])
    return [frozenset(its) for its in out["itemsets"]]


def _finalize(rid, fam, r, meta) -> dict:
    preds = []
    for p in r["pids"]:
        m = meta.loc[p]
        preds.append(dict(field=m["field"], klass=m["klass"], ptype=m["ptype"],
                          value=m["value"]))
    n_art = sum(1 for x in preds if x["klass"] == "artifact")
    n_mech = sum(1 for x in preds if x["klass"] == "mechanism")
    comp = ("mechanism_only" if n_art == 0 else
            "artifact_only" if n_mech == 0 else "mixed")
    return dict(rule_id=f"{fam.lower()}_{rid}", family=fam,
                n_predicates=len(preds), n_artifact=n_art, n_mechanism=n_mech,
                composition=comp, train_precision=round(r["precision"], 4),
                train_coverage=r["coverage"], predicates=preds)


def _report(rdf: pd.DataFrame) -> None:
    print(f"\ntotal rules: {len(rdf)}")
    if rdf.empty:
        return
    print("\nby composition:")
    print(rdf.composition.value_counts().to_string())
    print("\nrules per family:")
    print(rdf.groupby("family", observed=True).size().sort_values(ascending=False).to_string())
    print(f"\npredicate count: mean {rdf.n_predicates.mean():.1f} "
          f"(min {rdf.n_predicates.min()}, max {rdf.n_predicates.max()})")
    print(f"train precision: mean {rdf.train_precision.mean():.3f}")
    print(f"\nwrote data/rules.parquet")


if __name__ == "__main__":
    main()
