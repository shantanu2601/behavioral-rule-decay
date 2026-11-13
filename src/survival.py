"""Survival analysis (build step 6 — the contribution).

Frozen rules/predicates from the 2018Q3-Q4 window, evaluated per 2019 quarter.

Per rule/predicate targeting family f, per quarter q:
  fire_g(q)          = fraction of family-g's quarter-q samples that match it
  recall(q)          = fire_f(q)                          # THE decay signal
  precision_fixed(q) = fire_f(q) / sum_g fire_g(q)        # uniform prior (pi cancels);
                       base-rate controlled — fire_g are rates, so the drifting
                       quarter family-mix cannot fake decay
Primary decay signal = recall-retention = recall(q) / train_recall.
  death  = recall_retention < 0.5   (behavior/value stopped matching)
           OR precision_fixed < theta_p (secondary path; theta_p swept)
Headline = stratified AUT of recall-retention (theta_p-INDEPENDENT):
  AUT_mechanism vs AUT_artifact, per PREDICATE (clean class split) and per rule.
KM + log-rank over the two cohorts test whether mechanism outlives artifact.

Outputs: data/survival_{predicates,rules}.parquet, reports/survival_*.png/.csv
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from scipy.stats import mannwhitneyu

from config import DATA_DIR, FAMILIES, LABELS_CSV, REPORTS_DIR

FREEZE_Q = ["2018Q3", "2018Q4"]
TEST_Q = ["2019Q1", "2019Q2", "2019Q3", "2019Q4"]
RECALL_DEATH_FRAC = 0.5
THETA_P_DEFAULT = 0.90
THETA_SWEEP = [0.85, 0.90, 0.95]
MIN_FAMILY_Q = 20          # censor a family-quarter thinner than this


def load():
    lab = pd.read_csv(LABELS_CSV, parse_dates=["date"]).rename(
        columns={"classification_family": "family", "sha256": "sha"})
    lab["quarter"] = lab["date"].dt.to_period("Q").astype(str)
    lab = lab[lab.quarter.isin(FREEZE_Q + TEST_Q)]
    pruned = pd.read_parquet(DATA_DIR / "predicates_pruned.parquet",
                             columns=["sha", "pid", "quarter", "family"])
    stats = pd.read_parquet(DATA_DIR / "predicate_stats.parquet")
    rules = pd.read_parquet(DATA_DIR / "rules.parquet")
    return lab, pruned, stats, rules


# --------------------------------------------------------------------------
# scoring — shared by the predicate and rule tracks
# --------------------------------------------------------------------------
def score(target, cnt_by_qf, fam_q_size, theta_p=THETA_P_DEFAULT):
    """cnt_by_qf: {(quarter, family): #matched}. Returns a dict of survival stats."""
    tr_num = sum(cnt_by_qf.get((q, target), 0) for q in FREEZE_Q)
    tr_den = sum(fam_q_size.get((target, q), 0) for q in FREEZE_Q)
    train_recall = tr_num / tr_den if tr_den else np.nan

    rec, precf, ret = {}, {}, {}
    for q in TEST_Q:
        if fam_q_size.get((target, q), 0) < MIN_FAMILY_Q:
            rec[q] = precf[q] = ret[q] = np.nan
            continue
        fires = {}
        for g in FAMILIES:
            den = fam_q_size.get((g, q), 0)
            if den >= MIN_FAMILY_Q:
                fires[g] = cnt_by_qf.get((q, g), 0) / den
        rf = fires.get(target, 0.0)
        sf = sum(fires.values())
        rec[q] = rf
        precf[q] = rf / sf if sf > 0 else np.nan
        ret[q] = rf / train_recall if train_recall and train_recall > 0 else np.nan

    # AUT of recall-retention (clip to 1), trapezoid over observed quarters
    xs = [i for i, q in enumerate(TEST_Q) if not np.isnan(ret[q])]
    ys = [min(1.0, ret[TEST_Q[i]]) for i in xs]
    if len(ys) >= 2:
        aut = float(np.trapezoid(ys, xs) / (xs[-1] - xs[0]))
    elif len(ys) == 1:
        aut = ys[0]
    else:
        aut = np.nan

    # lifetime under theta_p: first quarter that dies on either path
    duration, event = 0, 0
    for i, q in enumerate(TEST_Q):
        if np.isnan(rec[q]):
            continue
        alive = (ret[q] >= RECALL_DEATH_FRAC) and (precf[q] >= theta_p)
        duration = i + 1
        if not alive:
            event = 1
            break
    out = dict(train_recall=train_recall, aut=aut, duration=duration, event=event)
    for q in TEST_Q:
        out[f"recall_{q}"] = rec[q]
        out[f"precf_{q}"] = precf[q]
    return out


def lifetime_under(cnt_by_qf, target, fam_q_size, theta_p):
    """recompute only (duration, event) for a theta_p sweep."""
    tr_num = sum(cnt_by_qf.get((q, target), 0) for q in FREEZE_Q)
    tr_den = sum(fam_q_size.get((target, q), 0) for q in FREEZE_Q)
    train_recall = tr_num / tr_den if tr_den else np.nan
    duration, event = 0, 0
    for i, q in enumerate(TEST_Q):
        if fam_q_size.get((target, q), 0) < MIN_FAMILY_Q or not train_recall:
            continue
        fires = {g: cnt_by_qf.get((q, g), 0) / fam_q_size[(g, q)]
                 for g in FAMILIES if fam_q_size.get((g, q), 0) >= MIN_FAMILY_Q}
        rf = fires.get(target, 0.0)
        sf = sum(fires.values())
        ret = rf / train_recall
        precf = rf / sf if sf > 0 else np.nan
        duration = i + 1
        if not (ret >= RECALL_DEATH_FRAC and (np.isnan(precf) or precf >= theta_p)):
            event = 1
            break
    return duration, event


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    lab, pruned, stats, rules = load()
    fam_q_size = {(f, q): n for (f, q), n in
                  lab.groupby(["family", "quarter"]).size().items()}

    # sha -> quarter/family via integer codes (light); pid -> set of scodes
    lab_idx = lab.set_index("sha")
    scode = {s: i for i, s in enumerate(pruned["sha"].unique())}
    pruned["sc"] = pruned["sha"].map(scode)
    sc2q = np.empty(len(scode), dtype=object)
    sc2f = np.empty(len(scode), dtype=object)
    for s, i in scode.items():
        sc2q[i] = lab_idx.at[s, "quarter"]
        sc2f[i] = lab_idx.at[s, "family"]

    # --- per-predicate matched counts: pid -> {(q,family): n} ---
    print("building per-predicate counts...")
    gp = pruned.groupby(["pid", "quarter", "family"], observed=True).size()
    cntmap: dict = {}
    for (pid, q, f), c in gp.items():
        cntmap.setdefault(pid, {})[(q, f)] = c
    pid2sc = {pid: set(g["sc"].values) for pid, g in pruned.groupby("pid")}

    meta = stats.set_index("pid")

    # ---------------- predicate track (primary) ----------------
    prows = []
    for pid, cbf in cntmap.items():
        if pid not in meta.index:
            continue
        m = meta.loc[pid]
        r = score(m["best_family"], cbf, fam_q_size)
        if np.isnan(r["aut"]):
            continue
        prows.append(dict(pid=pid, klass=m["klass"], field=m["field"],
                          target=m["best_family"], value=m["value"], **r))
    pred_df = pd.DataFrame(prows)
    pred_df.to_parquet(DATA_DIR / "survival_predicates.parquet", index=False)

    # ---------------- rule track (supporting) ----------------
    fv2pid = {(f, v): p for p, f, v in zip(meta.index, meta["field"], meta["value"])}
    rrows = []
    for _, rule in rules.iterrows():
        pids = [fv2pid.get((p["field"], p["value"])) for p in rule["predicates"]]
        if any(p is None for p in pids):
            continue
        matched = set.intersection(*(pid2sc[p] for p in pids))
        cbf: dict = {}
        for sc in matched:
            cbf[(sc2q[sc], sc2f[sc])] = cbf.get((sc2q[sc], sc2f[sc]), 0) + 1
        r = score(rule["family"], cbf, fam_q_size)
        if np.isnan(r["aut"]):
            continue
        rrows.append(dict(rule_id=rule["rule_id"], family=rule["family"],
                          composition=rule["composition"], n_artifact=rule["n_artifact"],
                          n_mechanism=rule["n_mechanism"], **r))
    rule_df = pd.DataFrame(rrows)
    rule_df.to_parquet(DATA_DIR / "survival_rules.parquet", index=False)

    _report(pred_df, rule_df, cntmap, meta, fam_q_size)


def _report(pred_df, rule_df, cntmap, meta, fam_q_size):
    print("\n" + "=" * 60)
    print("PER-PREDICATE SURVIVAL (primary class contrast)")
    print("=" * 60)
    for k in ["mechanism", "artifact"]:
        d = pred_df[pred_df.klass == k]
        print(f"  {k:10} n={len(d):5}  AUT(recall-retention) "
              f"mean={d.aut.mean():.3f} median={d.aut.median():.3f}  "
              f"median lifetime={d.duration.median():.1f}q")
    a = pred_df[pred_df.klass == "artifact"]["aut"]
    m = pred_df[pred_df.klass == "mechanism"]["aut"]
    if len(a) and len(m):
        u, p = mannwhitneyu(m, a, alternative="greater")
        print(f"  Mann-Whitney (mechanism AUT > artifact AUT): p={p:.2e}")

    # theta_p sweep — KM log-rank on predicate cohorts
    print("\ntheta_p sweep (KM log-rank, mechanism vs artifact predicates):")
    for th in THETA_SWEEP:
        durs, evs, kl = [], [], []
        for pid, cbf in cntmap.items():
            if pid not in meta.index:
                continue
            d, e = lifetime_under(cbf, meta.at[pid, "best_family"], fam_q_size, th)
            if d == 0:
                continue
            durs.append(d); evs.append(e); kl.append(meta.at[pid, "klass"])
        s = pd.DataFrame(dict(dur=durs, ev=evs, kl=kl))
        A, M = s[s.kl == "artifact"], s[s.kl == "mechanism"]
        lr = logrank_test(M.dur, A.dur, M.ev, A.ev)
        print(f"  theta_p={th}: median lifetime  mech={M.dur.median():.1f}q  "
              f"arti={A.dur.median():.1f}q   log-rank p={lr.p_value:.2e}")

    print("\n" + "=" * 60)
    print("PER-RULE SURVIVAL (by composition)")
    print("=" * 60)
    for c in ["mechanism_only", "mixed", "artifact_only"]:
        d = rule_df[rule_df.composition == c]
        if len(d):
            print(f"  {c:15} n={len(d):4}  AUT mean={d.aut.mean():.3f}  "
                  f"median lifetime={d.duration.median():.1f}q")

    _plots(pred_df)
    print("\nwrote data/survival_{predicates,rules}.parquet + reports/survival_*.png")


def _plots(pred_df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # KM curves
    fig, ax = plt.subplots(figsize=(6, 4))
    km = KaplanMeierFitter()
    for k, c in [("mechanism", "tab:blue"), ("artifact", "tab:red")]:
        d = pred_df[pred_df.klass == k]
        km.fit(d.duration, d.event, label=f"{k} (n={len(d)})")
        km.plot_survival_function(ax=ax, color=c, ci_show=True)
    ax.set_xlabel("quarters survived"); ax.set_ylabel("fraction still alive")
    ax.set_title("Per-predicate survival: mechanism vs artifact")
    fig.tight_layout(); fig.savefig(REPORTS_DIR / "survival_km.png", dpi=130)

    # AUT distributions
    fig, ax = plt.subplots(figsize=(6, 4))
    for k, c in [("mechanism", "tab:blue"), ("artifact", "tab:red")]:
        ax.hist(pred_df[pred_df.klass == k]["aut"], bins=20, alpha=0.5,
                density=True, label=k, color=c)
    ax.set_xlabel("AUT (recall-retention)"); ax.set_ylabel("density")
    ax.set_title("Predicate AUT by class"); ax.legend()
    fig.tight_layout(); fig.savefig(REPORTS_DIR / "survival_aut.png", dpi=130)


if __name__ == "__main__":
    main()
