"""Option 1: the FAIR per-rule class comparison.

The raw per-rule table (AUT by composition) is confounded: a rule dies if ANY of
its predicates dies, and which predicates the miner bundled + which family it
belongs to drive the averages more than class does. Here we control for both:

  1. Fixed-effects OLS:  aut ~ artifact_fraction + C(family) + C(n_predicates)
     -> holding family and rule-size constant, does more artifact content predict
        faster decay? (negative coefficient on artifact_fraction = yes.)
  2. Family-level sign test on 2-predicate rules only (conservative; treats each
     family as one independent observation).

Caveat printed: rules within a family are near-duplicates, so the OLS p-value is
anti-conservative; the coefficient sign/size and the family sign test are the
robust reads.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import wilcoxon

from config import DATA_DIR


def main() -> None:
    r = pd.read_parquet(DATA_DIR / "survival_rules.parquet")
    r["n_predicates"] = r["n_artifact"] + r["n_mechanism"]
    r["artifact_fraction"] = r["n_artifact"] / r["n_predicates"]

    print("rules:", len(r), "| size distribution:",
          r["n_predicates"].value_counts().sort_index().to_dict())

    # ---------- raw (confounded) picture, for contrast ----------
    print("\n--- RAW (confounded) mean AUT by composition ---")
    print(r.groupby("composition", observed=True)["aut"].agg(["mean", "count"]).round(3).to_string())

    # ---------- 1. fixed-effects regression ----------
    print("\n--- CONTROLLED: aut ~ artifact_fraction + C(family) + C(n_predicates) ---")
    model = smf.ols("aut ~ artifact_fraction + C(family) + C(n_predicates)", data=r).fit()
    coef = model.params["artifact_fraction"]
    pval = model.pvalues["artifact_fraction"]
    ci = model.conf_int().loc["artifact_fraction"]
    print(f"  artifact_fraction coef = {coef:+.3f}  (95% CI {ci[0]:+.3f}..{ci[1]:+.3f})  p={pval:.2e}")
    print(f"  reading: holding family+size fixed, going from 0% to 100% artifact")
    print(f"           predicts a {coef:+.2f} change in AUT (negative = artifacts decay faster)")
    print("  [caveat] near-duplicate rules within a family -> p is anti-conservative")

    # ---------- 2. conservative family-level sign test (2-predicate rules) ----------
    print("\n--- CONTROLLED: 2-predicate rules only, per family ---")
    two = r[r["n_predicates"] == 2]
    rows = []
    for fam, g in two.groupby("family", observed=True):
        mech = g[g.artifact_fraction == 0]["aut"]
        arti = g[g.artifact_fraction > 0]["aut"]
        if len(mech) >= 3 and len(arti) >= 3:
            rows.append(dict(family=fam, mech_aut=mech.mean(), arti_aut=arti.mean(),
                             diff=mech.mean() - arti.mean(),
                             n_mech=len(mech), n_arti=len(arti)))
    cmp = pd.DataFrame(rows)
    if not cmp.empty:
        print(cmp.round(3).to_string(index=False))
        favored_mech = int((cmp["diff"] > 0).sum())
        print(f"\n  families where mechanism rules survive longer: {favored_mech}/{len(cmp)}")
        if len(cmp) >= 5:
            try:
                w = wilcoxon(cmp["mech_aut"], cmp["arti_aut"])
                print(f"  Wilcoxon signed-rank across families: p={w.pvalue:.3f}")
            except ValueError as e:
                print(f"  (Wilcoxon not run: {e})")
    else:
        print("  too few families have both classes among 2-predicate rules for the sign test")

    _verdict(coef, pval, cmp if not cmp.empty else None)


def _verdict(coef, pval, cmp) -> None:
    print("\n" + "=" * 60)
    if coef < -0.03 and pval < 0.05:
        print("VERDICT: rule-level signal SURVIVES the fair comparison —")
        print("  more artifact content predicts faster decay even within family+size.")
    elif abs(coef) <= 0.05:
        print("VERDICT: rule-level class signal WASHES OUT once family+size are")
        print("  controlled (as expected: rules only contain the durable, reusable")
        print("  artifacts, and a rule inherits its weakest predicate). The clean")
        print("  test is per-PREDICATE, which shows mechanism > artifact.")
    else:
        print(f"VERDICT: mixed — coef {coef:+.3f}, p={pval:.2e}; inspect the per-family table.")
    print("=" * 60)


if __name__ == "__main__":
    main()
