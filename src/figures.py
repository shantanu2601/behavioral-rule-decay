"""Regenerate all publication figures from the final artifacts.

    python src/figures.py

Writes reports/fig_*.png. Academic style: colourblind-safe (Wong) palette,
serif, minimal spines, greyscale-legible.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter

from config import DATA_DIR, REPORTS_DIR

MECH, ARTI, ACC = "#0072B2", "#D55E00", "#666666"   # blue, vermillion, grey
QLAB = ["2019Q1", "2019Q2", "2019Q3", "2019Q4"]
QCOL = [f"recall_{q}" for q in QLAB]

plt.rcParams.update({
    "figure.dpi": 200, "savefig.dpi": 200, "font.size": 10,
    "font.family": "serif", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "legend.frameon": False, "figure.constrained_layout.use": True,
})


def _save(fig, name):
    p = REPORTS_DIR / name
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p)


def fig_drift():
    d = pd.read_csv(REPORTS_DIR / "drift_gate_curve.csv")
    test = d[d["slice"] != "holdout"]
    hold = d[d["slice"] == "holdout"]["macro_f1"].iloc[0]
    fig, ax = plt.subplots(figsize=(4.3, 3.0))
    ax.axhline(hold, ls="--", c=ACC, lw=1, label=f"same-period holdout ({hold:.2f})")
    ax.plot(test["slice"], test["macro_f1"], "o-", c=MECH, lw=1.6, label="macro-F1")
    ax.plot(test["slice"], test["accuracy"], "s--", c=ARTI, lw=1.2, label="accuracy")
    ax.set_ylim(0.75, 1.0)
    ax.set_ylabel("score"); ax.set_xlabel("test quarter (train = 2018Q3–Q4)")
    ax.legend(loc="lower left", fontsize=8)
    _save(fig, "fig_drift.png")


def fig_aut_dist(sp):
    fig, ax = plt.subplots(figsize=(4.3, 3.0))
    data = [sp[sp.klass == "mechanism"].aut.values, sp[sp.klass == "artifact"].aut.values]
    parts = ax.violinplot(data, showmedians=True, showextrema=False, widths=0.85)
    for b, c in zip(parts["bodies"], (MECH, ARTI)):
        b.set_facecolor(c); b.set_alpha(0.45); b.set_edgecolor(c)
    parts["cmedians"].set_color("black"); parts["cmedians"].set_linewidth(1.4)
    for i, arr in enumerate(data, 1):
        ax.text(i, np.median(arr) + 0.03, f"median {np.median(arr):.2f}",
                ha="center", fontsize=8)
    ax.set_xticks([1, 2]); ax.set_xticklabels(
        [f"mechanism\n(n={len(data[0])})", f"artifact\n(n={len(data[1])})"])
    ax.set_ylabel("AUT of recall-retention"); ax.set_ylim(-0.02, 1.05)
    ax.set_title("Per-predicate durability  (p = 2.7×10⁻³⁹)", fontsize=9)
    _save(fig, "fig_aut_dist.png")


def fig_km(sp):
    fig, ax = plt.subplots(figsize=(4.3, 3.0))
    kmf = KaplanMeierFitter()
    for klass, c in [("mechanism", MECH), ("artifact", ARTI)]:
        g = sp[sp.klass == klass]
        kmf.fit(g.duration, g.event, label=f"{klass} (n={len(g)})")
        kmf.plot_survival_function(ax=ax, ci_show=False, color=c, lw=1.8)
    ax.set_xlabel("quarters survived"); ax.set_ylabel("surviving fraction")
    ax.set_ylim(0, 1.02); ax.legend(loc="upper right", fontsize=8)
    _save(fig, "fig_km.png")


def fig_family(sp):
    rows = []
    for fam, g in sp.groupby("target"):
        m = g[g.klass == "mechanism"].aut.median()
        a = g[g.klass == "artifact"].aut
        rows.append((fam, m, a.median() if len(a) else np.nan))
    t = pd.DataFrame(rows, columns=["fam", "mech", "arti"]).sort_values("mech")
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    y = np.arange(len(t)); h = 0.38
    ax.barh(y + h / 2, t["mech"], h, color=MECH, label="mechanism")
    ax.barh(y - h / 2, t["arti"].fillna(0), h, color=ARTI, label="artifact")
    ax.set_yticks(y); ax.set_yticklabels(t["fam"])
    ax.set_xlabel("median predicate AUT"); ax.set_xlim(0, 1.0)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title("Durability by family and class", fontsize=9)
    _save(fig, "fig_family.png")


def fig_examples(sr):
    # (rule_id, label, colour, marker, linestyle) — cool = mechanism, warm = artifact,
    # but each rule has its own hue + marker so all four are distinguishable.
    picks = [
        ("qakbot_161", "Qakbot — mechanism rule", "#0072B2", "o", "-"),
        ("emotet_52",  "Emotet — mechanism rule", "#56B4E9", "s", "-"),
        ("emotet_0",   "Emotet — artifact rule (never rotated)", "#E69F00", "^", "--"),
        ("ursnif_421", "Ursnif — artifact rule (rotated away)", "#D55E00", "D", "-"),
    ]
    fig, ax = plt.subplots(figsize=(5.6, 3.5))
    for rid, lab, c, mk, ls in picks:
        r = sr[sr.rule_id == rid]
        if not len(r):
            print("missing", rid); continue
        y = r[QCOL].iloc[0].values.astype(float)
        ax.plot(QLAB, y, ls, marker=mk, color=c, lw=1.8, ms=7, label=lab)
    ax.set_ylabel("recall on target family"); ax.set_ylim(-0.03, 1.05)
    ax.set_xlabel("quarter (rules frozen on 2018Q3–Q4)")
    ax.legend(loc="lower left", fontsize=7.2)
    ax.set_title("Four frozen rules over two years", fontsize=9)
    _save(fig, "fig_examples.png")


def fig_rules():
    """Render a mechanism rule and an artifact rule in a compact rule notation,
    class tags colour-coded. Values are taken verbatim from data/rules.parquet
    (emotet_52, ursnif_421)."""
    from matplotlib.patches import FancyBboxPatch

    cards = [
        ("emotet_process_injection", "Emotet", MECH, [
            ("api_seq", "mechanism",
             "NtQueryAttributesFile → NtQueryFullAttributesFile → LdrLoadDll"),
            ("file", "mechanism", "drop_location = %LOCALAPPDATA%\\*\\"),
            ("api_seq", "mechanism",
             "OpenSCManagerW → GetSystemTimeAsFileTime → SHGetFolderPathW"),
        ]),
        ("ursnif_dropped_marker", "Ursnif", ARTI, [
            ("file", "artifact", 'dropped_name = "msl0"'),
            ("file", "artifact", "path = \\??\\mailslot\\"),
        ]),
    ]

    fig, axes = plt.subplots(2, 1, figsize=(8.2, 3.6),
                             gridspec_kw={"height_ratios": [3, 2]})
    for ax, (name, fam, accent, preds) in zip(axes, cards):
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
        ax.add_patch(FancyBboxPatch(
            (0.006, 0.04), 0.988, 0.92, boxstyle="round,pad=0.004,rounding_size=0.02",
            fc="#f6f6f6", ec=accent, lw=1.6, mutation_aspect=0.3))
        ax.add_patch(plt.Rectangle((0.006, 0.04), 0.010, 0.92, fc=accent, ec="none"))
        mono = {"family": "monospace"}
        ax.text(0.040, 0.86, f"rule {name}", fontweight="bold", fontsize=10, **mono)
        ax.text(0.986, 0.86, f"family = {fam}", ha="right", fontsize=9,
                color="#333", **mono)
        ax.text(0.040, 0.66, "all_of:", fontsize=8, color="#555", **mono)
        ys = np.linspace(0.50, 0.14, len(preds))
        for (field, klass, match), y in zip(preds, ys):
            ax.text(0.070, y, f"{field:<8}", fontsize=8, color="#444", **mono)
            ax.text(0.215, y, f"[{klass}]", fontsize=8, color=accent,
                    fontweight="bold", **mono)
            ax.text(0.350, y, match, fontsize=7.3, color="#111", **mono)
    _save(fig, "fig_rules.png")


def main():
    fig_rules()
    sp = pd.read_parquet(DATA_DIR / "survival_predicates.parquet")
    sr = pd.read_parquet(DATA_DIR / "survival_rules.parquet")
    fig_drift()
    fig_aut_dist(sp)
    fig_km(sp)
    fig_family(sp)
    fig_examples(sr)


if __name__ == "__main__":
    main()
