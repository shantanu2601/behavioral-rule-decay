"""Drift gate — the Week-2 kill switch (build step 3).

Premise of the whole project: malware BEHAVIOR drifts over time, so rules learned
on old samples decay on newer ones. Verify that cheaply before mining anything.

Train a LightGBM family classifier on an EARLY, time-ordered slice; test on strictly
LATER quarters. If future-quarter performance craters vs. a same-period holdout ->
drift is real, project is alive. If it barely drops -> no drift -> stop / pivot.

TESSERACT C1 enforced: every training sample predates every test sample (asserted).
Feature vocabulary is selected on TRAIN ONLY (no temporal leakage).

Outputs: printed verdict + reports/drift_gate_curve.csv + reports/drift_gate.png
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import accuracy_score, f1_score

import lightgbm as lgb

from config import DATA_DIR, FAMILIES, LABELS_CSV, REPORTS_DIR, SEED

TRAIN_Q = ["2018Q3", "2018Q4"]                 # early slice (all 10 families present)
EVAL_Q = ["2019Q1", "2019Q2", "2019Q3", "2019Q4"]  # strictly later quarters
MIN_DF = 20                                    # predicate must hit >=20 TRAIN samples


def load_samples() -> pd.DataFrame:
    df = pd.read_csv(LABELS_CSV, parse_dates=["date"]).rename(
        columns={"classification_family": "family"})
    df["quarter"] = df["date"].dt.to_period("Q").astype(str)
    df = df[df["quarter"].isin(TRAIN_Q + EVAL_Q)].copy()
    return df[["sha256", "family", "date", "quarter"]].rename(columns={"sha256": "sha"})


def build_pid(preds: pd.DataFrame) -> np.ndarray:
    """Unique int id per (field, value) with no giant string concat."""
    fcodes = preds["field"].astype("category").cat.codes.to_numpy()
    vcodes, vuniq = pd.factorize(preds["value"])
    n_fields = int(fcodes.max()) + 1
    return vcodes.astype(np.int64) * n_fields + fcodes


def main() -> None:
    rng = np.random.default_rng(SEED)
    samples = load_samples()
    # temporal constraint (C1): train strictly before test
    tmax = samples.loc[samples.quarter.isin(TRAIN_Q), "date"].max()
    tmin = samples.loc[samples.quarter.isin(EVAL_Q), "date"].min()
    assert tmax < tmin, f"C1 violated: train max {tmax} !< test min {tmin}"
    print(f"C1 ok: train ends {tmax.date()} < test starts {tmin.date()}")

    # sample row index
    samples = samples.reset_index(drop=True)
    sha2row = {s: i for i, s in enumerate(samples["sha"])}
    n = len(samples)
    y = samples["family"].astype(pd.CategoricalDtype(FAMILIES)).cat.codes.to_numpy()
    print(f"usable samples: {n:,}  (train {samples.quarter.isin(TRAIN_Q).sum():,}, "
          f"eval {samples.quarter.isin(EVAL_Q).sum():,})")

    # predicates -> rows for usable samples only
    preds = pd.read_parquet(DATA_DIR / "predicates.parquet", columns=["sha", "field", "value"])
    preds = preds[preds["sha"].isin(sha2row)].copy()
    preds["pid"] = build_pid(preds)
    preds["row"] = preds["sha"].map(sha2row).to_numpy()

    # vocab from TRAIN samples only (no leakage)
    train_rows = np.where(samples.quarter.isin(TRAIN_Q).to_numpy())[0]
    is_train = preds["row"].isin(set(train_rows.tolist()))
    df_counts = preds.loc[is_train, "pid"].value_counts()
    vocab = df_counts[df_counts >= MIN_DF].index.to_numpy()
    pid2col = -np.ones(int(preds["pid"].max()) + 1, dtype=np.int64)
    pid2col[vocab] = np.arange(len(vocab))
    print(f"features (train df>={MIN_DF}): {len(vocab):,}")

    # sparse binary sample x feature matrix
    col = pid2col[preds["pid"].to_numpy()]
    keep = col >= 0
    X = sparse.coo_matrix(
        (np.ones(keep.sum(), dtype=np.float32),
         (preds["row"].to_numpy()[keep], col[keep])),
        shape=(n, len(vocab)),
    ).tocsr()

    # split: train (minus 20% same-period holdout) / holdout / each eval quarter
    q = samples["quarter"].to_numpy()
    tr_all = np.where(np.isin(q, TRAIN_Q))[0]
    rng.shuffle(tr_all)
    cut = int(0.8 * len(tr_all))
    tr, hold = tr_all[:cut], tr_all[cut:]

    params = dict(objective="multiclass", num_class=len(FAMILIES), metric="multi_logloss",
                  learning_rate=0.1, num_leaves=63, min_data_in_leaf=20,
                  feature_fraction=0.6, bagging_fraction=0.8, bagging_freq=1,
                  seed=SEED, verbose=-1)
    dtrain = lgb.Dataset(X[tr], label=y[tr])
    model = lgb.train(params, dtrain, num_boost_round=300)

    def macro_f1(idx):
        pred = model.predict(X[idx]).argmax(1)
        return f1_score(y[idx], pred, average="macro"), accuracy_score(y[idx], pred)

    hold_f1, hold_acc = macro_f1(hold)
    print(f"\nsame-period holdout (no-drift reference):  macroF1={hold_f1:.3f}  acc={hold_acc:.3f}")
    print("future quarters (drift shows here):")
    curve = [("holdout", hold_f1, hold_acc, len(hold))]
    for qq in EVAL_Q:
        idx = np.where(q == qq)[0]
        f1, acc = macro_f1(idx)
        curve.append((qq, f1, acc, len(idx)))
        print(f"  {qq}:  macroF1={f1:.3f}  acc={acc:.3f}  (n={len(idx):,})")

    cdf = pd.DataFrame(curve, columns=["slice", "macro_f1", "accuracy", "n"])
    cdf.to_csv(REPORTS_DIR / "drift_gate_curve.csv", index=False)

    last_f1 = curve[-1][1]
    drop = hold_f1 - last_f1
    rel = drop / hold_f1 if hold_f1 else 0
    print(f"\nF1 drop holdout -> {EVAL_Q[-1]}: {drop:+.3f} ({rel:+.0%})")
    verdict = ("DRIFT PRESENT -> project premise holds" if rel >= 0.10
               else "NO MEANINGFUL DRIFT -> reconsider / pivot")
    print(f"VERDICT: {verdict}")

    _plot(cdf)
    print(f"wrote reports/drift_gate_curve.csv and reports/drift_gate.png")


def _plot(cdf: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(cdf["slice"], cdf["macro_f1"], "o-", label="macro F1")
    ax.plot(cdf["slice"], cdf["accuracy"], "s--", alpha=0.6, label="accuracy")
    ax.axvspan(-0.4, 0.4, color="tab:green", alpha=0.08)
    ax.set_ylim(0, 1)
    ax.set_ylabel("score")
    ax.set_title("Drift gate: trained on 2018Q3–Q4, tested on later quarters")
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "drift_gate.png", dpi=130)


if __name__ == "__main__":
    main()
