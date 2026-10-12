"""UCI Heart Disease sliced by sex: the fairness audit and the mitigation.

The data file is the one of the robustness lab
(`labs/robustness/data/heart_disease_cleveland.csv`, UCI, CC BY 4.0), the
XGBoost recipe is the shared one. Two splits (decisions of the author,
24 September 2026):

    audit        75 / 25 %, stratified, seed 42 - the split of the
                 explainability lecture, so the audited model is the one that
                 was explained; repeated over 20 seeds to show the spread
    mitigation   50 / 25 / 25 % - train, calibration (the thresholds are
                 fitted here), test (touched once) - as in robustness

The fairness metrics are written by hand; the mitigation is `fairlearn`'s
`ThresholdOptimizer` (equalized odds, and demographic parity as the contrast).

Run as a script to print the reference numbers:

    uv run python lectures/bias-fairness/code/heart.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split

DATA = Path(__file__).resolve().parents[3] / "labs" / "robustness" / "data"
CSV = DATA / "heart_disease_cleveland.csv"
SEED = 42
N_SPLITS = 20
N_DRAWS = 200                          # draws of the randomized ThresholdOptimizer
OTHER_DRAW = 6                         # a second draw, shown next to random_state = SEED
GROUPS = {0: "female", 1: "male"}      # the `sex` column: 1 = male, 0 = female


def load_heart(path: Path = CSV) -> tuple[pd.DataFrame, pd.Series]:
    """13 features (missing values filled with the median) and disease = num > 0."""
    table = pd.read_csv(path)
    y = (table.pop("num") > 0).astype(int)
    return table.fillna(table.median(numeric_only=True)), y


class XGBClassifier64(xgb.XGBClassifier):
    """XGBoost whose predict_proba returns float64: `fairlearn`'s
    ThresholdOptimizer writes its interpolated probabilities into an array of
    the model's output dtype and fails on XGBoost's float32. The numbers are
    the same."""

    def predict_proba(self, X, *args, **kwargs):
        return super().predict_proba(X, *args, **kwargs).astype(np.float64)


def train_xgboost(X_train: pd.DataFrame, y_train: pd.Series, seed: int = SEED):
    """The shared recipe of the tabular running example."""
    model = XGBClassifier64(n_estimators=200, max_depth=3, learning_rate=0.1,
                            eval_metric="logloss", random_state=seed)
    return model.fit(X_train, y_train)


def group_metrics(y_true, y_pred, sex) -> pd.DataFrame:
    """Per group: size, number of sick patients, base rate, selection rate,
    TPR, FPR and PPV of the thresholded predictions."""
    y_true, y_pred, sex = np.asarray(y_true), np.asarray(y_pred), np.asarray(sex)
    rows = {}
    for code, name in GROUPS.items():
        g = sex == code
        t, p = y_true[g], y_pred[g]
        rows[name] = {
            "n": int(g.sum()),
            "sick": int(t.sum()),
            "base rate": t.mean(),
            "selection rate": p.mean(),
            "TPR": p[t == 1].mean() if (t == 1).any() else np.nan,
            "FPR": p[t == 0].mean() if (t == 0).any() else np.nan,
            "PPV": t[p == 1].mean() if (p == 1).any() else np.nan,
        }
    return pd.DataFrame(rows).T


def gaps(metrics: pd.DataFrame) -> dict:
    """Male minus female, for the four rates; the demographic-parity gap is
    the selection-rate difference, the equalized-odds gap the larger of the
    TPR and FPR differences (in absolute value)."""
    d = metrics.loc["male"] - metrics.loc["female"]
    return {"selection rate": d["selection rate"], "TPR": d["TPR"], "FPR": d["FPR"],
            "PPV": d["PPV"], "demographic parity": abs(d["selection rate"]),
            "equalized odds": max(abs(d["TPR"]), abs(d["FPR"]))}


def audit(seed: int = SEED) -> dict:
    """The explainability split (75 / 25) and model, sliced by sex."""
    X, y = load_heart()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
                                                        random_state=seed, stratify=y)
    model = train_xgboost(X_train, y_train, seed)
    y_pred = model.predict(X_test)
    metrics = group_metrics(y_test, y_pred, X_test["sex"])
    return {"metrics": metrics, "gaps": gaps(metrics),
            "accuracy": float((y_pred == y_test.to_numpy()).mean())}


def repeated_audit(n_splits: int = N_SPLITS) -> pd.DataFrame:
    """The audit on seeds 0 ... 19: one row per split and group."""
    rows = []
    for seed in range(n_splits):
        m = audit(seed)["metrics"]
        for group, values in m.iterrows():
            rows.append({"seed": seed, "group": group, **values.to_dict()})
    return pd.DataFrame(rows)


def mitigation_split(seed: int = SEED):
    """Train / calibration / test = 50 / 25 / 25 %, stratified."""
    X, y = load_heart()
    X_train, X_rest, y_train, y_rest = train_test_split(X, y, test_size=0.5,
                                                        random_state=seed, stratify=y)
    X_calib, X_test, y_calib, y_test = train_test_split(X_rest, y_rest, test_size=0.5,
                                                        random_state=seed, stratify=y_rest)
    return X_train, X_calib, X_test, y_train, y_calib, y_test


def mitigate(model, X_calib, y_calib, constraint: str, seed: int = SEED):
    """`fairlearn`'s ThresholdOptimizer on a fitted model: per-group
    (randomized) thresholds on the predicted probability, fitted on the
    calibration part so that the constraint holds there."""
    from fairlearn.postprocessing import ThresholdOptimizer
    optimizer = ThresholdOptimizer(estimator=model, constraints=constraint,
                                   predict_method="predict_proba", prefit=True)
    return optimizer.fit(X_calib, y_calib, sensitive_features=X_calib["sex"])


def mitigation_study(seed: int = SEED) -> dict:
    """The unmitigated model and the two mitigations, judged on the test part."""
    X_train, X_calib, X_test, y_train, y_calib, y_test = mitigation_split(seed)
    model = train_xgboost(X_train, y_train, seed)
    predictions = {"unmitigated": model.predict(X_test)}
    for constraint in ("equalized_odds", "demographic_parity"):
        optimizer = mitigate(model, X_calib, y_calib, constraint, seed)
        predictions[constraint] = np.asarray(
            optimizer.predict(X_test, sensitive_features=X_test["sex"], random_state=seed))
    result = {}
    for name, y_pred in predictions.items():
        metrics = group_metrics(y_test, y_pred, X_test["sex"])
        result[name] = {"metrics": metrics, "gaps": gaps(metrics),
                        "accuracy": float((y_pred == y_test.to_numpy()).mean())}
    return result


def mitigation_draws(seed: int = SEED, n_draws: int = N_DRAWS) -> pd.DataFrame:
    """The ThresholdOptimizer flags a patient between its two thresholds with
    a probability, so every `predict` is a draw. The two mitigations of
    `mitigation_study` on the test part for random_state 0 ... n_draws - 1:
    one row per constraint, draw and group."""
    X_train, X_calib, X_test, y_train, y_calib, y_test = mitigation_split(seed)
    model = train_xgboost(X_train, y_train, seed)
    rows = []
    for constraint in ("equalized_odds", "demographic_parity"):
        optimizer = mitigate(model, X_calib, y_calib, constraint, seed)
        for draw in range(n_draws):
            y_draw = np.asarray(optimizer.predict(X_test, sensitive_features=X_test["sex"],
                                                  random_state=draw))
            m = group_metrics(y_test, y_draw, X_test["sex"])
            g = gaps(m)
            for group, values in m.iterrows():
                rows.append({"constraint": constraint, "draw": draw, "group": group,
                             "accuracy": float((y_draw == y_test.to_numpy()).mean()),
                             "equalized-odds gap": g["equalized odds"],
                             "demographic-parity gap": g["demographic parity"],
                             **values.to_dict()})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    X, y = load_heart()
    for code, name in GROUPS.items():
        print(f"{name}: {int((X['sex'] == code).sum())} patients, "
              f"{y[X['sex'] == code].mean():.3f} sick")
    a = audit()
    print(f"\naudit (75 / 25, seed 42): accuracy {a['accuracy']:.3f}")
    print(a["metrics"].round(3))
    print({k: round(v, 3) for k, v in a["gaps"].items()})
    r = repeated_audit()
    print(f"\n{N_SPLITS} splits, median and 10th-90th percentile:")
    for metric in ("selection rate", "TPR", "FPR", "PPV"):
        for group in ("female", "male"):
            v = r.loc[r["group"] == group, metric]
            print(f"  {metric:15s} {group:6s} {v.median():.3f} "
                  f"[{v.quantile(0.1):.3f}, {v.quantile(0.9):.3f}]")
        d = (r[r.group == "male"][metric].to_numpy() - r[r.group == "female"][metric].to_numpy())
        print(f"  {metric:15s} male - female: median {np.median(d):+.3f}, "
              f"positive in {(d > 0).sum()} of {N_SPLITS}")
    m = mitigation_study()
    print("\nmitigation (50 / 25 / 25, seed 42):")
    for name, v in m.items():
        print(f"  {name:18s} accuracy {v['accuracy']:.3f}  "
              + "  ".join(f"{k} {g:+.3f}" if k in ("selection rate", "TPR", "FPR") else f"{k} {g:.3f}"
                          for k, g in v["gaps"].items() if k != "PPV"))
        print(v["metrics"][["n", "sick", "selection rate", "TPR", "FPR"]].round(3).to_string())
    d = mitigation_draws()
    print(f"\nthe same mitigations, {N_DRAWS} draws (random_state 0 ... {N_DRAWS - 1}):")
    for constraint in ("equalized_odds", "demographic_parity"):
        c = d[d.constraint == constraint]
        for label, rows in ((f"draw {SEED}", c[c.draw == SEED]), (f"draw {OTHER_DRAW}", c[c.draw == OTHER_DRAW]),
                            ("mean", c), ("10th pct", None), ("90th pct", None)):
            if rows is None:
                q = 0.1 if label == "10th pct" else 0.9
                v = c.groupby("group")[["TPR", "FPR"]].quantile(q)
                per_draw = c[c.group == "female"][["accuracy", "equalized-odds gap", "demographic-parity gap"]].quantile(q)
            else:
                v = rows.groupby("group")[["TPR", "FPR"]].mean()
                per_draw = rows[rows.group == "female"][["accuracy", "equalized-odds gap",
                                                         "demographic-parity gap"]].mean()
            print(f"  {constraint:18s} {label:9s} TPR {v.loc['female', 'TPR']:.3f} / {v.loc['male', 'TPR']:.3f}  "
                  f"FPR {v.loc['female', 'FPR']:.3f} / {v.loc['male', 'FPR']:.3f}  "
                  f"accuracy {per_draw['accuracy']:.3f}  EO gap {per_draw['equalized-odds gap']:.3f}  "
                  f"DP gap {per_draw['demographic-parity gap']:.3f}")
        women = c[c.group == "female"]
        print(f"  {constraint:18s} women's TPR below 0.714 in {(women.TPR < 0.71).sum()} of {N_DRAWS} draws")
    # The leak of the MPA-PRM slide (question 3 of the lab's part 2): the
    # thresholds fitted on the test part and judged on it.
    X_train, X_calib, X_test, y_train, y_calib, y_test = mitigation_split()
    model = train_xgboost(X_train, y_train)
    print("\nleaked - fitted on the test part and judged on it:")
    for constraint in ("equalized_odds", "demographic_parity"):
        y_pred = np.asarray(mitigate(model, X_test, y_test, constraint).predict(
            X_test, sensitive_features=X_test["sex"], random_state=SEED))
        g = gaps(group_metrics(y_test, y_pred, X_test["sex"]))
        print(f"  {constraint:18s} accuracy {(y_pred == y_test.to_numpy()).mean():.3f}  "
              f"demographic parity {g['demographic parity']:.3f}  equalized odds {g['equalized odds']:.3f}")
