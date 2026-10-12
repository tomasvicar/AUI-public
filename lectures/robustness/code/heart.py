"""UCI Heart Disease and an XGBoost classifier: the tabular running example.

Calibration (ECE, Brier score, reliability diagram, temperature scaling) and
monitoring (PSI and the Kolmogorov-Smirnov test over weekly batches) are
written by hand in NumPy/SciPy - no `netcal`, no `evidently` (decision of the
author, 24 September 2026). The lecture figures and the lab call the same
functions.

One split for everything: 50 % train, 25 % calibration, 25 % test,
stratified, seed 42 - the temperature is fitted on the calibration part and
judged on the test part, never on the data it was fitted on.

Run as a script to print the reference numbers:

    uv run python lectures/robustness/code/heart.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.optimize import minimize_scalar
from scipy.stats import ks_2samp
from sklearn.model_selection import train_test_split

DATA = Path(__file__).resolve().parents[3] / "labs" / "robustness" / "data"
CSV = DATA / "heart_disease_cleveland.csv"
# The same data set was collected in four hospitals; the model is trained in
# Cleveland only, the other three are "a different hospital" on our own data.
SITES = {"Cleveland": "heart_disease_cleveland.csv",
         "Hungary": "heart_disease_hungarian.csv",
         "Switzerland": "heart_disease_switzerland.csv",
         "VA Long Beach": "heart_disease_va_long_beach.csv"}
OTHER_SITES = ["Hungary", "Switzerland", "VA Long Beach"]
SEED = 42
N_BINS = 5               # 76 test patients: ten bins would hold a handful each


def load_heart(path: Path = CSV) -> tuple[pd.DataFrame, pd.Series]:
    """13 features (missing values filled with the median) and disease = num > 0."""
    table = pd.read_csv(path)
    y = (table.pop("num") > 0).astype(int)
    return table.fillna(table.median(numeric_only=True)), y


def split(X: pd.DataFrame, y: pd.Series, seed: int = SEED):
    """Train / calibration / test = 50 / 25 / 25 %, stratified."""
    X_train, X_rest, y_train, y_rest = train_test_split(
        X, y, test_size=0.5, random_state=seed, stratify=y)
    X_calib, X_test, y_calib, y_test = train_test_split(
        X_rest, y_rest, test_size=0.5, random_state=seed, stratify=y_rest)
    return X_train, X_calib, X_test, y_train, y_calib, y_test


def train_xgboost(X_train: pd.DataFrame, y_train: pd.Series, seed: int = SEED):
    """The shared recipe of the tabular running example."""
    model = xgb.XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.1,
                              eval_metric="logloss", random_state=seed)
    return model.fit(X_train, y_train)


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def load_site_raw(site: str) -> tuple[pd.DataFrame, pd.Series]:
    """One hospital as recorded: missing values left as NaN. A cholesterol or a
    resting blood pressure of 0 means "not measured" (all of Switzerland, 49
    VA patients) and becomes NaN as well."""
    table = pd.read_csv(DATA / SITES[site])
    y = (table.pop("num") > 0).astype(int)
    for column in ("chol", "trestbps"):
        table.loc[table[column] == 0, column] = np.nan
    return table, y


def load_site(site: str, medians: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """One hospital with its gaps filled by the **Cleveland** medians - the
    deployed pipeline knows only the data it was built on."""
    X, y = load_site_raw(site)
    return X.fillna(medians), y


def cleveland_medians() -> pd.Series:
    return load_site_raw("Cleveland")[0].median()


# ---------------------------------------------------------------- calibration


def reliability_bins(p: np.ndarray, y: np.ndarray, n_bins: int = N_BINS) -> list[dict]:
    """Equal-width bins of the predicted probability: count, mean confidence,
    observed frequency of the disease. The first bin includes 0."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    index = np.clip(np.digitize(p, edges[1:-1], right=True), 0, n_bins - 1)
    bins = []
    for b in range(n_bins):
        mask = index == b
        bins.append({"low": edges[b], "high": edges[b + 1], "count": int(mask.sum()),
                     "confidence": float(p[mask].mean()) if mask.any() else None,
                     "frequency": float(y[mask].mean()) if mask.any() else None})
    return bins


def ece(p: np.ndarray, y: np.ndarray, n_bins: int = N_BINS) -> float:
    """Expected calibration error: the count-weighted mean gap per bin."""
    return float(sum(b["count"] / len(p) * abs(b["frequency"] - b["confidence"])
                     for b in reliability_bins(p, y, n_bins) if b["count"]))


def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """The temperature T minimizing the negative log-likelihood of sigmoid(z / T)."""
    def nll(T):
        p = np.clip(sigmoid(logits / T), 1e-7, 1 - 1e-7)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    return float(minimize_scalar(nll, bounds=(0.05, 10.0), method="bounded").x)


def calibration_study(seed: int = SEED) -> dict:
    """Before and after temperature scaling on one split."""
    X, y = load_heart()
    X_train, X_calib, X_test, y_train, y_calib, y_test = split(X, y, seed)
    model = train_xgboost(X_train, y_train, seed)
    T = fit_temperature(model.predict(X_calib, output_margin=True), y_calib.to_numpy())
    logits = model.predict(X_test, output_margin=True)
    y_test = y_test.to_numpy()
    before, after = sigmoid(logits), sigmoid(logits / T)
    return {"T": T, "p_before": before, "p_after": after, "y": y_test,
            "ece_before": ece(before, y_test), "ece_after": ece(after, y_test),
            "brier_before": brier(before, y_test), "brier_after": brier(after, y_test),
            "accuracy": float(((before > 0.5) == y_test).mean()),
            "accuracy_after": float(((after > 0.5) == y_test).mean())}


# -------------------------------------------------------------- other sites


def auc(p: np.ndarray, y: np.ndarray) -> float:
    """Area under the ROC curve: the probability that a random sick patient
    scores above a random healthy one (ties count one half)."""
    sick, healthy = p[y == 1], p[y == 0]
    greater = (sick[:, None] > healthy[None, :]).mean()
    ties = (sick[:, None] == healthy[None, :]).mean()
    return float(greater + 0.5 * ties)


def recalibrate_intercept(logits: np.ndarray, y: np.ndarray) -> float:
    """Calibration in the large: the shift b of the logit for which the mean of
    sigmoid(z + b) equals the observed prevalence."""
    from scipy.optimize import brentq
    return float(brentq(lambda b: sigmoid(logits + b).mean() - y.mean(), -20.0, 20.0))


def sites_study(seed: int = SEED) -> dict:
    """The Cleveland model (train part, T from the calibration part) on the
    Cleveland test part and on the three other hospitals. For each site also a
    local fix: the intercept re-fitted on half of the site's patients (stratified),
    judged on the other half."""
    X, y = load_heart()
    X_train, X_calib, X_test, y_train, y_calib, y_test = split(X, y, seed)
    model = train_xgboost(X_train, y_train, seed)
    T = fit_temperature(model.predict(X_calib, output_margin=True), y_calib.to_numpy())
    medians = cleveland_medians()
    rows = {}
    for site in ["Cleveland"] + OTHER_SITES:
        if site == "Cleveland":
            Xs, ys = X_test, y_test
            missing = {}
        else:
            raw, ys = load_site_raw(site)
            missing = {c: int(raw[c].isna().sum()) for c in raw.columns if raw[c].isna().any()}
            Xs = raw.fillna(medians)
        ys = ys.to_numpy()
        z = model.predict(Xs, output_margin=True) / T
        p = sigmoid(z)
        row = {"n": len(ys), "prevalence": float(ys.mean()), "auc": auc(p, ys),
               "accuracy": float(((p > 0.5) == ys).mean()), "mean_p": float(p.mean()),
               "ece": ece(p, ys), "brier": brier(p, ys), "missing": missing, "p": p, "y": ys}
        if site != "Cleveland":
            fit_part, eval_part = train_test_split(np.arange(len(ys)), test_size=0.5,
                                                   random_state=seed, stratify=ys)
            b = recalibrate_intercept(z[fit_part], ys[fit_part])
            p_fix = sigmoid(z[eval_part] + b)
            y_eval = ys[eval_part]
            row.update({"b": b, "n_eval": len(eval_part),
                        "accuracy_eval": float(((p[eval_part] > 0.5) == y_eval).mean()),
                        "ece_eval": ece(p[eval_part], y_eval),
                        "accuracy_fixed": float(((p_fix > 0.5) == y_eval).mean()),
                        "ece_fixed": ece(p_fix, y_eval),
                        "auc_eval": auc(p[eval_part], y_eval),
                        "auc_fixed": auc(p_fix, y_eval)})
        rows[site] = row
    return {"T": T, "sites": rows}


VALIDATION_REPEATS = 5   # repetitions of the random 5-fold cross-validation


def pooled_sites() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """All four hospitals in one table, as recorded (gaps as NaN), with the site of each patient."""
    tables, labels, sites = [], [], []
    for site in SITES:
        X, y = load_site_raw(site)
        tables.append(X)
        labels.append(y.to_numpy())
        sites += [site] * len(y)
    return pd.concat(tables, ignore_index=True), np.concatenate(labels), np.array(sites)


def validation_study(seed: int = SEED) -> dict:
    """Two ways to validate the shared recipe on the pooled 920 patients: random
    stratified 5-fold cross-validation (repeated VALIDATION_REPEATS times, out-of-fold
    predictions), and leave-one-site-out (train on three hospitals, test on the
    fourth). Gaps are filled with the medians of the training part. Per site
    the AUC and the accuracy of the out-of-fold or held-out predictions."""
    from sklearn.model_selection import StratifiedKFold
    X, y, sites = pooled_sites()

    def predict_part(train, test):
        medians = X.iloc[train].median()
        model = train_xgboost(X.iloc[train].fillna(medians), y[train], seed)
        return model.predict_proba(X.iloc[test].fillna(medians))[:, 1]

    def scores(p, mask):
        return {"auc": auc(p[mask], y[mask]), "accuracy": float(((p[mask] > 0.5) == y[mask]).mean())}

    random_runs = []
    for repeat in range(VALIDATION_REPEATS):
        p = np.zeros(len(y))
        for train, test in StratifiedKFold(5, shuffle=True, random_state=repeat).split(X, y):
            p[test] = predict_part(train, test)
        random_runs.append({"all": scores(p, np.ones(len(y), bool)),
                            **{site: scores(p, sites == site) for site in SITES}})
    random = {key: {m: float(np.mean([r[key][m] for r in random_runs])) for m in ("auc", "accuracy")}
              for key in random_runs[0]}
    held_out = {}
    for site in SITES:
        test = np.flatnonzero(sites == site)
        p = predict_part(np.flatnonzero(sites != site), test)
        held_out[site] = {"auc": auc(p, y[test]), "accuracy": float(((p > 0.5) == y[test]).mean())}
    return {"random": random, "held_out": held_out, "n": len(y)}


def sites_over_splits(n_splits: int = 20) -> dict:
    """`sites_study` for seeds 0..n-1: per site and measure an array over splits."""
    runs = [sites_study(seed) for seed in range(n_splits)]
    keys = ["auc", "accuracy", "ece", "ece_eval", "ece_fixed", "accuracy_eval", "accuracy_fixed"]
    return {site: {k: np.array([r["sites"][site][k] for r in runs])
                   for k in keys if k in runs[0]["sites"][site]}
            for site in runs[0]["sites"]}


# ----------------------------------------------------------------- monitoring

N_WEEKS = 10
DRIFT_START_WEEK = 6
BATCH_SIZE = 60
PREVALENCE_STEP = 0.10   # from the drift week the share of sick patients grows by 10 points a week
MAX_PREVALENCE = 0.95
PSI_BINS = 5             # 60 patients a week: ten bins would be mostly noise
WATCH, ALERT = 0.10, 0.25


def psi(reference: np.ndarray, current: np.ndarray, n_bins: int = PSI_BINS) -> float:
    """Population stability index of two samples of predicted probabilities,
    on equal-width bins of [0, 1]; empty bins get a share of 1e-4."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ref = np.clip(np.histogram(reference, edges)[0] / len(reference), 1e-4, None)
    cur = np.clip(np.histogram(current, edges)[0] / len(current), 1e-4, None)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def weekly_prevalence(week: int, base: float) -> float:
    """The share of sick patients the clinic sends in a given week."""
    return min(MAX_PREVALENCE, base + PREVALENCE_STEP * max(0, week - DRIFT_START_WEEK + 1))


def weekly_batches(X_pool: pd.DataFrame, y_pool: np.ndarray, seed: int = SEED):
    """Ten weekly batches of 60 patients bootstrapped from the pool. Weeks 1-5
    keep the pool's case mix; from week 6 a new referral pathway sends sicker
    patients - the prevalence grows by 10 points a week (label shift)."""
    rng = np.random.RandomState(seed)
    base = y_pool.mean()
    batches = []
    for week in range(1, N_WEEKS + 1):
        prevalence = weekly_prevalence(week, base)
        weights = np.where(y_pool == 1, prevalence / base, (1 - prevalence) / (1 - base))
        index = rng.choice(len(X_pool), BATCH_SIZE, replace=True, p=weights / weights.sum())
        batches.append((X_pool.iloc[index].reset_index(drop=True), y_pool[index]))
    return batches


def status(value: float) -> str:
    return "alert" if value >= ALERT else "watch" if value >= WATCH else "ok"


def monitoring_study(seed: int = SEED) -> dict:
    """PSI and the Kolmogorov-Smirnov p-value of each week's predicted
    probabilities against the reference - the model's predictions on the test
    part, the distribution it was validated on - and the accuracy the clinic
    would only learn weeks later."""
    X, y = load_heart()
    X_train, X_calib, X_test, y_train, y_calib, y_test = split(X, y, seed)
    model = train_xgboost(X_train, y_train, seed)
    y_pool = y_test.to_numpy()
    reference = model.predict_proba(X_test)[:, 1]
    weeks = []
    for week, (batch, labels) in enumerate(weekly_batches(X_test, y_pool, seed), start=1):
        p = model.predict_proba(batch)[:, 1]
        value = psi(reference, p)
        weeks.append({"week": week, "prevalence": weekly_prevalence(week, y_pool.mean()),
                      "psi": value, "status": status(value),
                      "ks_p": float(ks_2samp(reference, p).pvalue),
                      "accuracy": float(((p > 0.5) == labels).mean())})
    return {"reference": reference, "weeks": weeks}


NEW_SITE = "VA Long Beach"
NEW_SITE_STEP = 0.20     # from the drift week the new hospital's share grows by 20 points a week


def new_site_share(week: int) -> float:
    return min(1.0, NEW_SITE_STEP * max(0, week - DRIFT_START_WEEK + 1))


def new_hospital_study(seed: int = SEED) -> dict:
    """The second scenario of the dashboard: from week 6 a new hospital (VA Long
    Beach) sends a growing share of the 60 weekly patients - a covariate
    shift with gaps in the records, not just more sick patients. Besides PSI,
    K-S and the accuracy, the share of patients with at least one empty field
    as it arrives, before the Cleveland medians fill it."""
    X, y = load_heart()
    X_train, X_calib, X_test, y_train, y_calib, y_test = split(X, y, seed)
    model = train_xgboost(X_train, y_train, seed)
    raw, y_new = load_site_raw(NEW_SITE)
    X_new, y_new = raw.fillna(cleveland_medians()), y_new.to_numpy()
    y_pool = y_test.to_numpy()
    reference = model.predict_proba(X_test)[:, 1]
    rng = np.random.RandomState(seed)
    weeks = []
    for week in range(1, N_WEEKS + 1):
        share = new_site_share(week)
        n_new = int(round(BATCH_SIZE * share))
        old = rng.choice(len(X_test), BATCH_SIZE - n_new, replace=True)
        new = rng.choice(len(X_new), n_new, replace=True)
        batch = pd.concat([X_test.iloc[old], X_new.iloc[new]])
        labels = np.r_[y_pool[old], y_new[new]]
        with_gaps = np.r_[np.zeros(len(old), bool), raw.iloc[new].isna().any(axis=1).to_numpy()]
        p = model.predict_proba(batch)[:, 1]
        value = psi(reference, p)
        weeks.append({"week": week, "share_new": share, "prevalence": float(labels.mean()),
                      "psi": value, "status": status(value),
                      "ks_p": float(ks_2samp(reference, p).pvalue),
                      "accuracy": float(((p > 0.5) == labels).mean()),
                      "with_gaps": float(with_gaps.mean())})
    return {"reference": reference, "weeks": weeks}


if __name__ == "__main__":
    c = calibration_study()
    print(f"test accuracy {c['accuracy']:.3f} (after scaling {c['accuracy_after']:.3f})")
    print(f"T = {c['T']:.3f}   ECE {c['ece_before']:.3f} -> {c['ece_after']:.3f}   "
          f"Brier {c['brier_before']:.3f} -> {c['brier_after']:.3f}")
    for name in ("p_before", "p_after"):
        print(name, [(b["count"], None if b["count"] == 0 else round(b["confidence"], 3),
                      None if b["count"] == 0 else round(b["frequency"], 3))
                     for b in reliability_bins(c[name], c["y"])])
    for seed in range(20):
        s = calibration_study(seed)
        print(f"seed {seed:2d}: T {s['T']:.2f}  ECE {s['ece_before']:.3f} -> {s['ece_after']:.3f}"
              f"  Brier {s['brier_before']:.3f} -> {s['brier_after']:.3f}")
    m = monitoring_study()
    for w in m["weeks"]:
        print(f"week {w['week']:2d}: prevalence {w['prevalence']:.2f}  PSI {w['psi']:.3f} "
              f"({w['status']}), K-S p {w['ks_p']:.4f}, accuracy {w['accuracy']:.3f}")
    r = sites_study()
    print(f"other sites, T = {r['T']:.3f} from Cleveland")
    for site, v in r["sites"].items():
        print(f"{site:14s} n {v['n']:3d}  sick {v['prevalence']:.3f}  AUC {v['auc']:.3f}  "
              f"accuracy {v['accuracy']:.3f}  mean p {v['mean_p']:.3f}  ECE {v['ece']:.3f}  "
              f"missing {v['missing']}")
        if "b" in v:
            print(f"{'':14s} intercept b {v['b']:+.3f} on half: accuracy {v['accuracy_eval']:.3f} -> "
                  f"{v['accuracy_fixed']:.3f}, ECE {v['ece_eval']:.3f} -> {v['ece_fixed']:.3f}")
    n = new_hospital_study()
    for w in n["weeks"]:
        print(f"week {w['week']:2d}: VA share {w['share_new']:.1f}  PSI {w['psi']:.3f} ({w['status']}), "
              f"K-S p {w['ks_p']:.4f}, accuracy {w['accuracy']:.3f}, with gaps {w['with_gaps']:.2f}")
