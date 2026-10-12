"""The studies of the second round (5 October 2026) on the Heart Disease model.

Every number of the slides added on 5 October 2026 is printed here; the
figures in `figures_studies.py` draw them and pin them with `assert`s. The
recipe, the data and the splits are those of `heart.py`:

    training_share      the share of women in a training set of fixed size,
                        women's and men's AUC on a fixed test part (the design
                        of Larrazabal et al. 2020 on chest X-rays)
    unawareness         the audit with and without the `sex` column, over the
                        twenty splits; how well the other twelve features
                        predict sex
    calibration_by_sex  predicted risk against the observed rate, by sex, on
                        the test parts of the twenty splits
    intersections       sex x age (< 55 / >= 55): how small the cells get
    interventions       pre-processing (reweighing), in-processing
                        (`fairlearn`'s ExponentiatedGradient) and
                        post-processing (ThresholdOptimizer) on the 50 / 25 / 25
                        split of the mitigation; the randomized ones on one
                        draw and as the mean of 200 draws
    coin_flips          which test patients the ThresholdOptimizer decides by
                        a coin flip

Run as a script to print the reference numbers (about two minutes):

    uv run python lectures/bias-fairness/code/studies.py
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

import heart

SHARES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
TRAIN_SIZE = 140              # the largest size with 50 % women in every training part
AGE_CUT = 55


def training_share(n_splits: int = heart.N_SPLITS) -> pd.DataFrame:
    """For each seed: a 75 / 25 split stratified by sex and disease; from the
    training part, a training set of 140 patients with the given share of
    women; the AUC of each group on the (fixed) test part."""
    X, y = heart.load_heart()
    strata = X["sex"].astype(int).astype(str) + y.astype(str)
    rows = []
    for seed in range(n_splits):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
                                                            random_state=seed, stratify=strata)
        rng = np.random.RandomState(seed)
        women, men = X_train.index[X_train["sex"] == 0], X_train.index[X_train["sex"] == 1]
        for share in SHARES:
            n_women = int(round(share * TRAIN_SIZE))
            chosen = np.concatenate([rng.choice(women, n_women, replace=False),
                                     rng.choice(men, TRAIN_SIZE - n_women, replace=False)])
            model = heart.train_xgboost(X_train.loc[chosen], y_train.loc[chosen], seed)
            score = model.predict_proba(X_test)[:, 1]
            for code, group in heart.GROUPS.items():
                g = (X_test["sex"] == code).to_numpy()
                rows.append({"seed": seed, "share": share, "group": group,
                             "AUC": roc_auc_score(y_test[g], score[g]),
                             "accuracy": float(((score[g] > 0.5) == y_test[g].to_numpy()).mean())})
    return pd.DataFrame(rows)


def unawareness(n_splits: int = heart.N_SPLITS) -> dict:
    """The audit of `heart.repeated_audit` with and without `sex`, and the
    AUC of the shared recipe predicting sex from the other twelve features."""
    X, y = heart.load_heart()
    without = [c for c in X.columns if c != "sex"]
    rows, sex_auc = [], []
    for seed in range(n_splits):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
                                                            random_state=seed, stratify=y)
        for name, columns in (("with sex", list(X.columns)), ("without sex", without)):
            y_pred = heart.train_xgboost(X_train[columns], y_train, seed).predict(X_test[columns])
            m = heart.group_metrics(y_test, y_pred, X_test["sex"])
            for group, values in m.iterrows():
                rows.append({"seed": seed, "model": name, "group": group, **values.to_dict()})
        Z_train, Z_test, s_train, s_test = train_test_split(X[without], X["sex"], test_size=0.25,
                                                            random_state=seed, stratify=X["sex"])
        sex_model = heart.train_xgboost(Z_train, s_train, seed)
        sex_auc.append(roc_auc_score(s_test, sex_model.predict_proba(Z_test)[:, 1]))
    means = X.groupby("sex")[without].mean()
    return {"audit": pd.DataFrame(rows), "sex AUC": np.array(sex_auc),
            "thal": {heart.GROUPS[c]: float(means.loc[c, "thal"]) for c in (0, 1)},
            "thal normal": {heart.GROUPS[c]: float((X.loc[X["sex"] == c, "thal"] == 3).mean())
                            for c in (0, 1)}}


def calibration_by_sex(n_splits: int = heart.N_SPLITS) -> pd.DataFrame:
    """The predicted risk of every test patient of the twenty audit splits."""
    X, y = heart.load_heart()
    frames = []
    for seed in range(n_splits):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
                                                            random_state=seed, stratify=y)
        score = heart.train_xgboost(X_train, y_train, seed).predict_proba(X_test)[:, 1]
        frames.append(pd.DataFrame({"seed": seed, "risk": score, "sick": y_test.to_numpy(),
                                    "group": X_test["sex"].map(heart.GROUPS).to_numpy()}))
    return pd.concat(frames, ignore_index=True)


def intersections(n_splits: int = heart.N_SPLITS) -> pd.DataFrame:
    """Sex x age: patients and sick patients in the data, and the number of
    sick patients per test part over the twenty audit splits."""
    X, y = heart.load_heart()
    older = X["age"] >= AGE_CUT
    rows = []
    for code, group in heart.GROUPS.items():
        for is_older in (False, True):
            cell = (X["sex"] == code) & (older == is_older)
            per_split = []
            for seed in range(n_splits):
                _, X_test, _, y_test = train_test_split(X, y, test_size=0.25,
                                                        random_state=seed, stratify=y)
                c = (X_test["sex"] == code) & ((X_test["age"] >= AGE_CUT) == is_older)
                per_split.append(int(y_test[c].sum()))
            rows.append({"group": group, "age": f">= {AGE_CUT}" if is_older else f"< {AGE_CUT}",
                         "patients": int(cell.sum()), "sick": int(y[cell].sum()),
                         "sick per test part": (min(per_split), max(per_split))})
    return pd.DataFrame(rows)


def reweighing_weights(sex, y) -> np.ndarray:
    """Kamiran and Calders (2012): w(a, y) = P(A = a) P(Y = y) / P(A = a, Y = y),
    so that sex and the label are independent in the weighted training data."""
    sex, y = np.asarray(sex), np.asarray(y)
    weights = np.empty(len(y))
    for a in (0, 1):
        for label in (0, 1):
            cell = (sex == a) & (y == label)
            weights[cell] = (sex == a).mean() * (y == label).mean() / cell.mean()
    return weights


def _xgboost(seed: int = heart.SEED):
    return heart.XGBClassifier64(n_estimators=200, max_depth=3, learning_rate=0.1,
                                 eval_metric="logloss", random_state=seed)


def _summary(y_test, y_pred, sex) -> dict:
    metrics = heart.group_metrics(y_test, y_pred, sex)
    return {"metrics": metrics, "gaps": heart.gaps(metrics),
            "accuracy": float((y_pred == y_test.to_numpy()).mean())}


def _mean_summary(summaries: list[dict]) -> dict:
    """The mean of per-draw summaries: metrics, gaps and accuracy averaged
    over the draws (the gaps are averaged per draw, not recomputed from the
    mean metrics)."""
    return {"metrics": sum(s["metrics"] for s in summaries) / len(summaries),
            "gaps": {k: float(np.mean([s["gaps"][k] for s in summaries])) for k in summaries[0]["gaps"]},
            "accuracy": float(np.mean([s["accuracy"] for s in summaries]))}


def interventions(seed: int = heart.SEED, n_draws: int = heart.N_DRAWS) -> dict:
    """The three points of intervention on the mitigation split, all judged
    on its test part. Returns, per approach, the per-group metrics, the gaps
    and the accuracy, plus the reweighing weights and the training fit.
    The in-processing reduction and the ThresholdOptimizer predict by a draw:
    `results` holds the draw random_state = seed, `means` the mean over
    random_state 0 ... n_draws - 1 (for the deterministic approaches the two
    are the same)."""
    from fairlearn.reductions import DemographicParity, EqualizedOdds, ExponentiatedGradient
    X_train, X_calib, X_test, y_train, y_calib, y_test = heart.mitigation_split(seed)
    model = heart.train_xgboost(X_train, y_train, seed)
    weights = reweighing_weights(X_train["sex"], y_train)
    reweighed = _xgboost(seed).fit(X_train, y_train, sample_weight=weights)
    predictions = {"unmitigated": model.predict(X_test), "reweighing": reweighed.predict(X_test)}
    randomized = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name, constraint in (("in: equalized odds", EqualizedOdds()),
                                 ("in: demographic parity", DemographicParity())):
            reduction = ExponentiatedGradient(_xgboost(seed), constraints=constraint)
            reduction.fit(X_train, y_train, sensitive_features=X_train["sex"])
            predictions[name] = np.asarray(reduction.predict(X_test, random_state=seed))
            randomized[name] = lambda r, reduction=reduction: reduction.predict(X_test, random_state=r)
    for name, constraint in (("post: equalized odds", "equalized_odds"),
                             ("post: demographic parity", "demographic_parity")):
        optimizer = heart.mitigate(model, X_calib, y_calib, constraint, seed)
        predictions[name] = np.asarray(optimizer.predict(X_test, sensitive_features=X_test["sex"],
                                                         random_state=seed))
        randomized[name] = lambda r, optimizer=optimizer: optimizer.predict(
            X_test, sensitive_features=X_test["sex"], random_state=r)
    result = {name: _summary(y_test, y_pred, X_test["sex"]) for name, y_pred in predictions.items()}
    means = dict(result)
    for name, predict in randomized.items():
        means[name] = _mean_summary([_summary(y_test, np.asarray(predict(r)), X_test["sex"])
                                     for r in range(n_draws)])
    cells = {(heart.GROUPS[a], label): float(weights[((X_train["sex"] == a) & (y_train == label)).to_numpy()][0])
             for a in (0, 1) for label in (0, 1)}
    train_fit = heart.group_metrics(y_train, model.predict(X_train), X_train["sex"])
    return {"results": result, "means": means, "weights": cells, "train fit": train_fit,
            "predictions": predictions}


def coin_flips(seed: int = heart.SEED, draws: int = 200) -> dict:
    """The equalized-odds ThresholdOptimizer of the mitigation: per group a
    lower and an upper threshold on the predicted risk; between them the
    patient is flagged with a probability p. Counts the test patients whose
    decision changes between `draws` predictions with different random states."""
    X_train, X_calib, X_test, y_train, y_calib, y_test = heart.mitigation_split(seed)
    model = heart.train_xgboost(X_train, y_train, seed)
    optimizer = heart.mitigate(model, X_calib, y_calib, "equalized_odds", seed)
    risk = model.predict_proba(X_test)[:, 1]
    rules = {}
    for code, group in heart.GROUPS.items():
        entry = optimizer.interpolated_thresholder_.interpolation_dict[code]
        thresholds = [float(str(entry[f"operation{k}"]).strip("[]>")) for k in (0, 1)]
        upper, lower = max(thresholds), min(thresholds)
        p_between = entry["p1"] if thresholds[1] == lower else entry["p0"]
        rules[group] = {"lower": lower, "upper": upper, "p": float(p_between),
                        "p_ignore": float(entry.get("p_ignore", 0.0))}
    decisions = np.array([np.asarray(optimizer.predict(X_test, sensitive_features=X_test["sex"],
                                                       random_state=r)) for r in range(draws)])
    varies = decisions.min(axis=0) != decisions.max(axis=0)
    sex = X_test["sex"].map(heart.GROUPS).to_numpy()
    return {"rules": rules, "risk": risk, "sex": sex, "sick": y_test.to_numpy(),
            "varies": varies, "share flagged": decisions.mean(axis=0),
            "varies by group": {g: int((varies & (sex == g)).sum()) for g in heart.GROUPS.values()}}


def roc_by_sex(seed: int = heart.SEED) -> dict:
    """The equalized-odds ThresholdOptimizer of the mitigation, seen as Hardt
    et al. draw it: each group's ROC curve on the calibration part (where the
    thresholds are fitted), its convex hull as `fairlearn` interpolates it,
    the common operating point it chooses, and each group's point at the
    default threshold 0.5. The AUC of each group on both parts."""
    from sklearn.metrics import roc_curve
    X_train, X_calib, X_test, y_train, y_calib, y_test = heart.mitigation_split(seed)
    model = heart.train_xgboost(X_train, y_train, seed)
    optimizer = heart.mitigate(model, X_calib, y_calib, "equalized_odds", seed)
    risk_calib, risk_test = model.predict_proba(X_calib)[:, 1], model.predict_proba(X_test)[:, 1]
    groups = {}
    for code, group in heart.GROUPS.items():
        c, t = (X_calib["sex"] == code).to_numpy(), (X_test["sex"] == code).to_numpy()
        fpr, tpr, _ = roc_curve(y_calib[c], risk_calib[c])
        flagged, sick = risk_calib[c] > 0.5, y_calib[c].to_numpy() == 1
        hull = optimizer._tradeoff_curve[code]
        groups[group] = {"roc": (fpr, tpr), "hull": (hull["x"].to_numpy(), hull["y"].to_numpy()),
                         "default": (float(flagged[~sick].mean()), float(flagged[sick].mean())),
                         "n": int(c.sum()), "sick": int(sick.sum()),
                         "AUC calibration": roc_auc_score(y_calib[c], risk_calib[c]),
                         "AUC test": roc_auc_score(y_test[t], risk_test[t])}
    return {"groups": groups, "common": (float(optimizer._x_best), float(optimizer._y_best))}


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """The Wilson score interval of a proportion (95 % by default)."""
    p = successes / n
    centre = (p + z ** 2 / (2 * n)) / (1 + z ** 2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / (1 + z ** 2 / n)
    return centre - half, centre + half


def audit_intervals() -> dict:
    """TPR and FPR of each group on the audited (explainability) split, with
    their counts and Wilson intervals."""
    X, y = heart.load_heart()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
                                                        random_state=heart.SEED, stratify=y)
    y_pred = heart.train_xgboost(X_train, y_train).predict(X_test)
    out = {}
    for code, group in heart.GROUPS.items():
        g = (X_test["sex"] == code).to_numpy()
        t, p = y_test[g].to_numpy(), y_pred[g]
        tp, sick = int(p[t == 1].sum()), int((t == 1).sum())
        fp, healthy = int(p[t == 0].sum()), int((t == 0).sum())
        out[group] = {"TPR": (tp, sick, wilson(tp, sick)), "FPR": (fp, healthy, wilson(fp, healthy))}
    return out


if __name__ == "__main__":
    s = training_share()
    print("training share of women (140 patients), median over 20 splits:")
    print(s.groupby(["group", "share"])[["AUC", "accuracy"]].median().round(3).unstack(0).to_string())
    u = unawareness()
    med = u["audit"].groupby(["model", "group"])[["selection rate", "TPR", "FPR", "PPV"]].median()
    print("\nunawareness, median over 20 splits:\n" + med.round(3).to_string())
    a = u["audit"]
    for model in ("with sex", "without sex"):
        w = a[(a.model == model) & (a.group == "female")].sort_values("seed")
        m = a[(a.model == model) & (a.group == "male")].sort_values("seed")
        for metric in ("selection rate", "TPR", "FPR"):
            d = m[metric].to_numpy() - w[metric].to_numpy()
            print(f"  {model:12s} {metric:15s} male - female median {np.median(d):+.3f}, "
                  f"positive in {(d > 0).sum()} of 20")
    print(f"sex from the other twelve features: AUC median {np.median(u['sex AUC']):.3f} "
          f"(range {u['sex AUC'].min():.3f}-{u['sex AUC'].max():.3f}); "
          f"thal mean {u['thal']}; thal normal {u['thal normal']}")
    c = calibration_by_sex()
    print("\ncalibration by sex (20 test parts pooled):")
    for group in ("female", "male"):
        q = c[c.group == group]
        bins = pd.cut(q.risk, [0, 0.2, 0.4, 0.6, 0.8, 1.0], include_lowest=True)
        print(f"  {group}: {len(q)} predictions, mean risk {q.risk.mean():.3f}, observed {q.sick.mean():.3f}")
        print(q.groupby(bins, observed=False).agg(n=("sick", "size"), risk=("risk", "mean"),
                                                  observed=("sick", "mean")).round(3).to_string())
    print("\nintersections:\n" + intersections().to_string())
    i = interventions()
    print("\ninterventions (test part of 50 / 25 / 25):")
    print("  weights", {k: round(v, 3) for k, v in i["weights"].items()})
    print("  training fit of XGBoost:\n" + i["train fit"][["TPR", "FPR"]].to_string())
    for name, r in [(n, r) for n, r in i["results"].items()] +             [(f"{n} (mean of {heart.N_DRAWS})", r) for n, r in i["means"].items() if n in ("in: demographic parity",
             "post: equalized odds", "post: demographic parity")]:
        m = r["metrics"]
        print(f"  {name:26s} acc {r['accuracy']:.3f}  "
              f"DP gap {r['gaps']['demographic parity']:.3f}  EO gap {r['gaps']['equalized odds']:.3f}  "
              + "  ".join(f"{g} sel {m.loc[g, 'selection rate']:.3f} TPR {m.loc[g, 'TPR']:.3f} "
                          f"FPR {m.loc[g, 'FPR']:.3f}" for g in ("female", "male")))
    f = coin_flips()
    print("\ncoin flips (equalized odds):", {g: {k: round(v, 4) for k, v in r.items()} for g, r in f["rules"].items()})
    print(f"  decisions that vary over 200 draws: {int(f['varies'].sum())} of {len(f['varies'])}, "
          f"by group {f['varies by group']}")
    r = roc_by_sex()
    print("\nROC by sex (calibration part):", {g: {k: (round(v, 3) if isinstance(v, float) else v)
          for k, v in d.items() if k not in ("roc", "hull")} for g, d in r["groups"].items()},
          "common point (FPR, TPR)", tuple(round(v, 3) for v in r["common"]))
    for group, metrics in audit_intervals().items():
        print(f"audit split, {group}: " + "; ".join(
            f"{m} {k}/{n} = {k / n:.3f}, 95 % Wilson [{lo:.2f}, {hi:.2f}]" for m, (k, n, (lo, hi)) in metrics.items()))
