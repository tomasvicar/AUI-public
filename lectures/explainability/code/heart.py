"""UCI Heart Disease: logistic regression, XGBoost and the tabular explanations.

The data file is the one of the robustness lab
(`labs/robustness/data/heart_disease_cleveland.csv`, UCI, CC BY 4.0); the
XGBoost recipe is the shared one. This lecture uses a 75 / 25 train / test
split (no calibration part is needed here), stratified, seed 42.

Every number of the tabular slides comes from here:

    compare_models          accuracy and AUC of logistic regression vs. XGBoost
    permutation_importance  the drop of test accuracy per shuffled feature
    shap_patients           TreeSHAP of three patients (lowest, borderline, highest risk)
    pdp_ice / ale           partial dependence, ICE curves and accumulated local effects
    surrogate_fidelity      R^2 of a depth-3 tree fitted to the XGBoost predictions
    rule_quality            precision and coverage of an if-then rule on the model
    readable_tree           a depth-2 tree whose leaves read as rules

Run as a script to print the reference numbers:

    uv run python lectures/explainability/code/heart.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.inspection import permutation_importance as sk_permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

DATA = Path(__file__).resolve().parents[3] / "labs" / "robustness" / "data"
CSV = DATA / "heart_disease_cleveland.csv"
SEED = 42
FEATURE_NAMES = {
    "age": "age", "sex": "sex (1 = male)", "cp": "chest pain type", "trestbps": "resting blood pressure",
    "chol": "cholesterol", "fbs": "fasting blood sugar > 120", "restecg": "resting ECG",
    "thalach": "max heart rate", "exang": "exercise angina", "oldpeak": "ST depression",
    "slope": "ST slope", "ca": "vessels on fluoroscopy", "thal": "thallium test",
}


def load_heart(path: Path = CSV) -> tuple[pd.DataFrame, pd.Series]:
    """13 features (missing values filled with the median) and disease = num > 0."""
    table = pd.read_csv(path)
    y = (table.pop("num") > 0).astype(int)
    return table.fillna(table.median(numeric_only=True)), y


def split(X: pd.DataFrame, y: pd.Series, seed: int = SEED):
    """Train / test = 75 / 25 %, stratified."""
    return train_test_split(X, y, test_size=0.25, random_state=seed, stratify=y)


def train_xgboost(X_train: pd.DataFrame, y_train: pd.Series, seed: int = SEED):
    """The shared recipe of the tabular running example."""
    model = xgb.XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.1,
                              eval_metric="logloss", random_state=seed)
    return model.fit(X_train, y_train)


def train_logistic(X_train: pd.DataFrame, y_train: pd.Series, seed: int = SEED):
    """Logistic regression on standardized features."""
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=seed))
    return model.fit(X_train, y_train)


def setup():
    """Data, split and both models - what every function below starts from."""
    X, y = load_heart()
    X_train, X_test, y_train, y_test = split(X, y)
    return {"X": X, "y": y, "X_train": X_train, "X_test": X_test, "y_train": y_train,
            "y_test": y_test, "xgb": train_xgboost(X_train, y_train),
            "logistic": train_logistic(X_train, y_train)}


def compare_models(s: dict) -> dict:
    result = {}
    for name in ("logistic", "xgb"):
        p = s[name].predict_proba(s["X_test"])[:, 1]
        result[name] = {"accuracy": float(((p > 0.5) == s["y_test"]).mean()),
                        "auc": float(roc_auc_score(s["y_test"], p))}
    return result


def permutation_importance(s: dict, n_repeats: int = 30) -> pd.DataFrame:
    """Mean and standard deviation of the accuracy drop over 30 shuffles per feature."""
    r = sk_permutation_importance(s["xgb"], s["X_test"], s["y_test"], n_repeats=n_repeats,
                                  random_state=SEED, scoring="accuracy")
    return (pd.DataFrame({"mean": r.importances_mean, "std": r.importances_std},
                         index=s["X"].columns).sort_values("mean", ascending=False))


def tree_shap(model, X: pd.DataFrame) -> tuple[np.ndarray, float]:
    """Exact TreeSHAP values in log-odds (one row per patient, one column per
    feature) and the base value, from `shap.TreeExplainer` (Lundberg et al.).
    Needs shap >= 0.52 for XGBoost 3.x models, hence Python 3.12 (sources.md)."""
    import shap
    explanation = shap.TreeExplainer(model)(X)
    return explanation.values, float(np.ravel(explanation.base_values)[0])


def shap_patients(s: dict) -> dict:
    """TreeSHAP values (log-odds) of the lowest-risk, the borderline and the
    highest-risk test patient, with the base value."""
    values, base = tree_shap(s["xgb"], s["X_test"])
    p = s["xgb"].predict_proba(s["X_test"])[:, 1]
    chosen = {"lowest risk": int(np.argmin(p)), "borderline": int(np.argmin(np.abs(p - 0.5))),
              "highest risk": int(np.argmax(p))}
    patients = {}
    for name, i in chosen.items():
        patients[name] = {"index": i, "p": float(p[i]), "label": int(s["y_test"].iloc[i]),
                          "values": pd.Series(values[i], index=s["X"].columns),
                          "features": s["X_test"].iloc[i]}
    return {"base": base, "patients": patients,
            "mean_abs": pd.Series(np.abs(values).mean(0), index=s["X"].columns)}


def pdp_ice(s: dict, feature: str, n_grid: int = 20, n_curves: int = 30):
    """ICE curves of 30 test patients and the partial dependence (their mean
    over all test patients) on a grid between the 2nd and 98th percentile."""
    X_test = s["X_test"]
    grid = np.linspace(X_test[feature].quantile(0.02), X_test[feature].quantile(0.98), n_grid)
    curves = []
    for value in grid:
        modified = X_test.copy()
        modified[feature] = value
        curves.append(s["xgb"].predict_proba(modified)[:, 1])
    curves = np.array(curves).T                    # (patients, grid)
    shown = np.random.RandomState(SEED).choice(len(X_test), n_curves, replace=False)
    return grid, curves[shown], curves.mean(0)


def pdp_at(s: dict, feature: str, value: float) -> float:
    """The partial dependence at one value: every test patient given that value."""
    modified = s["X_test"].copy()
    modified[feature] = value
    return float(s["xgb"].predict_proba(modified)[:, 1].mean())


def ale(s: dict, feature: str, n_bins: int = 12):
    """First-order accumulated local effects on quantile bins of the whole data:
    within each bin only the patients whose value lies in it are moved to the
    bin's two edges; the mean differences are accumulated and centred."""
    X = s["X"]
    values = X[feature].to_numpy(dtype=float)
    edges = np.unique(np.quantile(values, np.linspace(0, 1, n_bins + 1)))
    deltas, centers = [], []
    for k in range(len(edges) - 1):
        low, high = edges[k], edges[k + 1]
        inside = (values >= low) & ((values <= high) if k == len(edges) - 2 else (values < high))
        if not inside.any():
            continue
        at_low, at_high = X[inside].copy(), X[inside].copy()
        at_low[feature], at_high[feature] = low, high
        deltas.append(float((s["xgb"].predict_proba(at_high)[:, 1]
                             - s["xgb"].predict_proba(at_low)[:, 1]).mean()))
        centers.append((low + high) / 2)
    effect = np.cumsum(deltas)
    return np.array(centers), effect - effect.mean()


def surrogate_fidelity(s: dict, depth: int = 3) -> dict:
    """A regression tree fitted to the XGBoost probabilities on the training
    part; its R^2 against the black box (not against the labels)."""
    target_train = s["xgb"].predict_proba(s["X_train"])[:, 1]
    target_test = s["xgb"].predict_proba(s["X_test"])[:, 1]
    tree = DecisionTreeRegressor(max_depth=depth, random_state=SEED).fit(s["X_train"], target_train)
    return {"tree": tree, "r2_train": float(r2_score(target_train, tree.predict(s["X_train"]))),
            "r2_test": float(r2_score(target_test, tree.predict(s["X_test"])))}


def rule_quality(s: dict, rule) -> dict:
    """Coverage (share of the 303 patients the rule fires on) and precision
    (share of those the model calls high risk) of an if-then rule, `rule`
    being a function of the feature table returning a boolean mask."""
    mask = rule(s["X"]).to_numpy()
    predicted = s["xgb"].predict(s["X"])
    return {"coverage": float(mask.mean()), "n": int(mask.sum()),
            "precision": float(predicted[mask].mean())}


RULES = {
    "age > 55 AND cholesterol > 240": lambda X: (X["age"] > 55) & (X["chol"] > 240),
    "asymptomatic chest pain AND at least one vessel": lambda X: (X["cp"] == 4) & (X["ca"] >= 1),
}


def readable_tree(s: dict) -> DecisionTreeClassifier:
    """A depth-2 classification tree on the training part: four leaves, each
    an if-then rule with the number of healthy and sick patients in it."""
    return DecisionTreeClassifier(max_depth=2, random_state=SEED).fit(s["X_train"], s["y_train"])


if __name__ == "__main__":
    from sklearn.tree import export_text
    s = setup()
    for name, r in compare_models(s).items():
        print(f"{name:9s} accuracy {r['accuracy']:.3f}, AUC {r['auc']:.3f}")
    print(permutation_importance(s).head(6).round(3))
    sh = shap_patients(s)
    print(f"SHAP base value {sh['base']:.3f} (log-odds)")
    for name, pt in sh["patients"].items():
        top = pt["values"].abs().sort_values(ascending=False).index[:3]
        print(f"  {name}: P = {pt['p']:.3f}, label {pt['label']}, top "
              + ", ".join(f"{f} {pt['values'][f]:+.2f}" for f in top))
    print(f"PDP at age 70: {pdp_at(s, 'age', 70):.3f}; corr(age, thalach) = "
          f"{s['X'][['age', 'thalach']].corr().iloc[0, 1]:.2f}")
    sur = surrogate_fidelity(s)
    print(f"surrogate depth 3: R^2 train {sur['r2_train']:.3f}, test {sur['r2_test']:.3f}")
    for name, rule in RULES.items():
        q = rule_quality(s, rule)
        print(f"rule '{name}': coverage {q['coverage']:.3f} (n {q['n']}), precision {q['precision']:.3f}")
    print(export_text(readable_tree(s), feature_names=list(s["X"].columns), show_weights=True))
