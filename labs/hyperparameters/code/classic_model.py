"""The classical model of part 2 - the single source of its numbers.

A support-vector machine with an RBF kernel on the breast-cancer biopsies
(569 samples, 30 features, benign or malignant; it ships with scikit-learn, so
nothing is downloaded). Two hyperparameters, `C` and `gamma`, both searched on
a **logarithmic** scale, and one evaluation of a configuration is a whole
5-fold cross-validation on the training part - not a single fit.

The point of the block is that the black box of part 1 and a model are the same
thing as far as `bayes_opt` is concerned: hyperparameters in, one number out.

The notebook contains a copy of this code; `build_notebooks.py --check`
compares the copy with this module.

Run from the repository root:

    uv run python labs/hyperparameters/code/classic_model.py
"""

import time

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# The box the search lives in: log10 C and log10 gamma.
SPACE = {"log_C": (-3.0, 4.0), "log_gamma": (-4.0, 1.0)}
N_SPLITS = 5
SEED = 0


def load_data():
    """Training and test part. The test part is touched once, at the very end."""
    data = load_breast_cancer()
    return train_test_split(data.data, data.target, test_size=0.25,
                            stratify=data.target, random_state=SEED)


def make_model(C: float, gamma: float):
    """Standardization and the classifier in one object.

    The scaler belongs **inside** the model: in a cross-validation it has to be
    fitted on the training folds only, and a pipeline is what guarantees that.
    """
    return make_pipeline(StandardScaler(), SVC(C=C, gamma=gamma))


def cross_validated_accuracy(x_train, y_train, C: float, gamma: float) -> float:
    """One evaluation of a configuration: 5-fold cross-validation, in percent."""
    folds = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    scores = cross_val_score(make_model(C, gamma), x_train, y_train, cv=folds)
    return 100 * float(scores.mean())


def bayes_opt_search(x_train, y_train, n_init=5, n_iter=15, seed=SEED, verbose=0):
    """`bayes_opt` over the two hyperparameters."""
    from bayes_opt import BayesianOptimization

    def objective(log_C, log_gamma):
        return cross_validated_accuracy(x_train, y_train, 10**log_C, 10**log_gamma)

    optimizer = BayesianOptimization(f=objective, pbounds=SPACE,
                                     random_state=seed, verbose=verbose)
    optimizer.maximize(init_points=n_init, n_iter=n_iter)
    return optimizer


def random_search(x_train, y_train, n_trials=20, seed=SEED):
    """The baseline at the same budget - without it a number is not a result."""
    rng = np.random.default_rng(seed)
    low, high = np.array(list(SPACE.values())).T
    points = low + rng.random((n_trials, 2)) * (high - low)
    scores = [cross_validated_accuracy(x_train, y_train, 10**a, 10**b)
              for a, b in points]
    best = int(np.argmax(scores))
    return points[best], scores[best]


def main() -> None:
    x_train, x_test, y_train, y_test = load_data()
    print(f"breast cancer: {len(y_train)} training and {len(y_test)} test biopsies, "
          f"{x_train.shape[1]} features, "
          f"{100 * y_train.mean():.1f} % benign in the training part")

    start = time.time()
    default = cross_validated_accuracy(x_train, y_train, C=1.0, gamma=1.0 / x_train.shape[1])
    print(f"\ndefault C = 1, gamma = 1/30: cross-validated accuracy {default:.2f} % "
          f"(one evaluation takes {time.time() - start:.2f} s)")

    start = time.time()
    optimizer = bayes_opt_search(x_train, y_train)
    best = optimizer.max
    tuned_C = 10**best["params"]["log_C"]
    tuned_gamma = 10**best["params"]["log_gamma"]
    print(f"bayes_opt, 5 + 15 evaluations in {time.time() - start:.1f} s: "
          f"{best['target']:.2f} % at C = {tuned_C:.3g}, gamma = {tuned_gamma:.3g}")

    point, score = random_search(x_train, y_train)
    print(f"random search, 20 evaluations:              {score:.2f} % "
          f"at C = {10**point[0]:.3g}, gamma = {10**point[1]:.3g}")

    model = make_model(tuned_C, tuned_gamma).fit(x_train, y_train)
    print(f"\nthe tuned model, measured ONCE on the test part: "
          f"{100 * model.score(x_test, y_test):.2f} %")
    default_model = make_model(1.0, 1.0 / x_train.shape[1]).fit(x_train, y_train)
    print(f"the default model on the same test part:         "
          f"{100 * default_model.score(x_test, y_test):.2f} %")


if __name__ == "__main__":
    main()
