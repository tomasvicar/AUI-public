"""Hyperparameter tuning of a small neural network with Optuna.

**The data are completely synthetic**, there is no measurement behind them:
two cell types described by area and circularity, with the label drawn from
the probability `sigma(s)` around a curved boundary, so a part of the cells
carries a "wrong" label on purpose.

Six hyperparameters are tuned, deliberately one of every kind:

    eta        learning rate      float, LOGARITHMIC scale    1e-4 .. 3e-1
    reg        weight decay       float, LOGARITHMIC scale    1e-6 .. 1e-1
    width      layer width        int                         4 .. 64
    layers     number of layers   int                         1 .. 3
    batch      batch size         categorical                 8 / 16 / 32 / 64
    optimizer  optimizer          categorical                 adam / sgd

One "evaluation" of a configuration is not one training run but a **three-fold
cross-validation** — the network is trained three times and the objective is
the mean accuracy on the held-out part of the data. That is exactly the
expensive and noisy black box the lecture talks about.

Under the hood Optuna does not use a Gaussian process by default but a
**tree-structured Parzen estimator (TPE)**; the loop is the same though —
model, acquisition, next point.

Everything is deterministic (fixed seed for the sampler and for the training),
so the numbers on the slides come out the same on every run.

Run it with:

    uv run python lectures/hyperparameters/code/optuna_hyperparameters.py
"""

from __future__ import annotations

import numpy as np
import optuna
import torch
from torch import nn

SEED = 7
N_CELLS = 600          # size of the whole set the folds are made of
N_FOLDS = 3            # three-fold cross-validation
N_EPOCHS = 25          # short training, so the demo finishes in a few dozen seconds
N_TRIALS = 30          # budget N: how many configurations are tried at all

# The hidden boundary between the two cell types.
AREA_CENTER, AREA_SCALE = 100.0, 40.0
CIRC_CENTER, CIRC_SCALE = 0.72, 0.16

BATCH_SIZES = [8, 16, 32, 64]
OPTIMIZERS = ["adam", "sgd"]


def cell_data(n: int = N_CELLS, seed: int = SEED):
    """Synthetic cells: features (area, circularity) and a type label 0/1."""
    rng = np.random.default_rng(seed)
    area = rng.uniform(30.0, 170.0, n)
    circularity = rng.uniform(0.45, 1.0, n)
    z1 = (area - AREA_CENTER) / AREA_SCALE
    z2 = (circularity - CIRC_CENTER) / CIRC_SCALE
    score = 1.3 * z1 - 1.5 * z2 + 0.6 * z1 * z2 + 0.2        # curved boundary
    p = 1.0 / (1.0 + np.exp(-1.8 * score))                   # some labels are noise
    label = (rng.uniform(size=n) < p).astype(np.float32)
    X = np.column_stack([area, circularity]).astype(np.float32)
    return X, label


def network(width: int, layers: int, seed: int) -> nn.Sequential:
    """A network with `layers` hidden layers of `width` neurons each."""
    torch.manual_seed(seed)
    parts: list[nn.Module] = []
    n_in = 2
    for _ in range(layers):
        parts += [nn.Linear(n_in, width), nn.Tanh()]
        n_in = width
    parts.append(nn.Linear(n_in, 1))
    return nn.Sequential(*parts)


def _train_and_score(X_train, y_train, X_val, y_val, config, seed) -> float:
    """One training run and its accuracy on the held-out part of the data."""
    mean, std = X_train.mean(axis=0), X_train.std(axis=0)
    Xt = torch.tensor((X_train - mean) / std)
    yt = torch.tensor(y_train)
    Xv = torch.tensor((X_val - mean) / std)
    yv = torch.tensor(y_val)

    model = network(config["width"], config["layers"], seed)
    loss_fn = nn.BCEWithLogitsLoss()
    if config["optimizer"] == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=config["eta"],
                               weight_decay=config["reg"])
    else:
        opt = torch.optim.SGD(model.parameters(), lr=config["eta"],
                              momentum=0.9, weight_decay=config["reg"])

    batch = config["batch"]
    generator = torch.Generator().manual_seed(seed)
    for _ in range(N_EPOCHS):
        order = torch.randperm(len(Xt), generator=generator)
        for start in range(0, len(Xt), batch):
            picked = order[start:start + batch]
            opt.zero_grad()
            loss = loss_fn(model(Xt[picked]).squeeze(1), yt[picked])
            loss.backward()
            opt.step()

    with torch.no_grad():
        logit = model(Xv).squeeze(1)
        return float(((logit > 0).float() == yv).float().mean())


def cross_validate(eta, reg, width, layers, batch, optimizer,
                   seed: int = SEED) -> float:
    """One **expensive** evaluation of a configuration: mean over `N_FOLDS` folds."""
    config = {"eta": eta, "reg": reg, "width": width, "layers": layers,
              "batch": batch, "optimizer": optimizer}
    X, y = cell_data(seed=seed)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(X))
    folds = np.array_split(order, N_FOLDS)

    accuracies = []
    for i, fold in enumerate(folds):
        mask = np.ones(len(X), dtype=bool)
        mask[fold] = False
        accuracies.append(_train_and_score(X[mask], y[mask], X[fold], y[fold],
                                           config, seed + i))
    return float(np.mean(accuracies))


def objective(trial: optuna.Trial) -> float:
    """Objective for Optuna: the cross-validated accuracy (maximized)."""
    eta = trial.suggest_float("eta", 1e-4, 3e-1, log=True)
    reg = trial.suggest_float("reg", 1e-6, 1e-1, log=True)
    width = trial.suggest_int("width", 4, 64)
    layers = trial.suggest_int("layers", 1, 3)
    batch = trial.suggest_categorical("batch", BATCH_SIZES)
    optimizer = trial.suggest_categorical("optimizer", OPTIMIZERS)
    # one expensive evaluation: three trainings, averaged
    return cross_validate(eta, reg, width, layers, batch, optimizer)


def run_study(n_trials: int = N_TRIALS, seed: int = SEED) -> optuna.Study:
    """Runs the whole tuning and returns the finished study (deterministically)."""
    torch.set_num_threads(1)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    return study


def importance(study: optuna.Study) -> dict[str, float]:
    """Hyperparameter importance (fANOVA) — deterministic, with a fixed seed."""
    return optuna.importance.get_param_importances(
        study, evaluator=optuna.importance.FanovaImportanceEvaluator(seed=SEED))


def main() -> None:
    X, y = cell_data()
    print("=" * 70)
    print("SYNTHETIC DATA - two cell types by area and circularity")
    print("=" * 70)
    print(f"  cells in total: {len(X)},  share of type 1: {y.mean():.2f}")
    print(f"  one evaluation = train the network {N_FOLDS}x for {N_EPOCHS} epochs")
    print(f"  budget: {N_TRIALS} trials, TPE sampler (seed {SEED})")

    study = run_study()

    print("\n" + "=" * 70)
    print("COURSE OF THE SEARCH")
    print("=" * 70)
    best_so_far = -np.inf
    for trial in study.trials:
        best_so_far = max(best_so_far, trial.value)
        if trial.number % 5 == 0 or trial.number == len(study.trials) - 1:
            # Optuna numbers the trials from zero, the slides and the plot from one.
            print(f"  trial {trial.number + 1:>3}   accuracy {trial.value:.4f}"
                  f"   best so far {best_so_far:.4f}")

    print("\n" + "=" * 70)
    print("BEST CONFIGURATION")
    print("=" * 70)
    print(f"  cross-validated accuracy: {study.best_value:.4f}"
          f"  (trial no. {study.best_trial.number + 1})")
    for key, value in study.best_params.items():
        if isinstance(value, float):
            print(f"  {key:<16} {value:.5g}")
        else:
            print(f"  {key:<16} {value}")

    print("\n" + "=" * 70)
    print("HYPERPARAMETER IMPORTANCE (fANOVA)")
    print("=" * 70)
    for key, value in importance(study).items():
        print(f"  {key:<16} {value:.3f}")
    print("\n  Careful: this number is the accuracy on the VALIDATION part of")
    print("  the data, the part that was tuned on. The honest number comes from")
    print("  a test set that took no part in the search.")


if __name__ == "__main__":
    main()
