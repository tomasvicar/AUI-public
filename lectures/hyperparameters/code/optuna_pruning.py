"""Pruning in Optuna: stop a hopeless candidate instead of paying for it in full.

This is the runnable version of the pruning slide. It tunes the **same six
hyperparameters of the same small network on the same synthetic cells** as
`optuna_hyperparameters.py`, with one difference that matters:

* `optuna_hyperparameters.py` scores a configuration by a three-fold
  cross-validation, so there is no natural "intermediate result" to report —
  the number only exists once all three trainings are done;
* here one evaluation is a **single train/validation split**, so the accuracy
  after every epoch is a legitimate intermediate value. That is what
  `trial.report()` publishes and what `trial.should_prune()` is asked about.

The schedule is a `HyperbandPruner` with a maximum resource of 81 epochs and a
reduction factor of 3, so its rungs sit at steps 1, 3, 9 and 27 — a candidate
cut at the first rung has bought two epochs of training, at the second four,
and so on. The sampler (TPE) decides *where* to look, the pruner decides *how
long* to look there — two independent objects.

Because most candidates are cut after one or three epochs, the budget is spent
on many more configurations than a full-training search of the same price:
that is the whole point, and the script prints the bill at the end.

Everything is deterministic (fixed seeds for the sampler, the split and the
training), so the numbers repeat on every run.

Run it with:

    uv run python lectures/hyperparameters/code/optuna_pruning.py
"""

from __future__ import annotations

import numpy as np
import optuna
import torch
from torch import nn

from optuna_hyperparameters import (BATCH_SIZES, OPTIMIZERS, SEED, cell_data,
                                    network)

MAX_EPOCHS = 81        # maximum resource of the schedule, in epochs
REDUCTION = 3          # one candidate in three survives a rung
N_TRIALS = 60          # budget in trials; with pruning it goes UP, not down
N_TRAIN = 400          # the rest of the 600 cells is the validation part


def _split():
    """One fixed train/validation split — no cross-validation here on purpose."""
    X, y = cell_data()
    order = np.random.default_rng(SEED).permutation(len(X))
    train, val = order[:N_TRAIN], order[N_TRAIN:]
    mean, std = X[train].mean(axis=0), X[train].std(axis=0)
    return (torch.tensor((X[train] - mean) / std), torch.tensor(y[train]),
            torch.tensor((X[val] - mean) / std), torch.tensor(y[val]))


DATA = _split()


def _accuracy(model, X_val, y_val) -> float:
    with torch.no_grad():
        return float(((model(X_val).squeeze(1) > 0).float() == y_val).float().mean())


def objective(trial: optuna.Trial) -> float:
    """Train epoch by epoch and let the schedule cut the trial at any rung."""
    eta = trial.suggest_float("eta", 1e-4, 3e-1, log=True)
    reg = trial.suggest_float("reg", 1e-6, 1e-1, log=True)
    width = trial.suggest_int("width", 4, 64)
    layers = trial.suggest_int("layers", 1, 3)
    batch = trial.suggest_categorical("batch", BATCH_SIZES)
    optimizer = trial.suggest_categorical("optimizer", OPTIMIZERS)

    X_train, y_train, X_val, y_val = DATA
    model = network(width, layers, SEED)
    loss_fn = nn.BCEWithLogitsLoss()
    if optimizer == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=eta, weight_decay=reg)
    else:
        opt = torch.optim.SGD(model.parameters(), lr=eta, momentum=0.9,
                              weight_decay=reg)
    generator = torch.Generator().manual_seed(SEED)

    accuracy = 0.0
    for epoch in range(MAX_EPOCHS):
        order = torch.randperm(len(X_train), generator=generator)
        for start in range(0, len(X_train), batch):
            picked = order[start:start + batch]
            opt.zero_grad()
            loss_fn(model(X_train[picked]).squeeze(1), y_train[picked]).backward()
            opt.step()

        accuracy = _accuracy(model, X_val, y_val)
        trial.report(accuracy, step=epoch)      # the schedule reads this
        if trial.should_prune():                # the rung says: stop
            raise optuna.TrialPruned()
    return accuracy


def run_study(n_trials: int = N_TRIALS, seed: int = SEED) -> optuna.Study:
    """The whole point of the script: sampler decides WHERE, pruner HOW LONG."""
    torch.set_num_threads(1)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        # HyperbandPruner picks the bracket of a trial from crc32 of the STUDY
        # NAME plus the trial number. Optuna generates a random study name, so
        # without this argument the run is not reproducible.
        study_name="hyperparameters-pruning-demo",
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.HyperbandPruner(min_resource=1,
                                              max_resource=MAX_EPOCHS,
                                              reduction_factor=REDUCTION))
    study.optimize(objective, n_trials=n_trials)
    return study


def epochs_spent(study: optuna.Study) -> int:
    """How many epochs the study actually paid for (a pruned trial paid less)."""
    return sum(len(trial.intermediate_values) for trial in study.trials)


def main() -> None:
    study = run_study()
    pruned = [t for t in study.trials
              if t.state == optuna.trial.TrialState.PRUNED]
    complete = [t for t in study.trials
                if t.state == optuna.trial.TrialState.COMPLETE]
    spent = epochs_spent(study)
    full = N_TRIALS * MAX_EPOCHS

    print("=" * 70)
    print("PRUNING: the rung stops the candidate, not the training")
    print("=" * 70)
    print(f"  trials:        {len(study.trials)}"
          f"  ({len(pruned)} pruned, {len(complete)} run to the end)")
    print(f"  epochs paid:   {spent} instead of {full}"
          f"  ({full / spent:.1f}x cheaper)")
    print(f"  best accuracy: {study.best_value:.4f}"
          f"  (trial no. {study.best_trial.number + 1})")
    for key, value in study.best_params.items():
        print(f"    {key:<10} {value:.5g}" if isinstance(value, float)
              else f"    {key:<10} {value}")

    lengths = np.array([len(t.intermediate_values) for t in pruned])
    print("\n  where the pruned candidates died (epochs of training bought):")
    for rung, count in zip(*np.unique(lengths, return_counts=True)):
        print(f"    after {int(rung):>3} epoch(s): {int(count):>3} candidates")

    # Checks, not decoration: pruning must actually happen and must actually save.
    # The slide `Pruning in Optuna` types these three numbers by hand.
    assert (spent, len(pruned)) == (1674, 42), (spent, len(pruned))
    assert int((lengths <= 4).sum()) == 34, int((lengths <= 4).sum())
    assert pruned, "no trial was pruned — pruning had no effect at all"
    assert complete, "every trial was pruned — the schedule would make no sense"
    assert full / spent > 2.5, (
        f"pruning should save at least 2.5x, saved only {full / spent:.1f}x")
    assert lengths.min() <= REDUCTION, (
        "the cheapest pruned trial should end at one of the first rungs, "
        f"ended only after {int(lengths.min())} epochs")

    print("\n  Careful: this is still a validation accuracy, and it comes from")
    print("  a single split, so it is noisier than the cross-validated number")
    print("  in optuna_hyperparameters.py. Pruning changes the price of the")
    print("  search, never the honesty of the score.")


if __name__ == "__main__":
    main()
