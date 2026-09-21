"""Figures for the budget block: successive halving, Hyperband, pruning.

Generates six figures with the ``budget-`` prefix:

* ``budget-successive-halving.svg`` — fifty actually trained configurations,
  rungs at 1, 10 and 100 epochs, and the cost in epochs against letting
  everyone run to the end;
* ``budget-hyperband-brackets.svg`` — five bracket schedules of Hyperband
  ($\\rho = 3$, $R = 81$) side by side;
* ``budget-early-vs-prune.svg`` — on the left early stopping (one training,
  validation loss), on the right pruning (more candidates, weak ones cut);
* ``budget-slow-starter.svg`` — the risk of the method: a configuration that
  starts the slowest but wins in the end;
* ``budget-bill-search.svg`` and ``budget-bill.svg`` — the "bill for
  tuning", the frame of the whole lecture: how many trainings grid, random,
  Bayesian optimization and successive halving want; the first version (end
  of the search block) still has its last row open, the second (Takeaways)
  is complete.

**The curves are not drawn by eye.** All learning curves are real runs of
the network from ``optuna_hyperparameters.py`` (functions ``cell_data`` and
``network``) on the same synthetic cells; configurations are drawn from the
same space the lecture's study tunes. Everything is deterministic — fixed
seeds for the data, the choice of configurations, and the training.

The run takes about a minute (fifty trainings of a hundred epochs each plus
one long training on a small sample).

Run from the repository root:

    MPLBACKEND=Agg uv run python lectures/hyperparameters/code/figures_budget.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common as s                                       # noqa: E402
from optuna_hyperparameters import (N_FOLDS, N_TRIALS,     # noqa: E402
                                    cell_data, network)

# --- Experiment setup (everything deterministic) ----------------------------

SEED_SPLIT = 11      # split of the cells into training and validation part
SEED_CONFIGS = 3     # drawing fifty configurations from the space
SEED_TRAINING = 7    # weight initialization and batch order

N_CONFIGS = 50
MAX_EPOCH = 100
N_TRAIN = 120             # small training part, so a hundred epochs mean something
N_VAL = 300

# Rungs of successive halving: how many epochs, and how many configurations survive them.
RUNGS = (1, 10, 100)
SURVIVE = (50, 10, 3)

# Hyperband: the canonical example from Li et al. (2018).
RHO = 3                 # reduction factor (eta in the paper; eta here is the learning rate)
R_MAX = 81

LABEL_BOX = {"facecolor": "white", "edgecolor": "none", "alpha": 0.82,
            "boxstyle": "round,pad=0.22"}
GREY_CURVE = "#c3cfd6"
GREY_DARK = "#7e8f99"


# --- Real training -----------------------------------------------------------


def _split_data(n_train: int = N_TRAIN, n_val: int = N_VAL):
    """Fixed split of the synthetic cells into a training and validation part."""
    X, y = cell_data()
    order = np.random.default_rng(SEED_SPLIT).permutation(len(X))
    i_train = order[:n_train]
    i_val = order[n_train:n_train + n_val]
    mean, std = X[i_train].mean(axis=0), X[i_train].std(axis=0)
    tensor = lambda i: (torch.tensor((X[i] - mean) / std),      # noqa: E731
                        torch.tensor(y[i]))
    return tensor(i_train), tensor(i_val)


def _optimizer(model: nn.Module, config: dict):
    if config["optimizer"] == "adam":
        return torch.optim.Adam(model.parameters(), lr=config["eta"],
                                weight_decay=config["reg"])
    return torch.optim.SGD(model.parameters(), lr=config["eta"], momentum=0.9,
                           weight_decay=config["reg"])


def learning_curve(config: dict, data, epoch: int = MAX_EPOCH, loss: bool = False):
    """Trains the network and returns the validation value **after every epoch**.

    With ``loss=False`` it returns the validation accuracy, with
    ``loss=True`` a pair (training loss, validation loss) — the latter is
    needed for the early-stopping figure.
    """
    (X_train, y_train), (X_val, y_val) = data
    model = network(config["width"], config["layers"], SEED_TRAINING)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = _optimizer(model, config)
    generator = torch.Generator().manual_seed(SEED_TRAINING)

    train_values, val_values = [], []
    for _ in range(epoch):
        order = torch.randperm(len(X_train), generator=generator)
        for start in range(0, len(X_train), config["batch"]):
            picked = order[start:start + config["batch"]]
            optimizer.zero_grad()
            loss_fn(model(X_train[picked]).squeeze(1), y_train[picked]).backward()
            optimizer.step()
        with torch.no_grad():
            if loss:
                train_values.append(float(loss_fn(model(X_train).squeeze(1), y_train)))
                val_values.append(float(loss_fn(model(X_val).squeeze(1), y_val)))
            else:
                logit = model(X_val).squeeze(1)
                val_values.append(float(((logit > 0).float() == y_val).float().mean()))
    if loss:
        return np.array(train_values), np.array(val_values)
    return np.array(val_values)


def random_configurations(how_many: int = N_CONFIGS) -> list[dict]:
    """Configurations drawn from **the same space** the lecture's study tunes."""
    rng = np.random.default_rng(SEED_CONFIGS)
    configs = []
    for _ in range(how_many):
        configs.append({
            "eta": float(10 ** rng.uniform(-4.0, np.log10(0.3))),
            "reg": float(10 ** rng.uniform(-6.0, -1.0)),
            "width": int(rng.integers(4, 65)),
            "layers": int(rng.integers(1, 4)),
            "batch": int(rng.choice([8, 16, 32, 64])),
            "optimizer": str(rng.choice(["adam", "sgd"])),
        })
    return configs


_CURVES: np.ndarray | None = None


def configuration_curves() -> np.ndarray:
    """Matrix (configuration × epoch) of validation accuracies; computed once."""
    global _CURVES
    if _CURVES is None:
        torch.set_num_threads(1)
        data = _split_data()
        _CURVES = np.array([learning_curve(k, data) for k in random_configurations()])
    return _CURVES


# --- Schedules: successive halving and Hyperband -----------------------------


def survivors(curves: np.ndarray, rungs=RUNGS, survive=SURVIVE) -> list[np.ndarray]:
    """Indices of the configurations surviving each rung (ordered by rung)."""
    alive = np.arange(len(curves))[:survive[0]]
    result = [alive]
    for epoch, how_many in zip(rungs[:-1], survive[1:]):
        score = curves[alive, epoch - 1]
        alive = alive[np.argsort(-score, kind="stable")[:how_many]]
        result.append(alive)
    return result


def schedule_cost(counts, budgets) -> int:
    """Cost of the schedule in epochs when training **resumes from a checkpoint**.

    A configuration that drops out at rung $i$ consumed `budgets[i]` epochs
    in total — not the sum of every rung it passed through. That is why the
    cost is `(how many end at the rung) × (budget of the rung)`.
    """
    counts = list(counts) + [0]
    return int(sum((counts[i] - counts[i + 1]) * budgets[i]
                   for i in range(len(budgets))))


def hyperband_brackets(rho: int = RHO, r_max: int = R_MAX):
    """Rung schedules of every Hyperband bracket, following Li et al. (2018)."""
    s_max = int(round(np.log(r_max) / np.log(rho)))
    budget = (s_max + 1) * r_max
    brackets = []
    for s_index in range(s_max, -1, -1):
        n = int(np.ceil(budget / r_max * rho ** s_index / (s_index + 1)))
        r = r_max * rho ** (-s_index)
        counts = [int(np.floor(n * rho ** (-i))) for i in range(s_index + 1)]
        budgets = [int(round(r * rho ** i)) for i in range(s_index + 1)]
        brackets.append({"s": s_index, "counts": counts, "budgets": budgets,
                         "cost": schedule_cost(counts, budgets)})
    return brackets, budget


# --- Figure 1: successive halving --------------------------------------------


def figure_successive_halving() -> None:
    """Fifty real curves, three rungs, and the bill in epochs."""
    curves = configuration_curves()
    epochs = np.arange(1, MAX_EPOCH + 1)
    rounds = survivors(curves)

    end = np.full(len(curves), RUNGS[0])
    end[rounds[1]] = RUNGS[1]
    end[rounds[2]] = RUNGS[2]
    winner_sh = int(rounds[2][np.argmax(curves[rounds[2], -1])])
    winner_all = int(np.argmax(curves[:, -1]))
    gap = float(curves[winner_all, -1] - curves[winner_sh, -1])

    cost = schedule_cost(SURVIVE, RUNGS)
    full = N_CONFIGS * RUNGS[-1]

    assert cost == 410, cost
    assert full == 5000, full
    # The cut really paid off: the winner of the bracket schedule is less
    # than one validation-set sample away from the truly best configuration.
    assert gap <= 1.0 / N_VAL + 1e-6, gap   # exactly one validation sample
    assert len(rounds[1]) == 10 and len(rounds[2]) == 3

    fig, (left, right) = plt.subplots(
        1, 2, figsize=(11.4, 4.3), gridspec_kw={"width_ratios": [2.15, 1.0]})

    for i in range(len(curves)):
        left.plot(epochs, curves[i], color=GREY_CURVE, lw=0.9, alpha=0.75, zorder=0)

    for i in range(len(curves)):
        mask = epochs <= end[i]
        if end[i] == RUNGS[0]:
            color, width, z = GREY_DARK, 1.4, 1
        elif end[i] == RUNGS[1]:
            color, width, z = s.BLUE, 2.0, 2
        else:
            color, width, z = s.TEAL, 2.8, 3
        if i == winner_sh:
            color, width, z = s.ORANGE, 3.4, 4
        left.plot(epochs[mask], curves[i, mask], color=color, lw=width,
                 zorder=z, solid_capstyle="round")
        left.plot([end[i]], [curves[i, end[i] - 1]], "o", color=color,
                 ms=4.5 if end[i] == RUNGS[0] else 6.5, zorder=z + 1)

    labels = ("50 configs × 1 epoch", "top 10 × 10 epochs",
             "top 3 × 100 epochs")
    # The label of the first rung goes below, the second and third above —
    # at the top, three labels side by side would not fit without their
    # white backgrounds overlapping.
    for epoch, label, side, shift, height, anchor in zip(
            RUNGS, labels, ("left", "right", "right"), (1.07, 0.93, 0.93),
            (0.245, 0.985, 0.985), ("bottom", "top", "top")):
        left.axvline(epoch, color=s.RED, ls="--", lw=1.3, alpha=0.6, zorder=0)
        left.text(epoch * shift, height, label, fontsize=11.5, color=s.RED,
                 va=anchor, ha=side, bbox=LABEL_BOX, zorder=6)

    left.text(0.99, 0.04, "pale: where a cut curve would have ended up",
             transform=left.transAxes, ha="right", va="bottom", fontsize=10.5,
             color="#6b7b85", bbox=LABEL_BOX, zorder=6)
    left.annotate("winner of the schedule",
                 xy=(MAX_EPOCH, curves[winner_sh, -1]), xytext=(9, 0.52),
                 fontsize=11.5, color=s.ORANGE, bbox=LABEL_BOX, zorder=7,
                 arrowprops={"arrowstyle": "->", "color": s.ORANGE, "lw": 1.3})

    left.set_xscale("log")
    left.set_xlim(0.88, 130)
    left.set_ylim(0.22, 1.03)
    left.set_xticks([1, 3, 10, 30, 100])
    left.set_xticklabels(["1", "3", "10", "30", "100"])
    left.set_xlabel("training epoch (log scale)", fontsize=12)
    left.set_ylabel("validation accuracy", fontsize=12)
    left.set_title("50 real runs, cut after 1 and after 10 epochs",
                  fontsize=13.5, color=s.BLUE)
    left.grid(alpha=0.25)

    # Right panel: the bill in epochs.
    parts = [(N_CONFIGS - SURVIVE[1]) * RUNGS[0],
            (SURVIVE[1] - SURVIVE[2]) * RUNGS[1],
            SURVIVE[2] * RUNGS[2]]
    assert sum(parts) == cost
    colors = (GREY_DARK, s.BLUE, s.TEAL)
    start = 0.0
    for part, color in zip(parts, colors):
        right.barh(1, part, left=start, height=0.5, color=color,
                  edgecolor="white", linewidth=1.0)
        start += part
    right.barh(0, full, height=0.5, color=GREY_CURVE, edgecolor="white",
              linewidth=1.0)

    right.text(cost + 160, 1, f"{cost} epochs", va="center", ha="left",
              fontsize=13, color=s.BLUE, fontweight="bold")
    right.text(cost + 160, 0.62, "40 × 1  +  7 × 10  +  3 × 100", va="center",
              ha="left", fontsize=10.5, color=GREY_DARK)
    right.text(full / 2, 0, f"{full} epochs", va="center", ha="center",
              fontsize=13, color=s.BLUE, zorder=3)
    right.text(full / 2, -0.38, "50 × 100", va="center", ha="center",
              fontsize=10.5, color=GREY_DARK, zorder=3)
    right.set_yticks([0, 1])
    right.set_yticklabels(["every candidate\nruns to the end",
                          "successive\nhalving"], fontsize=11)
    right.text(3050, -0.95,
              f"the schedule ends {gap:.3f} below the best of all 50 —\n"
              f"that gap is one single validation sample",
              va="center", ha="center", fontsize=10.5, color=GREY_DARK)
    right.set_xlim(0, 6100)
    right.set_ylim(-1.25, 1.6)
    right.set_xlabel("total training epochs", fontsize=12)
    right.set_title(f"{full // cost}× cheaper, almost the same winner",
                   fontsize=13.5, color=s.BLUE)
    right.grid(axis="x", alpha=0.25)
    for side in ("top", "right", "left"):
        right.spines[side].set_visible(False)

    fig.tight_layout()
    s.save_figure(fig, "budget-successive-halving")
    print(f"  successive halving: {cost} epochs instead of {full}; schedule winner "
          f"{curves[winner_sh, -1]:.4f}, best of fifty {curves[winner_all, -1]:.4f}")


# --- Figure 2: Hyperband brackets --------------------------------------------


def figure_hyperband_brackets() -> None:
    """Five bracket schedules side by side — from the most aggressive to random search."""
    brackets, budget = hyperband_brackets()
    costs = [b["cost"] for b in brackets]

    assert len(brackets) == 5, len(brackets)
    assert [len(b["counts"]) for b in brackets] == [5, 4, 3, 2, 1]
    assert brackets[0]["counts"][0] == 81 and brackets[-1]["counts"] == [5]
    assert brackets[-1]["budgets"] == [R_MAX]
    # Design property of Hyperband: every bracket spends a comparable budget.
    assert max(costs) / min(costs) < 1.5, costs
    assert sum(costs) == 1581, costs

    fig, ax = plt.subplots(figsize=(7.8, 3.6))
    colors = [s.RED, s.ORANGE, s.TEAL, s.BLUE, GREY_DARK]

    for row, (bracket, color) in enumerate(zip(brackets, colors)):
        y = len(brackets) - 1 - row
        budgets = bracket["budgets"]
        counts = bracket["counts"]
        ax.plot(budgets, [y] * len(budgets), color=color, lw=1.6,
               alpha=0.55, zorder=1, solid_capstyle="round")
        for r, n in zip(budgets, counts):
            ax.scatter([r], [y], s=26 + 5.6 * n, color=color, alpha=0.85,
                      zorder=2, edgecolors="white", linewidths=1.0)
            ax.text(r, y + 0.30, str(n), ha="center", va="bottom",
                   fontsize=9, color=color, zorder=3)
        ax.text(115, y, f"{bracket['cost']} epochs", va="center", ha="left",
               fontsize=9, color=color)

    ax.text(115, len(brackets) - 0.12, "cost", va="center", ha="left",
           fontsize=8.6, color=s.BLUE, style="italic")
    ax.text(1, len(brackets) - 0.12, "number of configurations still alive",
           va="center", ha="left", fontsize=8.6, color=s.BLUE, style="italic")

    # Both punch lines — the most aggressive bracket, and that the bottom one
    # is random search — are spoken as text on the slide; annotations would
    # only fit into the figure through the count labels.
    ax.set_xscale("log")
    ax.set_xlim(0.7, 300)
    ax.set_ylim(-0.75, len(brackets) + 0.18)
    ax.set_xticks([1, 3, 9, 27, 81])
    ax.set_xticklabels(["1", "3", "9", "27", "81"])
    ax.set_yticks(range(len(brackets)))
    ax.set_yticklabels([f"bracket $s = {b['s']}$" for b in reversed(brackets)],
                       fontsize=9)
    ax.set_xlabel("epochs given to one configuration (log scale)", fontsize=9.4)
    ax.set_title(f"Hyperband with $\\rho = {RHO}$ and $R = {R_MAX}$: five schedules "
                 f"side by side, {sum(costs)} epochs in total",
                 fontsize=10.5, color=s.BLUE)
    ax.grid(axis="x", alpha=0.25)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)

    fig.tight_layout()
    s.save_figure(fig, "budget-hyperband-brackets")
    print(f"  hyperband: brackets {[b['counts'][0] for b in brackets]}, "
          f"costs {costs}, total {sum(costs)} epochs (budget B = {budget})")


# --- Figure 3: early stopping versus pruning ---------------------------------

CONFIG_OVERFIT = {"eta": 0.003, "reg": 0.0, "width": 64, "layers": 3,
                  "batch": 16, "optimizer": "adam"}
EPOCH_OVERFIT = 80
N_IN_PANEL = 12


def figure_early_vs_prune() -> None:
    """On the left training stops, on the right a candidate does."""
    torch.set_num_threads(1)
    data = _split_data(n_train=150)
    train, val = learning_curve(CONFIG_OVERFIT, data, epoch=EPOCH_OVERFIT, loss=True)
    epochs_ez = np.arange(1, EPOCH_OVERFIT + 1)
    best = int(np.argmin(val))

    assert best + 1 == 11, (
        "the slide `Early stopping is not pruning` says epoch 11 of "
        f"{EPOCH_OVERFIT}, the figure computes {best + 1}")
    assert 3 <= best <= EPOCH_OVERFIT - 20, best
    # The validation loss really does get much worse after stopping, the training one keeps falling.
    assert val[-1] > 1.5 * val[best], (val[best], val[-1])
    assert train[-1] < 0.4 * train[0], (train[0], train[-1])

    curves = configuration_curves()[:N_IN_PANEL]
    epochs = np.arange(1, MAX_EPOCH + 1)
    survive = (N_IN_PANEL, 4, 2)
    rounds = survivors(curves, RUNGS, survive)
    end = np.full(N_IN_PANEL, RUNGS[0])
    end[rounds[1]] = RUNGS[1]
    end[rounds[2]] = RUNGS[2]

    assert int((end == RUNGS[-1]).sum()) == 2 < N_IN_PANEL
    assert sorted(set(end.tolist())) == list(RUNGS)

    fig, (left, right) = plt.subplots(1, 2, figsize=(11.2, 4.0))

    left.plot(epochs_ez, train, color=s.BLUE, lw=2.2, label="training loss")
    left.plot(epochs_ez, val, color=s.RED, lw=2.4, label="validation loss")
    left.axvline(best + 1, color=s.TEAL, lw=1.8, ls="--")
    left.plot([best + 1], [val[best]], "o", color=s.TEAL, ms=9,
             zorder=5)
    left.axvspan(best + 1, EPOCH_OVERFIT, color=s.RED, alpha=0.06,
                zorder=0)
    left.text(best + 3, val.max() * 0.94,
             f"stop here (epoch {best + 1})\nand keep these weights",
             fontsize=11.5, color=s.TEAL, va="top", ha="left", bbox=LABEL_BOX)
    left.text(EPOCH_OVERFIT - 2, val.max() * 0.46,
             "everything past the line\nis memorising the training set",
             fontsize=11, color=s.RED, va="center", ha="right", bbox=LABEL_BOX)
    left.set_xlim(0, EPOCH_OVERFIT)
    left.set_ylim(0, val.max() * 1.08)
    left.set_xlabel("training epoch", fontsize=12)
    left.set_ylabel("cross-entropy loss", fontsize=12)
    left.set_title("Early stopping: one candidate, one run", fontsize=13.5,
                  color=s.BLUE)
    left.legend(loc="lower left", fontsize=11, framealpha=0.9)
    left.grid(alpha=0.25)

    for i in range(N_IN_PANEL):
        right.plot(epochs, curves[i], color=GREY_CURVE, lw=0.9, alpha=0.75,
                  zorder=0)
    for i in range(N_IN_PANEL):
        mask = epochs <= end[i]
        if end[i] == RUNGS[0]:
            color, width, z = GREY_DARK, 1.5, 1
        elif end[i] == RUNGS[1]:
            color, width, z = s.BLUE, 2.1, 2
        else:
            color, width, z = s.TEAL, 2.9, 3
        right.plot(epochs[mask], curves[i, mask], color=color, lw=width,
                  zorder=z, solid_capstyle="round")
        right.plot([end[i]], [curves[i, end[i] - 1]], "o", color=color,
                  ms=6.0, zorder=z + 1)
    for epoch in RUNGS[:-1]:
        right.axvline(epoch, color=s.RED, ls="--", lw=1.3, alpha=0.6,
                     zorder=0)
    right.text(1.12, 0.255, "8 candidates\ndropped here", fontsize=11,
              color=s.RED, va="bottom", ha="left", bbox=LABEL_BOX, zorder=6)
    right.text(11, 0.255, "2 more\ndropped here", fontsize=11, color=s.RED,
              va="bottom", ha="left", bbox=LABEL_BOX, zorder=6)
    right.set_xscale("log")
    right.set_xlim(0.88, 130)
    right.set_ylim(0.22, 0.96)
    right.set_xticks([1, 3, 10, 30, 100])
    right.set_xticklabels(["1", "3", "10", "30", "100"])
    right.set_xlabel("training epoch (log scale)", fontsize=12)
    right.set_ylabel("validation accuracy", fontsize=12)
    right.set_title("Pruning: twelve candidates, ten of them stopped",
                   fontsize=13.5, color=s.BLUE)
    right.grid(alpha=0.25)

    fig.tight_layout()
    s.save_figure(fig, "budget-early-vs-prune")
    print(f"  early stopping: minimum validation loss at epoch {best + 1} "
          f"({val[best]:.3f}), at the end {val[-1]:.3f}")


# --- Figure 4: slow starter ---------------------------------------------------

SLOW = 24                       # index of the configuration with the slowest start
RIVALS = (21, 28, 34, 11)        # four configurations that start better and end worse


def figure_slow_starter() -> None:
    """A configuration the first rung would cut — and which wins in the end."""
    curves = configuration_curves()
    epochs = np.arange(1, MAX_EPOCH + 1)
    third = MAX_EPOCH // 3

    chosen = (SLOW,) + RIVALS
    third_mean = curves[list(chosen), :third].mean(axis=1)
    ends = curves[list(chosen), -1]

    assert third_mean[0] < third_mean[1:].min() - 1e-6, third_mean
    assert curves[SLOW, 0] < curves[list(RIVALS), 0].min() - 1e-6
    assert ends[0] > ends[1:].max() + 1e-6, ends
    # And most importantly: the first rung really would cut it — after one
    # epoch it is deep below the cutoff of the first cut (actually 45th of
    # 50, i.e. in the bottom tenth).
    rank = int(np.argsort(-curves[:, 0], kind="stable").tolist().index(SLOW))
    rank_end = int(np.argsort(-curves[:, -1], kind="stable").tolist().index(SLOW))
    assert rank >= SURVIVE[1], rank
    # and after a hundred epochs it is instead in the best tenth of all fifty
    assert rank_end < 5, rank_end
    # Both ranks are spelled out on the Hyperband slide, so they are checked
    # exactly: if any seed ever changed, this fails here, not on the screen.
    assert rank + 1 == 45, (
        f"the slide claims 45th of 50 after the first epoch, got {rank + 1}.")
    assert rank_end + 1 == 4, (
        f"the slide claims 4th of 50 after a hundred epochs, got {rank_end + 1}.")

    fig, ax = plt.subplots(figsize=(7.0, 3.5))

    for i, rival in enumerate(RIVALS):
        ax.plot(epochs, curves[rival], color=s.BLUE, lw=1.6, alpha=0.75,
               zorder=2, label="fast starters" if i == 0 else None)
    ax.plot(epochs, curves[SLOW], color=s.ORANGE, lw=2.6, zorder=3,
           label="slow starter")

    ax.axvline(RUNGS[0], color=s.RED, ls="--", lw=1.5, alpha=0.75, zorder=1)
    ax.plot([1], [curves[SLOW, 0]], "o", color=s.RED, ms=9, zorder=4)
    ax.annotate(f"{rank + 1}th of all 50 after one epoch —\nthe first rung cuts it here",
               xy=(1, curves[SLOW, 0]), xytext=(1.6, 0.36), fontsize=9,
               color=s.RED, bbox=LABEL_BOX, zorder=6,
               arrowprops={"arrowstyle": "->", "color": s.RED, "lw": 1.3})
    ax.annotate(f"{rank_end + 1}th of all 50\nafter 100 epochs ({ends[0]:.3f})",
               xy=(88, ends[0]), xytext=(5.5, 0.60), fontsize=9,
               color=s.ORANGE, bbox=LABEL_BOX, zorder=6,
               arrowprops={"arrowstyle": "->", "color": s.ORANGE, "lw": 1.3})
    ax.axvspan(0.88, third, color=s.GREY, alpha=0.7, zorder=0)
    ax.text(1.02, 0.975, "first third of training", fontsize=8.6,
           color=GREY_DARK, va="center", ha="left", zorder=1)

    ax.set_xscale("log")
    ax.set_xlim(0.88, 115)
    ax.set_ylim(0.30, 1.02)
    ax.set_xticks([1, 3, 10, 30, 100])
    ax.set_xticklabels(["1", "3", "10", "30", "100"])
    ax.set_xlabel("training epoch (log scale)", fontsize=9.4)
    ax.set_ylabel("validation accuracy", fontsize=9.4)
    ax.set_title("The risk of cutting early: a slow start is not a bad configuration",
                fontsize=10.5, color=s.BLUE)
    ax.legend(loc="lower right", fontsize=8.6, framealpha=0.9)
    ax.grid(alpha=0.25)

    fig.tight_layout()
    s.save_figure(fig, "budget-slow-starter")
    print(f"  slow starter: configuration {SLOW} has {curves[SLOW, 0]:.3f} "
          f"after the first epoch (rank {rank + 1} of {N_CONFIGS}), "
          f"{curves[SLOW, -1]:.3f} after a hundred epochs (rank {rank_end + 1})")


# --- Figure 5: the bill for tuning --------------------------------------------
#
# No number is typed by hand: the grid is 10 values for 5 hyperparameters
# (slide `Grid search`), random search is the smallest n with
# 1 - 0.95^n >= 0.95 (slide `How many random trials`), Bayesian optimization
# is the study's budget times the number of folds, and successive halving is
# the schedule from the first figure of this script.

GRID_VALUES, GRID_DIMS = 10, 5


def _bill() -> list[dict]:
    grid = GRID_VALUES ** GRID_DIMS
    random = int(np.ceil(np.log(0.05) / np.log(0.95)))
    bayes = N_TRIALS * N_FOLDS
    halving = schedule_cost(SURVIVE, RUNGS)
    full = N_CONFIGS * RUNGS[-1]
    assert grid == 100_000 and random == 59 and bayes == 90, (grid, random, bayes)
    assert random * N_FOLDS == 177, random * N_FOLDS
    assert halving == 410 and full == 5000, (halving, full)
    return [
        {"name": "grid search", "value": grid, "color": s.RED,
         "caption": f"{GRID_VALUES} values × {GRID_DIMS} hyperparameters = {grid:,} trainings".replace(",", " ")},
        {"name": "random search", "value": random * N_FOLDS, "color": s.ORANGE,
         "caption": f"{random} trials × {N_FOLDS} folds = {random * N_FOLDS} trainings"},
        {"name": "Bayesian optimization\n(TPE, our study)", "value": bayes, "color": s.TEAL,
         "caption": f"{N_TRIALS} trials × {N_FOLDS} folds = {bayes} trainings"},
        {"name": "successive halving", "value": halving / RUNGS[-1], "color": s.BLUE,
         "shadow": full / RUNGS[-1],
         "caption": f"{N_CONFIGS} candidates screened for {halving} epochs instead of {full}\n"
                  f"= {halving / RUNGS[-1]:.1f} full runs instead of {full // RUNGS[-1]}"},
    ]


def figure_bill(complete: bool) -> None:
    """Bars of trainings on one log axis; `complete=False` leaves the last row open.

    Labels are **above** the bars, not to their right — otherwise the
    longest label would stretch the figure wide, and it would shrink into
    illegibility in a slide column.
    """
    items = _bill()
    budget = N_TRIALS * N_FOLDS
    fig, ax = plt.subplots(figsize=(6.6, 4.3))
    height = 0.34
    for row, item in enumerate(items):
        y = len(items) - 1 - row
        open_row = (not complete) and row == len(items) - 1
        if open_row:
            ax.barh(y, 2e5, height=height, color="none", edgecolor=GREY_DARK,
                   hatch="///", linewidth=0.8, zorder=1)
            ax.text(1.25, y + 0.24, "how long does each of them run?  — next block",
                   va="bottom", ha="left", fontsize=10, color=GREY_DARK,
                   style="italic", zorder=3, bbox=LABEL_BOX)
            continue
        if "shadow" in item:
            ax.barh(y, item["shadow"], height=height, color=GREY_CURVE, zorder=1)
        ax.barh(y, item["value"], height=height, color=item["color"], zorder=2)
        ax.text(1.25, y + 0.24, item["caption"], va="bottom", ha="left",
               fontsize=10, color=item["color"], zorder=3, linespacing=1.15,
               bbox=LABEL_BOX)
    ax.axvline(budget, color=s.BLUE, ls="--", lw=1.4, alpha=0.7, zorder=0)
    ax.text(budget * 1.12, -0.42, f"our budget: {budget} trainings",
           ha="left", va="center", fontsize=10, color=s.BLUE, style="italic")
    ax.set_xscale("log")
    ax.set_xlim(1, 2e5)
    ax.set_xticks([1, 10, 100, 1e3, 1e4, 1e5])
    ax.set_xticklabels(["1", "10", "100", "$10^3$", "$10^4$", "$10^5$"], fontsize=10)
    ax.set_ylim(-0.65, len(items) - 0.05)
    ax.set_yticks(range(len(items)))
    ax.set_yticklabels([p["name"] for p in reversed(items)], fontsize=10.5)
    ax.set_xlabel("trainings needed (log scale)", fontsize=11)
    ax.set_title("The bill for tuning" + ("" if complete else " — so far"),
                fontsize=12.5, color=s.BLUE, loc="left")
    ax.grid(axis="x", alpha=0.25)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    s.save_figure(fig, "budget-bill" if complete else "budget-bill-search")


def main() -> None:
    figure_bill(complete=False)
    figure_bill(complete=True)
    print("  bill for tuning: budget-bill-search.svg, budget-bill.svg")
    print("computing a hundred epochs for fifty configurations (real training)...")
    configuration_curves()
    figure_successive_halving()
    figure_hyperband_brackets()
    figure_early_vs_prune()
    figure_slow_starter()
    print("done: budget-successive-halving.svg, budget-hyperband-brackets.svg,\n"
          "      budget-early-vs-prune.svg, budget-slow-starter.svg,\n"
          "      budget-bill-search.svg, budget-bill.svg")


if __name__ == "__main__":
    main()
