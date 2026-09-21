"""The small convolutional network of part 3 - the single source of its numbers.

The task: four species of leaf (apple, cherry, chestnut, maple) from a
photograph, 700 training images of 96 x 96 pixels. The dataset is the one from
the MLR course (`MLR-public/exercises/data/ex08_leaves_images.zip`); it is
vendored unchanged in this lab's data directory.

Everything here is written so that **one training is cheap**: the images are
read by a custom Dataset and resized to 48 x 48, the network has three
convolutional blocks, and the default budget is 8 epochs. One training takes
seconds to tens of seconds on two CPU threads, depending on network width and
augmentation, so a search of a dozen trials fits inside a lab.

The notebook of the lab contains a copy of this code (it has to run in Colab,
where the repository does not exist); `build_notebooks.py --check` compares the
copy with this module.

Run from the repository root:

    uv run python labs/hyperparameters/code/leaf_cnn.py            # data, one training
    uv run python labs/hyperparameters/code/leaf_cnn.py --search   # + both searches
"""

import os
import math
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

import torch
import torch.nn as nn
from leaf_data import CLASSES, IMAGE_SIZE, make_loaders, split_leaf_samples

DATA_URL = "https://raw.githubusercontent.com/tomasvicar/AUI-public/master/labs/hyperparameters/data/leaf_species_images.zip"
DATA_ROOT = Path(__file__).resolve().parents[1] / "data"
DATA_ARCHIVE = DATA_ROOT / "leaf_species_images.zip"

N_EPOCHS = 8             # fixed part of the budget, not a tuned hyperparameter


# ------------------------------------------------------------------ the data


def download_leaves(root: Path = DATA_ROOT) -> Path:
    """Download and unzip the photographs once; return the folder with them."""
    root.mkdir(parents=True, exist_ok=True)
    leaves = root / "leaves"
    if not leaves.exists():
        archive = root / "leaf_species_images.zip"
        if DATA_ARCHIVE.is_file():
            archive = DATA_ARCHIVE
        elif not archive.is_file():
            urllib.request.urlretrieve(DATA_URL, archive)
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(root)
    return leaves


def load_leaves(root: Path = DATA_ROOT):
    """Return fixed training, validation and test file/label lists."""
    return split_leaf_samples(download_leaves(root))


# --------------------------------------------------------------- the network


class LeafCNN(nn.Module):
    """Three blocks of conv 3x3 + ReLU + max pool, then one linear layer.

    `n_filters` is the width of the first block; the next two double it. The
    global average pool at the end means the number of parameters of the last
    layer does not depend on the size of the image.
    """

    def __init__(self, n_filters: int = 16, dropout: float = 0.0,
                 n_classes: int = len(CLASSES)):
        super().__init__()
        layers, channels = [], 3
        for block in range(3):
            width = n_filters * 2**block
            layers += [nn.Conv2d(channels, width, kernel_size=3, padding=1),
                       nn.ReLU(), nn.MaxPool2d(2)]
            channels = width
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                        nn.Dropout(dropout),
                                        nn.Linear(channels, n_classes))

    def forward(self, x):
        return self.classifier(self.features(x))


def accuracy(model, loader, device) -> float:
    """Count correct predictions across all batches, including a short last one."""
    model.eval()
    correct, total = 0, 0
    with torch.inference_mode():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            correct += (model(images).argmax(1) == labels).sum().item()
            total += len(labels)
    return correct / total


def train_and_validate(train_samples, valid_samples, *,
                       learning_rate: float = 1e-3, weight_decay: float = 0.0,
                       n_filters: int = 16, dropout: float = 0.0,
                       batch_size: int = 32, rotation_deg: float = 0.0,
                       color_jitter: float = 0.0, n_epochs: int = N_EPOCHS,
                       seed: int = 0, device=None,
                       history: list | None = None) -> float:
    """Train one configuration from scratch and return its validation accuracy.

    **This is one evaluation of the black box.** Seven hyperparameters in, one
    number out; everything else - the data, the number of epochs, the seed - is
    held fixed, so two calls differ only by what was tuned.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)

    model = LeafCNN(n_filters=int(n_filters), dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                 weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    loader, valid_loader = make_loaders(train_samples, valid_samples, batch_size,
                                        rotation_deg, color_jitter, seed)

    for _ in range(int(n_epochs)):
        model.train()
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
        if history is not None:
            history.append(accuracy(model, valid_loader, device))

    return accuracy(model, valid_loader, device)


# ------------------------------------------------------- the reference runs


DEFAULT_CONFIG = dict(learning_rate=1e-3, weight_decay=0.0, n_filters=16,
                      dropout=0.0, batch_size=32, rotation_deg=0.0, color_jitter=0.0)

# The two positive scale parameters are searched in logarithms. The box
# interface rounds the filter count and fixes batch size; Optuna uses explicit
# integer/categorical suggestions. Both tune the two augmentation strengths.
SPACE = {
    "log_learning_rate": (-4.0, math.log10(3e-2)),     # 1e-4 ... 3e-2
    "log_weight_decay": (-6.0, -2.0),      # 1e-6 ... 1e-2
    "n_filters": (8.0, 24.0),              # integer, rounded inside the objective
    "dropout": (0.0, 0.5),
    "rotation_deg": (0.0, 45.0),
    "color_jitter": (0.0, 0.4),
}
BATCH_SIZES = [16, 32, 64]                 # the discrete parameter added in Optuna


def bayes_opt_search(data, n_init=4, n_iter=8, seed=0, verbose=2):
    """`bayes_opt` over model and augmentation hyperparameters."""
    from bayes_opt import BayesianOptimization

    train_samples, valid_samples = data[:2]

    def objective(log_learning_rate, log_weight_decay, n_filters, dropout,
                  rotation_deg, color_jitter):
        return train_and_validate(
            train_samples, valid_samples,
            learning_rate=10**log_learning_rate,
            weight_decay=10**log_weight_decay,
            n_filters=int(round(n_filters)), dropout=dropout,
            rotation_deg=rotation_deg, color_jitter=color_jitter)

    optimizer = BayesianOptimization(f=objective, pbounds=SPACE,
                                     random_state=seed, verbose=verbose)
    optimizer.maximize(init_points=n_init, n_iter=n_iter)
    return optimizer


def optuna_search(data, n_trials=12, seed=0):
    """The same space in Optuna, plus the batch size as a discrete parameter."""
    import optuna

    train_samples, valid_samples = data[:2]

    def objective(trial):
        return train_and_validate(
            train_samples, valid_samples,
            learning_rate=trial.suggest_float("learning_rate", 1e-4, 3e-2, log=True),
            weight_decay=trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
            n_filters=trial.suggest_int("n_filters", 8, 24),
            dropout=trial.suggest_float("dropout", 0.0, 0.5),
            rotation_deg=trial.suggest_float("rotation_deg", 0.0, 45.0),
            color_jitter=trial.suggest_float("color_jitter", 0.0, 0.4),
            batch_size=trial.suggest_categorical("batch_size", BATCH_SIZES))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    return study


def main() -> None:
    torch.set_num_threads(min(2, os.cpu_count() or 1))   # about what Colab gives
    data = load_leaves()
    train_samples, valid_samples, test_samples = data
    print(f"train {len(train_samples)}  validation {len(valid_samples)}  test {len(test_samples)} "
          f"images of {IMAGE_SIZE} x {IMAGE_SIZE}, {len(CLASSES)} classes")
    print("class counts in the training set:",
          {c: sum(label == i for _, label in train_samples) for i, c in enumerate(CLASSES)})

    model = LeafCNN(**{k: DEFAULT_CONFIG[k] for k in ("n_filters", "dropout")})
    print(f"parameters of the default network: "
          f"{sum(p.numel() for p in model.parameters())}")

    history: list[float] = []
    start = time.time()
    default = train_and_validate(train_samples, valid_samples,
                                 history=history, **DEFAULT_CONFIG)
    seconds = time.time() - start
    print(f"\ndefault configuration {DEFAULT_CONFIG}")
    print(f"  validation accuracy after every epoch: "
          f"{' '.join(f'{a:.3f}' for a in history)}")
    print(f"  validation accuracy {default:.3f} in {seconds:.1f} s "
          f"({seconds / N_EPOCHS:.1f} s per epoch)")

    if "--search" not in sys.argv:
        print("\n(run with --search for both searches, several minutes on CPU)")
        return

    start = time.time()
    optimizer = bayes_opt_search(data, verbose=0)
    best = optimizer.max
    print(f"\nbayes_opt, 4 + 8 trainings in {time.time() - start:.0f} s")
    print(f"  best validation accuracy {best['target']:.3f}")
    print(f"  at learning rate {10**best['params']['log_learning_rate']:.2e}, "
          f"weight decay {10**best['params']['log_weight_decay']:.2e}, "
          f"{int(round(best['params']['n_filters']))} filters, "
          f"dropout {best['params']['dropout']:.2f}, "
          f"rotation ±{best['params']['rotation_deg']:.1f} degrees, "
          f"color jitter {best['params']['color_jitter']:.2f}")

    start = time.time()
    study = optuna_search(data)
    print(f"\noptuna, 12 trials in {time.time() - start:.0f} s")
    print(f"  best validation accuracy {study.best_value:.3f}")
    print(f"  at {study.best_params}")

    on_test = train_and_validate(train_samples, test_samples,
                                 **study.best_params)
    print(f"\nthe winner retrained and measured ONCE on the test set: {on_test:.3f}"
          f"  (validation said {study.best_value:.3f})")


if __name__ == "__main__":
    main()
