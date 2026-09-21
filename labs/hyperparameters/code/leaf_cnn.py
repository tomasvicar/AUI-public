"""The small convolutional network of part 3 - the single source of its numbers.

The task: four species of leaf (apple, cherry, chestnut, maple) from a
photograph, 700 training images of 96 x 96 pixels. The dataset is the one from
the MLR course (`MLR-public/exercises/data/ex08_leaves_images.zip`); it is
downloaded once and cached.

Everything here is written so that **one training is cheap**: the images are
resized to 48 x 48 and held in memory as one tensor, the network has three
convolutional blocks, and the default budget is 8 epochs. One training takes
from 3 to 18 seconds on two CPU threads, depending on the width of the network,
so a search of a dozen trials fits inside a lab.

The notebook of the lab contains a copy of this code (it has to run in Colab,
where the repository does not exist); `build_notebooks.py --check` compares the
copy with this module.

Run from the repository root:

    uv run python labs/hyperparameters/code/leaf_cnn.py            # data, one training
    uv run python labs/hyperparameters/code/leaf_cnn.py --search   # + both searches (~3 min)
"""

import glob
import os
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

DATA_URL = ("https://raw.githubusercontent.com/tomasvicar/MLR-public/master/"
            "exercises/data/ex08_leaves_images.zip")
DATA_ROOT = Path("labs/hyperparameters/data")   # cache, not in version control

IMAGE_SIZE = 48          # the photographs are 96 x 96; smaller = faster training
CLASSES = ["apple", "cherry", "chestnut", "maple"]
N_EPOCHS = 8             # fixed part of the budget, not a tuned hyperparameter


# ------------------------------------------------------------------ the data


def download_leaves(root: Path = DATA_ROOT) -> Path:
    """Download and unzip the photographs once; return the folder with them."""
    root.mkdir(parents=True, exist_ok=True)
    leaves = root / "leaves"
    if not leaves.exists():
        archive = root / "ex08_leaves_images.zip"
        urllib.request.urlretrieve(DATA_URL, archive)
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(root)
    return leaves


def load_split(leaves: Path, split: str):
    """All images of one split as one float tensor (n, 3, size, size) and labels."""
    from PIL import Image

    files = sorted(glob.glob(str(leaves / split / "*" / "*.jpg")))
    size = IMAGE_SIZE
    images = np.stack([
        np.asarray(Image.open(f).convert("RGB").resize((size, size)), dtype=np.float32) / 255.0
        for f in files
    ])
    labels = [CLASSES.index(os.path.basename(os.path.dirname(f))) for f in files]
    return (torch.from_numpy(images).permute(0, 3, 1, 2),
            torch.tensor(labels, dtype=torch.long))


def load_leaves(root: Path = DATA_ROOT):
    """Training, validation and test tensors.

    The archive has two folders, `train` (700 images) and `val` (120). The 120
    are split in half into a **validation** set, which every trial of the search
    sees, and a **test** set, which is looked at once at the very end - the
    tuned validation number is optimistic exactly because the search chose the
    best of many.
    """
    leaves = download_leaves(root)
    x_train, y_train = load_split(leaves, "train")
    x_rest, y_rest = load_split(leaves, "val")

    order = torch.randperm(len(y_rest), generator=torch.Generator().manual_seed(0))
    validation, test = order[:60], order[60:]
    return (x_train, y_train, x_rest[validation], y_rest[validation],
            x_rest[test], y_rest[test])


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


def accuracy(model, x, y, device) -> float:
    model.eval()
    with torch.no_grad():
        return float((model(x.to(device)).argmax(1).cpu() == y).float().mean())


def train_and_validate(x_train, y_train, x_valid, y_valid, *,
                       learning_rate: float = 1e-3, weight_decay: float = 0.0,
                       n_filters: int = 16, dropout: float = 0.0,
                       batch_size: int = 32, n_epochs: int = N_EPOCHS,
                       seed: int = 0, device=None,
                       history: list | None = None) -> float:
    """Train one configuration from scratch and return its validation accuracy.

    **This is one evaluation of the black box.** Five hyperparameters in, one
    number out; everything else - the data, the number of epochs, the seed - is
    held fixed, so two calls differ only by what was tuned.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)

    model = LeafCNN(n_filters=int(n_filters), dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                 weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(x_train, y_train),
                        batch_size=int(batch_size), shuffle=True,
                        generator=torch.Generator().manual_seed(seed))

    for _ in range(int(n_epochs)):
        model.train()
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
        if history is not None:
            history.append(accuracy(model, x_valid, y_valid, device))

    return accuracy(model, x_valid, y_valid, device)


# ------------------------------------------------------- the reference runs


DEFAULT_CONFIG = dict(learning_rate=1e-3, weight_decay=0.0, n_filters=16,
                      dropout=0.0, batch_size=32)

# The search space, written once and read by both searches. The two continuous
# parameters are searched on a log scale (the lecture's point); `bayes_opt`
# knows only boxes of real numbers, so the integer and the categorical one are
# its problem, not Optuna's.
SPACE = {
    "log_learning_rate": (-4.0, -1.5),     # 1e-4 ... 3e-2
    "log_weight_decay": (-6.0, -2.0),      # 1e-6 ... 1e-2
    "n_filters": (8.0, 24.0),              # integer, rounded inside the objective
    "dropout": (0.0, 0.5),
}
BATCH_SIZES = [16, 32, 64]                 # the discrete parameter added in Optuna


def bayes_opt_search(data, n_init=4, n_iter=8, seed=0, verbose=2):
    """`bayes_opt` over the four continuous hyperparameters."""
    from bayes_opt import BayesianOptimization

    x_train, y_train, x_valid, y_valid = data[:4]

    def objective(log_learning_rate, log_weight_decay, n_filters, dropout):
        return train_and_validate(
            x_train, y_train, x_valid, y_valid,
            learning_rate=10**log_learning_rate,
            weight_decay=10**log_weight_decay,
            n_filters=int(round(n_filters)), dropout=dropout)

    optimizer = BayesianOptimization(f=objective, pbounds=SPACE,
                                     random_state=seed, verbose=verbose)
    optimizer.maximize(init_points=n_init, n_iter=n_iter)
    return optimizer


def optuna_search(data, n_trials=12, seed=0):
    """The same space in Optuna, plus the batch size as a discrete parameter."""
    import optuna

    x_train, y_train, x_valid, y_valid = data[:4]

    def objective(trial):
        return train_and_validate(
            x_train, y_train, x_valid, y_valid,
            learning_rate=trial.suggest_float("learning_rate", 1e-4, 3e-2, log=True),
            weight_decay=trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
            n_filters=trial.suggest_int("n_filters", 8, 24),
            dropout=trial.suggest_float("dropout", 0.0, 0.5),
            batch_size=trial.suggest_categorical("batch_size", BATCH_SIZES))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    return study


def main() -> None:
    torch.set_num_threads(min(2, os.cpu_count() or 1))   # about what Colab gives
    data = load_leaves()
    x_train, y_train, x_valid, y_valid, x_test, y_test = data
    print(f"train {len(y_train)}  validation {len(y_valid)}  test {len(y_test)} "
          f"images of {IMAGE_SIZE} x {IMAGE_SIZE}, {len(CLASSES)} classes")
    print("class counts in the training set:",
          {c: int((y_train == i).sum()) for i, c in enumerate(CLASSES)})

    model = LeafCNN(**{k: DEFAULT_CONFIG[k] for k in ("n_filters", "dropout")})
    print(f"parameters of the default network: "
          f"{sum(p.numel() for p in model.parameters())}")

    history: list[float] = []
    start = time.time()
    default = train_and_validate(x_train, y_train, x_valid, y_valid,
                                 history=history, **DEFAULT_CONFIG)
    seconds = time.time() - start
    print(f"\ndefault configuration {DEFAULT_CONFIG}")
    print(f"  validation accuracy after every epoch: "
          f"{' '.join(f'{a:.3f}' for a in history)}")
    print(f"  validation accuracy {default:.3f} in {seconds:.1f} s "
          f"({seconds / N_EPOCHS:.1f} s per epoch)")

    if "--search" not in sys.argv:
        print("\n(run with --search for both searches, about three minutes)")
        return

    start = time.time()
    optimizer = bayes_opt_search(data, verbose=0)
    best = optimizer.max
    print(f"\nbayes_opt, 4 + 8 trainings in {time.time() - start:.0f} s")
    print(f"  best validation accuracy {best['target']:.3f}")
    print(f"  at learning rate {10**best['params']['log_learning_rate']:.2e}, "
          f"weight decay {10**best['params']['log_weight_decay']:.2e}, "
          f"{int(round(best['params']['n_filters']))} filters, "
          f"dropout {best['params']['dropout']:.2f}")

    start = time.time()
    study = optuna_search(data)
    print(f"\noptuna, 12 trials in {time.time() - start:.0f} s")
    print(f"  best validation accuracy {study.best_value:.3f}")
    print(f"  at {study.best_params}")

    on_test = train_and_validate(x_train, y_train, x_test, y_test,
                                 **study.best_params)
    print(f"\nthe winner retrained and measured ONCE on the test set: {on_test:.3f}"
          f"  (validation said {study.best_value:.3f})")


if __name__ == "__main__":
    main()
