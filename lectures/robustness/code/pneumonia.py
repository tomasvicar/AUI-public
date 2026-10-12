"""PneumoniaMNIST and a fine-tuned ResNet18: the image running example.

The recipe is the one shared with the Grad-CAM part of the explainability
lab: an ImageNet-pretrained ResNet18, the 28 x 28 X-rays upsampled to
224 x 224, a random subset of 1500 training images, three epochs of Adam.
The lecture figures and the corruption part of the lab both call these
functions, so a number on a slide and in the notebook come from one place.

Corruptions are applied **at the native 28 x 28 resolution, before the
upsampling** - the noise of a real detector is added where the image is
acquired. (The MPA-PRM version corrupted the upsampled 224 x 224 image;
noise on a bicubically smoothed image is unlike anything the model saw in
training and dropped it to the majority class at the lowest severity.)

Run as a script to print the reference numbers:

    MPLBACKEND=Agg uv run python lectures/robustness/code/pneumonia.py
"""

import hashlib
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
from scipy.ndimage import gaussian_filter

URL = "https://zenodo.org/records/10519652/files/pneumoniamnist.npz?download=1"
MD5 = "28209eda62fecd6e6a2d98b1501bb15f"
DATA = Path(__file__).resolve().parents[3] / "labs" / "robustness" / "data"

SEED = 42
N_TRAIN = 1500           # random subset of the 4708 training images
N_EPOCHS = 3
N_EVAL = 200             # fixed random subset of the 624 test images
CLASSES = ["normal", "pneumonia"]

CORRUPTIONS = ["gaussian_noise", "gaussian_blur", "brightness_down", "contrast_down"]
CORRUPTION_LABELS = {"gaussian_noise": "Gaussian noise", "gaussian_blur": "Gaussian blur",
                     "brightness_down": "Brightness down", "contrast_down": "Contrast down"}
SEVERITIES = [0, 1, 2, 3, 4, 5]
# Parameters per severity 1-5, in units of the 28 x 28 image with values in [0, 1].
NOISE_SIGMA = [0.02, 0.04, 0.06, 0.09, 0.12]
BLUR_SIGMA = [0.3, 0.5, 0.7, 1.0, 1.4]       # pixels of the 28 x 28 image
BRIGHTNESS = [0.9, 0.78, 0.65, 0.52, 0.4]
CONTRAST = [0.75, 0.6, 0.45, 0.32, 0.2]


def download(path: Path = DATA / "pneumoniamnist.npz") -> Path:
    """Downloads the 4.2 MB PneumoniaMNIST archive once and checks it."""
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL, path)
    if hashlib.md5(path.read_bytes()).hexdigest() != MD5:
        raise RuntimeError(f"{path} is damaged - delete it and run again")
    return path


def load_pneumonia(path: Path | None = None) -> dict[str, np.ndarray]:
    """Images as float32 arrays (N, 28, 28) in [0, 1], labels as int64 (N,)."""
    archive = np.load(download() if path is None else path)
    data = {}
    for split in ("train", "val", "test"):
        data[f"x_{split}"] = archive[f"{split}_images"].astype(np.float32) / 255.0
        data[f"y_{split}"] = archive[f"{split}_labels"].reshape(-1).astype(np.int64)
    return data


def to_input(images: np.ndarray) -> torch.Tensor:
    """(N, 28, 28) in [0, 1] -> the network input (N, 3, 224, 224), normalized to [-1, 1]."""
    x = torch.from_numpy(np.ascontiguousarray(images, dtype=np.float32)).unsqueeze(1)
    x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
    return ((x - 0.5) / 0.5).repeat(1, 3, 1, 1)


# Brightness and contrast jitter of the augmented model: a random factor per
# image, drawn uniformly - a darker or flatter scanner seen during training.
# (Jittering only half of the images, tried on 12 October 2026, cost the same
# clean accuracy and rescued less.)
AUGMENT_BRIGHTNESS = (0.4, 1.0)
AUGMENT_CONTRAST = (0.2, 1.0)


def augment(images: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    """Every image gets a random brightness factor and a random contrast factor."""
    n = len(images)
    brightness = rng.uniform(*AUGMENT_BRIGHTNESS, size=(n, 1, 1))
    contrast = rng.uniform(*AUGMENT_CONTRAST, size=(n, 1, 1))
    mean = images.mean(axis=(1, 2), keepdims=True)
    out = ((images - mean) * contrast + mean) * brightness
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def train_resnet(x_train: np.ndarray, y_train: np.ndarray,
                 n_train: int = N_TRAIN, n_epochs: int = N_EPOCHS,
                 device: str = "cpu", verbose: bool = True,
                 seed: int = SEED, augmented: bool = False) -> nn.Module:
    """Fine-tunes an ImageNet ResNet18 on a random subset - the shared recipe.
    The subset is always drawn with SEED; `seed` sets the new layer and the
    order of the batches (the members of a deep ensemble differ only in it);
    `augmented` adds the brightness and contrast jitter."""
    torch.manual_seed(seed)
    subset = np.random.RandomState(SEED).choice(len(x_train), size=n_train, replace=False)
    x, y = x_train[subset], torch.from_numpy(y_train[subset])
    jitter = np.random.RandomState(seed)

    model = tv_models.resnet18(weights=tv_models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.to(device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    order = torch.Generator().manual_seed(seed)
    for epoch in range(n_epochs):
        total = 0.0
        for batch in torch.randperm(n_train, generator=order).split(32):
            images = x[batch.numpy()]
            if augmented:
                images = augment(images, jitter)
            inputs = to_input(images).to(device)
            loss = F.cross_entropy(model(inputs), y[batch].to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(batch)
        if verbose:
            print(f"epoch {epoch + 1}: loss {total / n_train:.4f}")
    return model.eval()


MODELS = DATA / "models"


def trained_resnet(data: dict, seed: int = SEED, augmented: bool = False) -> nn.Module:
    """`train_resnet` with a cache on disk (not versioned), so that the figure
    scripts of several blocks train each model once. Delete the folder to
    retrain from scratch."""
    path = MODELS / f"resnet18_seed{seed}{'_augmented' if augmented else ''}.pt"
    if path.is_file():
        model = tv_models.resnet18()
        model.fc = nn.Linear(model.fc.in_features, 2)
        model.load_state_dict(torch.load(path))
        return model.eval()
    model = train_resnet(data["x_train"], data["y_train"], seed=seed, augmented=augmented)
    MODELS.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    return model


def predict(model: nn.Module, images: np.ndarray, device: str = "cpu") -> np.ndarray:
    """Probability of pneumonia for every image, in batches of 64."""
    probabilities = []
    with torch.inference_mode():
        for start in range(0, len(images), 64):
            logits = model(to_input(images[start:start + 64]).to(device))
            probabilities.append(logits.softmax(1)[:, 1].cpu().numpy())
    return np.concatenate(probabilities)


def corrupt(images: np.ndarray, kind: str, severity: int, seed: int = 0) -> np.ndarray:
    """One corruption at one severity (0 = clean) on (N, 28, 28) images in [0, 1]."""
    if severity == 0:
        return images
    level = severity - 1
    if kind == "gaussian_noise":
        rng = np.random.default_rng(seed)
        out = images + rng.normal(0.0, NOISE_SIGMA[level], size=images.shape)
    elif kind == "gaussian_blur":
        out = gaussian_filter(images, sigma=(0, BLUR_SIGMA[level], BLUR_SIGMA[level]))
    elif kind == "brightness_down":
        out = images * BRIGHTNESS[level]
    elif kind == "contrast_down":
        mean = images.mean(axis=(1, 2), keepdims=True)
        out = (images - mean) * CONTRAST[level] + mean
    else:
        raise ValueError(kind)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def evaluation_subset(x_test: np.ndarray, y_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The fixed 200 test images every severity is compared on."""
    index = np.random.RandomState(SEED).choice(len(x_test), size=N_EVAL, replace=False)
    return x_test[index], y_test[index]


def corruption_accuracy(model: nn.Module, images: np.ndarray, labels: np.ndarray,
                        device: str = "cpu") -> dict[str, list[float]]:
    """Accuracy for every corruption at severities 0-5 on the same images."""
    results = {}
    for kind in CORRUPTIONS:
        results[kind] = []
        for severity in SEVERITIES:
            p = predict(model, corrupt(images, kind, severity), device)
            results[kind].append(float(((p > 0.5) == labels).mean()))
    return results


def predict_logits(model: nn.Module, images: np.ndarray, device: str = "cpu") -> np.ndarray:
    """The two raw scores of the network for every image, (N, 2)."""
    logits = []
    with torch.inference_mode():
        for start in range(0, len(images), 64):
            logits.append(model(to_input(images[start:start + 64]).to(device)).cpu().numpy())
    return np.concatenate(logits)


def softmax(logits: np.ndarray, T: float = 1.0) -> np.ndarray:
    z = logits / T
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """T minimizing the negative log-likelihood of softmax(z / T)."""
    from scipy.optimize import minimize_scalar

    def nll(T):
        return float(-np.mean(np.log(softmax(logits, T)[np.arange(len(labels)), labels] + 1e-12)))
    return float(minimize_scalar(nll, bounds=(0.05, 10.0), method="bounded").x)


# ------------------------------------------------------------ conformal sets

ALPHA = 0.1


def conformal_threshold(probabilities: np.ndarray, labels: np.ndarray,
                        alpha: float = ALPHA) -> float:
    """Split conformal prediction: the score is 1 - (probability of the true
    label); the threshold is its ceil((n + 1)(1 - alpha)) / n quantile."""
    scores = 1.0 - probabilities[np.arange(len(labels)), labels]
    n = len(scores)
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(scores, level, method="higher"))


def prediction_sets(probabilities: np.ndarray, threshold: float) -> np.ndarray:
    """Boolean (N, 2): which labels are in the set of each image."""
    return (1.0 - probabilities) <= threshold


def set_summary(sets: np.ndarray, labels: np.ndarray) -> dict:
    size = sets.sum(axis=1)
    return {"coverage": float(sets[np.arange(len(labels)), labels].mean()),
            "single": float((size == 1).mean()), "both": float((size == 2).mean()),
            "empty": float((size == 0).mean())}


# --------------------------------------------- uncertainty on the image model

ENSEMBLE_SEEDS = [42, 43, 44, 45, 46]      # the first member is the shared model
SHIFTS = {"clean": ("clean", 0), "darker": ("brightness_down", 3),
          "flatter": ("contrast_down", 4)}  # 35 % darker, 32 % contrast
FLAG_SHARE = 0.05        # the flag threshold lets 5 % of the validation images through


def two_class(p: np.ndarray) -> np.ndarray:
    """(N,) probability of pneumonia -> (N, 2) probabilities of both labels."""
    return np.stack([1.0 - p, p], axis=1)


def confidence(p: np.ndarray) -> np.ndarray:
    """Maximum softmax probability: the probability of the predicted label."""
    return np.maximum(p, 1.0 - p)


_PREDICTIONS: dict = {}


def ensemble_predictions(data: dict) -> dict:
    """Probability of pneumonia of each ensemble member on the validation split
    and on the 624 test images, clean and under the two intensity shifts.
    Arrays (members, images); computed once per process."""
    if not _PREDICTIONS:
        members = [trained_resnet(data, seed=seed) for seed in ENSEMBLE_SEEDS]
        _PREDICTIONS["val"] = np.stack([predict(m, data["x_val"]) for m in members])
        for name, (kind, severity) in SHIFTS.items():
            images = corrupt(data["x_test"], kind, severity)
            _PREDICTIONS[name] = np.stack([predict(m, images) for m in members])
        model = members[0]
        _PREDICTIONS["logits_val"] = predict_logits(model, data["x_val"])
        _PREDICTIONS["logits_test"] = predict_logits(model, data["x_test"])
    return _PREDICTIONS


def test_halves(labels: np.ndarray, seed: int = SEED) -> tuple[np.ndarray, np.ndarray]:
    """The 624 test images split in two stratified halves: one to calibrate, one to judge."""
    from sklearn.model_selection import train_test_split
    return train_test_split(np.arange(len(labels)), test_size=0.5, random_state=seed,
                            stratify=labels)


def fit_temperature_intercept(z: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """T and b minimizing the negative log-likelihood of sigmoid(z / T + b),
    z the difference of the two logits (Platt scaling in the form of the slides)."""
    from scipy.optimize import minimize

    def nll(v):
        p = np.clip(1.0 / (1.0 + np.exp(-(z / v[0] + v[1]))), 1e-7, 1 - 1e-7)
        return float(-np.mean(labels * np.log(p) + (1 - labels) * np.log(1 - p)))
    T, b = minimize(nll, [1.0, 0.0], method="Nelder-Mead").x
    return float(T), float(b)


def calibration_resnet(data: dict, seed: int = SEED) -> dict:
    """The ResNet18 recalibrated three ways, judged on one half of the test
    split (ECE with 10 bins): T fitted on the validation split; T fitted on the
    other half of the test split; T and the intercept b fitted on that half."""
    from heart import brier, ece
    pred = ensemble_predictions(data)
    y_val, y_test = data["y_val"], data["y_test"]
    fit, judge = test_halves(y_test, seed)
    logits = pred["logits_test"]
    z = logits[:, 1] - logits[:, 0]
    T_val = fit_temperature(pred["logits_val"], y_val)
    T_test = fit_temperature(logits[fit], y_test[fit])
    T_both, b_both = fit_temperature_intercept(z[fit], y_test[fit])
    out = {"T_val": T_val, "T_test": T_test, "T_both": T_both, "b_both": b_both,
           "y": y_test[judge],
           "accuracy_on_val": float((pred["logits_val"].argmax(1) == y_val).mean()),
           "accuracy_on_test": float((logits.argmax(1) == y_test).mean())}
    for name, T, b in [("raw", 1.0, 0.0), ("val", T_val, 0.0), ("test", T_test, 0.0),
                       ("both", T_both, b_both)]:
        p = 1.0 / (1.0 + np.exp(-(z[judge] / T + b)))
        out[f"p_{name}"] = p
        out[f"ece_{name}"] = ece(p, y_test[judge], 10)
        out[f"brier_{name}"] = brier(p, y_test[judge])
        out[f"accuracy_{name}"] = float(((p > 0.5) == y_test[judge]).mean())
    return out


def conformal_study(data: dict, seed: int = SEED) -> dict:
    """Split conformal at ALPHA on the shared model: calibrated on the
    validation split, and on one half of the test split; judged on the other
    half, clean and under the two intensity shifts."""
    pred = ensemble_predictions(data)
    y_test = data["y_test"]
    fit, judge = test_halves(y_test, seed)
    q_val = conformal_threshold(two_class(pred["val"][0]), data["y_val"])
    q_test = conformal_threshold(two_class(pred["clean"][0][fit]), y_test[fit])
    out = {"q_val": q_val, "q_test": q_test}
    for name in SHIFTS:
        p = two_class(pred[name][0][judge])
        out[f"val_{name}"] = set_summary(prediction_sets(p, q_val), y_test[judge])
        out[f"test_{name}"] = set_summary(prediction_sets(p, q_test), y_test[judge])
    return out


def flag_study(data: dict) -> dict:
    """Does the model know it is lost? Two scores - the confidence of the
    shared model and the spread of the five members - with the threshold that
    flags FLAG_SHARE of the validation images; per test condition the share of
    images flagged, the share misclassified and the share of the errors caught."""
    pred = ensemble_predictions(data)
    y = data["y_test"]
    scores = {"confidence": lambda P: -confidence(P[0]),    # low confidence = suspicious
              "spread": lambda P: P.std(axis=0)}
    out = {}
    for score_name, score in scores.items():
        threshold = np.quantile(score(pred["val"]), 1 - FLAG_SHARE)
        for name in SHIFTS:
            P = pred[name]
            p = P[0] if score_name == "confidence" else P.mean(axis=0)
            flagged, wrong = score(P) > threshold, (p > 0.5) != y
            out[(score_name, name)] = {"flagged": float(flagged.mean()), "wrong": float(wrong.mean()),
                                       "caught": float(flagged[wrong].mean()),
                                       "mean_score": float(np.abs(score(P)).mean()),
                                       "score": np.abs(score(P))}
    return out


def referral_curve(p: np.ndarray, labels: np.ndarray, shares=np.linspace(0, 0.6, 61)):
    """Selective prediction: refer the least confident share of the images to a
    clinician, report the accuracy on the images the model keeps."""
    order = np.argsort(-confidence(p), kind="stable")
    correct = ((p > 0.5) == labels)[order]
    kept = [correct[:len(p) - int(round(share * len(p)))].mean() for share in shares]
    return shares, np.array(kept)


def to_input_differentiable(x: torch.Tensor) -> torch.Tensor:
    """`to_input` for a tensor that carries a gradient."""
    x = F.interpolate(x.unsqueeze(1), size=(224, 224), mode="bilinear", align_corners=False)
    return ((x - 0.5) / 0.5).repeat(1, 3, 1, 1)


def pgd(model: nn.Module, image: np.ndarray, label: int, epsilon: float,
        steps: int = 10) -> np.ndarray:
    """Projected gradient descent on the 28 x 28 image (Madry et al. 2018):
    `steps` signed-gradient steps of size epsilon / 4 that increase the loss of
    the true label, each projected back to at most epsilon from the original.
    One step of size epsilon without the projection is FGSM."""
    original = torch.from_numpy(image[None].astype(np.float32))
    x = original.clone()
    for _ in range(steps):
        x.requires_grad_(True)
        loss = F.cross_entropy(model(to_input_differentiable(x)), torch.tensor([label]))
        gradient, = torch.autograd.grad(loss, x)
        x = x.detach() + epsilon / 4 * gradient.sign()
        x = torch.clamp(torch.clamp(x, original - epsilon, original + epsilon), 0.0, 1.0)
    return x[0].numpy()


def smallest_flip(model: nn.Module, image: np.ndarray, label: int,
                  epsilons=(0.002, 0.004, 0.006, 0.008, 0.01, 0.015, 0.02, 0.03)):
    """The smallest epsilon of the list whose PGD attack flips the prediction."""
    for epsilon in epsilons:
        adversarial = pgd(model, image, label, epsilon)
        if (predict(model, adversarial[None])[0] > 0.5) != label:
            return epsilon, adversarial
    return None, None


if __name__ == "__main__":
    import time

    data = load_pneumonia()
    start = time.time()
    model = train_resnet(data["x_train"], data["y_train"])
    print(f"training took {time.time() - start:.0f} s")
    p_test = predict(model, data["x_test"])
    print(f"test accuracy (all 624): {((p_test > 0.5) == data['y_test']).mean():.3f}")
    images, labels = evaluation_subset(data["x_test"], data["y_test"])
    print(f"majority class of the 200: {max(labels.mean(), 1 - labels.mean()):.3f}")
    for kind, accuracies in corruption_accuracy(model, images, labels).items():
        print(f"{kind:16s}", " ".join(f"{a:.3f}" for a in accuracies))
