"""PneumoniaMNIST, a fine-tuned ResNet18 and four ways to explain it.

The data and the recipe are the ones of `lectures/robustness/code/pneumonia.py`
(copied, not imported, so that each lecture's code stands alone when it is
published): an ImageNet-pretrained ResNet18, the 28 x 28 X-rays upsampled to
224 x 224, a random subset of 1500 training images, three epochs of Adam.

The explanations are written by hand (decision of the author, 24 September
2026 - Grad-CAM without `pytorch-grad-cam`):

    grad_cam            Selvaraju et al. (2017), last block of layer4
    vanilla_gradient    |d score / d input|, Simonyan et al. (2014)
    guided_backprop     Springenberg et al. (2015): the ReLUs pass back only
                        positive gradients at positive inputs
    occlusion           Zeiler & Fergus (2014): a grey patch slid over the
                        28 x 28 image, the drop of the class log-odds

`randomized_copy` re-initializes layer4 and the classifier head - the model
randomization test of Adebayo et al. (2018).

Run as a script to print the reference numbers:

    MPLBACKEND=Agg uv run python lectures/explainability/code/pneumonia.py
"""

import copy
import hashlib
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models

URL = "https://zenodo.org/records/10519652/files/pneumoniamnist.npz?download=1"
MD5 = "28209eda62fecd6e6a2d98b1501bb15f"
DATA = Path(__file__).resolve().parents[3] / "labs" / "robustness" / "data"

SEED = 42
N_TRAIN = 1500           # random subset of the 4708 training images
N_EPOCHS = 3
CLASSES = ["normal", "pneumonia"]


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


def train_resnet(x_train: np.ndarray, y_train: np.ndarray,
                 n_train: int = N_TRAIN, n_epochs: int = N_EPOCHS,
                 device: str = "cpu", verbose: bool = True) -> nn.Module:
    """Fine-tunes an ImageNet ResNet18 on a random subset - the shared recipe."""
    torch.manual_seed(SEED)
    subset = np.random.RandomState(SEED).choice(len(x_train), size=n_train, replace=False)
    x, y = x_train[subset], torch.from_numpy(y_train[subset])

    model = tv_models.resnet18(weights=tv_models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.to(device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    order = torch.Generator().manual_seed(SEED)
    for epoch in range(n_epochs):
        total = 0.0
        for batch in torch.randperm(n_train, generator=order).split(32):
            inputs = to_input(x[batch.numpy()]).to(device)
            loss = F.cross_entropy(model(inputs), y[batch].to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(batch)
        if verbose:
            print(f"epoch {epoch + 1}: loss {total / n_train:.4f}")
    return model.eval()


def predict(model: nn.Module, images: np.ndarray, device: str = "cpu") -> np.ndarray:
    """Probability of pneumonia for every image, in batches of 64."""
    probabilities = []
    with torch.inference_mode():
        for start in range(0, len(images), 64):
            logits = model(to_input(images[start:start + 64]).to(device))
            probabilities.append(logits.softmax(1)[:, 1].cpu().numpy())
    return np.concatenate(probabilities)


def to_input_differentiable(x: torch.Tensor) -> torch.Tensor:
    """`to_input` for a (N, 28, 28) tensor that carries a gradient."""
    x = F.interpolate(x.unsqueeze(1), size=(224, 224), mode="bilinear", align_corners=False)
    return ((x - 0.5) / 0.5).repeat(1, 3, 1, 1)


def grad_cam(model: nn.Module, image: np.ndarray, target: int) -> np.ndarray:
    """Grad-CAM of one 28 x 28 image for class `target`, a 224 x 224 map in [0, 1]:

        alpha_k = mean over the map of d y_target / d A_k,   map = ReLU(sum_k alpha_k A_k)
    """
    stored = {}
    hook = model.layer4[-1].register_forward_hook(
        lambda module, inputs, output: stored.update(A=output))
    score = model(to_input(image[None]))[0, target]
    hook.remove()
    A = stored["A"]                                     # (1, 512, 7, 7)
    gradient, = torch.autograd.grad(score, A)
    alpha = gradient.mean(dim=(2, 3), keepdim=True)     # one weight per feature map
    cam = F.relu((alpha * A).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)[0, 0]
    cam = cam.detach().numpy()
    return cam / cam.max() if cam.max() > 0 else cam


def vanilla_gradient(model: nn.Module, image: np.ndarray, target: int) -> np.ndarray:
    """|d y_target / d x| on the 28 x 28 image, scaled to [0, 1]."""
    x = torch.from_numpy(image[None].astype(np.float32)).requires_grad_(True)
    score = model(to_input_differentiable(x))[0, target]
    gradient, = torch.autograd.grad(score, x)
    saliency = gradient[0].abs().numpy()
    return saliency / saliency.max() if saliency.max() > 0 else saliency


class _GuidedReLU(torch.autograd.Function):
    """A ReLU whose backward pass lets through only positive gradients at
    positive inputs - the whole of guided backpropagation."""

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        return grad_output * (x > 0) * (grad_output > 0)


class _GuidedReLUModule(nn.Module):
    def forward(self, x):
        return _GuidedReLU.apply(x)


def _guided(model: nn.Module) -> nn.Module:
    """A copy of the model with every ReLU replaced by the guided one."""
    guided = copy.deepcopy(model)
    for module in list(guided.modules()):
        for name, child in module.named_children():
            if isinstance(child, nn.ReLU):
                setattr(module, name, _GuidedReLUModule())
    return guided.eval()


def guided_backprop(model: nn.Module, image: np.ndarray, target: int) -> np.ndarray:
    """|guided gradient| on the 28 x 28 image, scaled to [0, 1]."""
    return vanilla_gradient(_guided(model), image, target)


def log_odds(model: nn.Module, images: np.ndarray, device: str = "cpu") -> np.ndarray:
    """The score of pneumonia against normal, logit_1 - logit_0, for every image."""
    scores = []
    with torch.inference_mode():
        for start in range(0, len(images), 64):
            logits = model(to_input(images[start:start + 64]).to(device))
            scores.append((logits[:, 1] - logits[:, 0]).cpu().numpy())
    return np.concatenate(scores)


def occlusion(model: nn.Module, image: np.ndarray, target: int,
              patch: int = 4, stride: int = 2) -> tuple[np.ndarray, float]:
    """Slide a grey patch (value 0.5) over the 28 x 28 image and record the drop
    of the class score at every position: 13 x 13 = 169 forward passes.

    The score is the log-odds of the target class, not its probability: for
    a confident prediction the probability is saturated at 1.000 and no
    single patch moves it. Returns the map of drops (28 x 28, averaged where
    patches overlap) and the score of the intact image."""
    positions = range(0, 28 - patch + 1, stride)
    batch, corners = [], []
    for i in positions:
        for j in positions:
            occluded = image.copy()
            occluded[i:i + patch, j:j + patch] = 0.5
            batch.append(occluded)
            corners.append((i, j))
    sign = 1.0 if target == 1 else -1.0
    scores = sign * log_odds(model, np.stack(batch))
    score0 = sign * float(log_odds(model, image[None])[0])
    drop, count = np.zeros((28, 28)), np.zeros((28, 28))
    for (i, j), s in zip(corners, scores):
        drop[i:i + patch, j:j + patch] += score0 - s
        count[i:i + patch, j:j + patch] += 1
    return drop / np.maximum(count, 1), score0


def randomized_copy(model: nn.Module, seed: int = SEED) -> nn.Module:
    """The model with layer4 and the classifier head re-initialized - their
    learned weights replaced by the ones a fresh network starts from."""
    torch.manual_seed(seed)
    random = copy.deepcopy(model)
    for module in list(random.layer4.modules()) + [random.fc]:
        if hasattr(module, "reset_parameters"):
            module.reset_parameters()
        if isinstance(module, nn.BatchNorm2d):
            module.reset_running_stats()
    return random.eval()


def rank_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation of two maps - the similarity measure of Adebayo et al."""
    from scipy.stats import spearmanr
    return float(spearmanr(a.ravel(), b.ravel()).statistic)


def example_images(x_test: np.ndarray, y_test: np.ndarray) -> dict[str, int]:
    """One test X-ray per class, drawn with the seed - the two images of the slides."""
    rng = np.random.RandomState(SEED)
    return {name: int(np.flatnonzero(y_test == label)[rng.randint((y_test == label).sum())])
            for name, label in (("pneumonia", 1), ("normal", 0))}


if __name__ == "__main__":
    data = load_pneumonia()
    model = train_resnet(data["x_train"], data["y_train"])
    p = predict(model, data["x_test"])
    print(f"test accuracy (all 624): {((p > 0.5) == data['y_test']).mean():.3f}")
    for name, index in example_images(data["x_test"], data["y_test"]).items():
        print(f"{name}: test image {index}, P(pneumonia) = {p[index]:.3f}")
