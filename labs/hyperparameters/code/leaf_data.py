"""Leaf images: a custom Dataset, standard DataLoaders and tunable transforms.

Follows the Dataset example in MLR exercises/ex08/ex08_teacher.ipynb, task 5.
This module is embedded verbatim in the standalone lab notebook by its builder.
"""

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode, v2

CLASSES = ["apple", "cherry", "chestnut", "maple"]
IMAGE_SIZE = 48


class LeafDataset(Dataset):
    """Read one RGB image and label; apply the transform on every access."""

    def __init__(self, samples, transform=None):
        self.samples = list(samples)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def split_leaf_samples(leaves):
    """Fixed train/validation/test file lists, independent of trial transforms."""
    leaves = Path(leaves)
    class_to_index = {name: index for index, name in enumerate(CLASSES)}

    def samples(split):
        return [(path, class_to_index[path.parent.name])
                for path in sorted((leaves / split).glob("*/*.jpg"))]

    training, held_out = samples("train"), samples("val")
    if len(training) != 700 or len(held_out) != 120:
        raise ValueError("Expected 700 training and 120 held-out leaf images")
    order = torch.randperm(120, generator=torch.Generator().manual_seed(0)).tolist()
    return training, [held_out[i] for i in order[:60]], [held_out[i] for i in order[60:]]


def leaf_transform(training=False, rotation_deg=0.0, color_jitter=0.0):
    """Two training-only strengths; zero disables the corresponding transform.

    Rotation is uniform in [-rotation_deg, +rotation_deg]. Brightness and
    contrast factors are sampled independently in [1-color_jitter, 1+color_jitter].
    Fixed normalization maps [0, 1] to [-1, 1]; no statistics use held-out data.
    """
    operations = [v2.Resize((IMAGE_SIZE, IMAGE_SIZE), antialias=True)]
    if training:
        if rotation_deg > 0:
            operations.append(v2.RandomRotation(rotation_deg,
                              interpolation=InterpolationMode.BILINEAR, fill=(128, 128, 128)))
        if color_jitter > 0:
            operations.append(v2.ColorJitter(brightness=color_jitter, contrast=color_jitter))
    operations += [v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
                   v2.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))]
    return v2.Compose(operations)


def make_loaders(train_samples, valid_samples, batch_size=32,
                 rotation_deg=0.0, color_jitter=0.0, seed=0):
    """Fresh transforms/loaders per trial; only training is augmented/shuffled.

    num_workers=0 keeps this small teaching example portable in Colab and on
    Windows. The caller seeds torch before training, including random transforms.
    Separate generators prevent validation iteration from changing training RNG.
    """
    train_dataset = LeafDataset(train_samples, leaf_transform(True, rotation_deg, color_jitter))
    valid_dataset = LeafDataset(valid_samples, leaf_transform())
    train_loader = DataLoader(train_dataset, batch_size=int(batch_size), shuffle=True,
                              num_workers=0, generator=torch.Generator().manual_seed(seed))
    valid_loader = DataLoader(valid_dataset, batch_size=int(batch_size), shuffle=False,
                              num_workers=0, generator=torch.Generator().manual_seed(seed))
    return train_loader, valid_loader
