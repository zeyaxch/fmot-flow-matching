"""Datasets: three Gaussian mixtures in 2D and the digits 3 and 4 of MNIST.

The 8- and 25-component mixtures follow the toy datasets of Gulrajani et al. (2017,
code: igul222/improved_wgan_training, gan_toy.py), including their final rescaling.
The 3-component mixture with unequal weights is our own choice.
"""
import gzip
from pathlib import Path

import numpy as np
import torch


class GaussianMixture2D:
    """Isotropic Gaussian mixture in the plane with exact sampling and log-density."""

    def __init__(self, name, means, std, weights):
        self.name = name
        self.means = np.asarray(means, dtype=np.float64)
        self.std = float(std)
        w = np.asarray(weights, dtype=np.float64)
        self.weights = w / w.sum()

    @property
    def n_components(self):
        return len(self.weights)

    def sample(self, n, rng):
        """Ancestral sampling: first the component, then the Gaussian noise."""
        k = rng.choice(self.n_components, size=n, p=self.weights)
        x = self.means[k] + self.std * rng.standard_normal((n, 2))
        return x.astype(np.float32), k


def eight_gaussians():
    scale = 2.0
    c = 1 / np.sqrt(2)
    centers = [(1, 0), (-1, 0), (0, 1), (0, -1), (c, c), (c, -c), (-c, c), (-c, -c)]
    means = scale * np.asarray(centers) / 1.414
    return GaussianMixture2D("8gaussians", means, 0.02 / 1.414, np.ones(8))


def twentyfive_gaussians():
    means = [(2.0 * x, 2.0 * y) for x in range(-2, 3) for y in range(-2, 3)]
    return GaussianMixture2D("25gaussians", np.asarray(means) / 2.828, 0.05 / 2.828, np.ones(25))


def three_unequal():
    angles = np.deg2rad([90, 210, 330])
    means = 1.5 * np.stack([np.cos(angles), np.sin(angles)], 1)
    return GaussianMixture2D("3unequal", means, 0.1, [12, 3, 1])  # weights 3/4, 3/16, 1/16


MIXTURES = {"8gaussians": eight_gaussians, "25gaussians": twentyfive_gaussians,
            "3unequal": three_unequal}


def get_mixture(name):
    return MIXTURES[name]()


_MNIST_FILES = {
    "train": ("train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz"),
    "test": ("t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"),
}


def load_mnist(root, split="train"):
    """Read the original IDX files with NumPy and scale the pixels to [-1, 1]."""
    root = Path(root)
    img_file, lab_file = _MNIST_FILES[split]
    with gzip.open(root / img_file, "rb") as f:
        images = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 1, 28, 28)
    with gzip.open(root / lab_file, "rb") as f:
        labels = np.frombuffer(f.read(), np.uint8, offset=8)
    x = torch.from_numpy(images.astype(np.float32) / 127.5 - 1.0)
    y = torch.from_numpy(labels.astype(np.int64))
    return x, y


def mnist_partitions(root, digits, val_fraction, seed):
    """Train, validation and test partitions of the chosen digits (LeCun et al., 1998).

    The official training set of the chosen digits is split at random into training and
    validation, and the official test set of those digits is the test partition. The function
    also returns the indices of the validation images within the official training set, so that
    the classifier never sees them.
    """
    x_all, y_all = load_mnist(root, "train")
    x_test_all, y_test_all = load_mnist(root, "test")
    keep = torch.isin(y_all, torch.tensor(digits))
    idx = torch.nonzero(keep).flatten()
    perm = idx[torch.randperm(len(idx), generator=torch.Generator().manual_seed(seed))]
    n_val = int(round(val_fraction * len(perm)))
    val_idx, train_idx = perm[:n_val].sort().values, perm[n_val:].sort().values
    keep_test = torch.isin(y_test_all, torch.tensor(digits))
    return {
        "train": (x_all[train_idx], y_all[train_idx]),
        "validation": (x_all[val_idx], y_all[val_idx]),
        "test": (x_test_all[keep_test], y_test_all[keep_test]),
        "validation_index": val_idx,
    }
