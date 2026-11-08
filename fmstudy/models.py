"""Networks shared by every method: an MLP for 2D data, a small U-Net and a classifier for MNIST.

All generative networks take (x, s) with s in [0, 1]. Flow matching uses s = t, and DDPM uses
s = t / (T - 1), so the same architecture serves the four methods.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


def sinusoidal_embedding(s, dim, scale=1000.0):
    """Sinusoidal encoding of the time, as in Ho et al. (2020) and Vaswani et al. (2017)."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=s.device) / half)
    angles = scale * s[:, None].float() * freqs[None]
    return torch.cat([angles.sin(), angles.cos()], dim=1)


class MLP2D(nn.Module):
    """Perceptron with four hidden layers of 256 SiLU units and sinusoidal time encoding."""

    def __init__(self, hidden=256, layers=4, time_dim=64):
        super().__init__()
        self.time_dim = time_dim
        dims = [2 + time_dim] + [hidden] * layers
        blocks = []
        for a, b in zip(dims[:-1], dims[1:]):
            blocks += [nn.Linear(a, b), nn.SiLU()]
        self.net = nn.Sequential(*blocks, nn.Linear(hidden, 2))

    def forward(self, x, s):
        return self.net(torch.cat([x, sinusoidal_embedding(s, self.time_dim)], dim=1))


class ResBlock(nn.Module):
    """Residual block with group normalisation and an additive time embedding."""

    def __init__(self, c_in, c_out, time_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, c_in)
        self.conv1 = nn.Conv2d(c_in, c_out, 3, padding=1)
        self.time = nn.Linear(time_dim, c_out)
        self.norm2 = nn.GroupNorm(8, c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1)
        self.skip = nn.Conv2d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x, emb):
        h = self.conv1(F.silu(self.norm1(x))) + self.time(emb)[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class SmallUNet(nn.Module):
    """U-Net with two resolutions (28 -> 14 -> 7) and skip connections (Ronneberger et al., 2015)."""

    def __init__(self, channels=32, time_dim=128):
        super().__init__()
        c = channels
        self.c = c
        self.time_mlp = nn.Sequential(nn.Linear(c, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim))
        self.inp = nn.Conv2d(1, c, 3, padding=1)
        self.down1 = ResBlock(c, c, time_dim)
        self.pool1 = nn.Conv2d(c, c, 3, stride=2, padding=1)
        self.down2 = ResBlock(c, 2 * c, time_dim)
        self.pool2 = nn.Conv2d(2 * c, 2 * c, 3, stride=2, padding=1)
        self.mid = ResBlock(2 * c, 2 * c, time_dim)
        self.up2 = ResBlock(4 * c, 2 * c, time_dim)
        self.up1 = ResBlock(3 * c, c, time_dim)
        self.out = nn.Conv2d(c, 1, 3, padding=1)

    def forward(self, x, s):
        emb = self.time_mlp(sinusoidal_embedding(s, self.c))
        h1 = self.down1(self.inp(x), emb)
        h2 = self.down2(self.pool1(h1), emb)
        h = self.mid(self.pool2(h2), emb)
        h = self.up2(torch.cat([F.interpolate(h, scale_factor=2.0), h2], 1), emb)
        h = self.up1(torch.cat([F.interpolate(h, scale_factor=2.0), h1], 1), emb)
        return self.out(h)


class MNISTClassifier(nn.Module):
    """Convolutional classifier whose 128-dimensional penultimate layer defines the Fréchet distance."""

    def __init__(self, feature_dim=128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(), nn.Dropout(0.3), nn.Linear(64 * 7 * 7, feature_dim), nn.ReLU(),
        )
        self.head = nn.Linear(feature_dim, 10)

    def forward(self, x, return_features=False):
        f = self.features(x)
        logits = self.head(f)
        return (logits, f) if return_features else logits


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())
