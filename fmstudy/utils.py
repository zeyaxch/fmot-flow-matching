"""Small helpers: seeding, devices, provenance and result files."""
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
RUNS = ROOT / "runs"
DATA = ROOT / "data"


def seed_everything(seed):
    """Fix the Python, NumPy and PyTorch generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device(prefer_gpu=True):
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    if prefer_gpu and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def load_config(name="experiments.json"):
    """Read a configuration file and keep only the values (the sources stay in the file)."""
    with open(ROOT / "configs" / name) as f:
        raw = json.load(f)

    def strip(node):
        if isinstance(node, dict) and "value" in node and "source" in node:
            return node["value"]
        if isinstance(node, dict):
            return {k: strip(v) for k, v in node.items() if k != "_comment"}
        return node

    return strip(raw)


def read_mnist_sweep(results, name, batches, default):
    """Rows of one MNIST result file for every minibatch size, with a batch_size column.

    The run with the minibatch of the configuration writes name.csv and the others name_b<B>.csv.
    """
    frames = []
    for b in batches:
        path = Path(results) / (f"{name}.csv" if b == default else f"{name}_b{b}.csv")
        if path.exists():
            frames.append(pd.read_csv(path).assign(batch_size=b))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def append_rows(path, rows):
    """Append dictionaries to a CSV file, writing the header the first time."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        return
    new = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if new:
            writer.writeheader()
        writer.writerows(rows)


class Timer:
    """Wall-clock timer that waits for the accelerator before reading the clock."""

    def __init__(self, device=None):
        self.device = device

    def __enter__(self):
        if self.device is not None:
            synchronize(self.device)
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if self.device is not None:
            synchronize(self.device)
        self.seconds = time.perf_counter() - self.start
