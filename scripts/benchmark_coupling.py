"""Cost of the minibatch OT coupling of FM-OT as a function of the minibatch size.

The training times of the experiments include the network and depend on whatever else runs on
the machine, so this script measures the assignment alone, on one thread, with the data of the
study: points of the eight Gaussians (dimension 2) and MNIST images of 3 and 4 (dimension 784),
each paired with standard Gaussian noise. Run it on an otherwise idle machine.

Usage: python scripts/benchmark_coupling.py [--repeats 20]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fmstudy.data import get_mixture, mnist_partitions  # noqa: E402
from fmstudy.methods import ot_coupling  # noqa: E402
from fmstudy.utils import DATA, RESULTS, append_rows, load_config  # noqa: E402


def main():
    cfg = load_config()
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--results", default=str(RESULTS))
    args = parser.parse_args()
    torch.set_num_threads(1)
    rng = np.random.default_rng(0)
    sources = {"8gaussians": lambda n: torch.from_numpy(get_mixture("8gaussians").sample(n, rng)[0])}
    if (DATA / "mnist").exists():
        mn = cfg["mnist"]
        x_train, _ = mnist_partitions(DATA / "mnist", mn["digits"], mn["val_fraction"], mn["split_seed"])["train"]
        sources["mnist34"] = lambda n: x_train[torch.from_numpy(rng.choice(len(x_train), n, replace=False))]
    sizes = sorted(set(cfg["toy"]["batch_sweep"]) | {cfg["mnist"]["batch_size"]})
    rows = []
    for name, draw in sources.items():
        for b in sizes:
            times = []
            for _ in range(args.repeats):
                x1 = draw(b)
                x0 = torch.randn_like(x1)
                start = time.perf_counter()
                ot_coupling(x0, x1)
                times.append(time.perf_counter() - start)
            ms = 1000 * float(np.median(times))
            rows.append({"data": name, "dim": int(x1[0].numel()), "batch_size": b, "repeats": args.repeats,
                         "median_ms": ms, "us_per_point": 1000 * ms / b})
            print(f"{name} B={b}: {ms:.2f} ms per assignment", flush=True)
    append_rows(Path(args.results) / "coupling_benchmark.csv", rows)


if __name__ == "__main__":
    main()
