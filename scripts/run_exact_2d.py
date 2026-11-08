"""Control experiment: Euler on the exact velocity field of flow matching, without any network.

For every mixture and every width sigma0 of the noise, the closed-form field of MixtureVelocity
is integrated with the same steps and the same fixed noise as the trained models. Whatever this
experiment shows cannot come from the minibatch, the learning rate or any other hyperparameter
of training, only from the integrator and the noise.

Usage: python scripts/run_exact_2d.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fmstudy import metrics, samplers  # noqa: E402
from fmstudy.data import get_mixture  # noqa: E402
from fmstudy.methods import MixtureVelocity  # noqa: E402
from fmstudy.utils import RESULTS, append_rows, load_config  # noqa: E402

FIGURE_STEPS = (1, 2, 3, 5, 10, 1000)


def main():
    cfg = load_config()
    toy = cfg["toy"]
    seed = cfg["seeds"][0]
    rows, geo = [], []
    for name in toy["datasets"]:
        mix = get_mixture(name)
        test, _ = mix.sample(toy["n_test"], np.random.default_rng(10_000))
        z = torch.randn(toy["n_test"], 2, generator=torch.Generator().manual_seed(4000 + seed))
        for s0 in toy["sigma0_sweep"]:
            field = MixtureVelocity(mix, s0)
            tag = {"experiment": "exact", "dataset": name, "sigma0": s0, "batch_size": 0,
                   "seed": seed, "method": "exact"}
            fig = {"test": test}
            for steps in cfg["steps_grid"]:
                x = samplers.euler(field, s0 * z, steps).numpy()
                rows.append({**tag, "sampler": "euler", "steps": steps, "w1": metrics.w1_exact(x, test),
                             **metrics.mode_statistics(x, mix), **metrics.placement_statistics(x, mix),
                             "gen_seconds": float("nan")})
                if steps in FIGURE_STEPS:
                    fig[f"samples_exact_euler_{steps}"] = x
            s_value, ratio = samplers.straightness(field, s0 * z, toy["straightness_steps"])
            geo.append({**tag, "straightness": s_value, "length_ratio": ratio})
            _, path, _ = samplers.euler(field, s0 * z[:toy["n_trajectories"]], 100, return_path=True)
            fig["path_exact"] = path.numpy()
            out = RESULTS / "2d_samples"
            out.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(out / f"exact_{name}_s{s0}_b0.npz", **fig)
            print(f"exact {name} sigma0={s0}: done", flush=True)
    append_rows(RESULTS / "2d_metrics.csv", rows)
    append_rows(RESULTS / "2d_straightness.csv", geo)


if __name__ == "__main__":
    main()
